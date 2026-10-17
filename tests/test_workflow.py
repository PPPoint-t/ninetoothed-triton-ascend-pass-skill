import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import engine
import pytest
from device_probe import validate_driver_target
from mdio import (
    WorkflowError,
    atomic_write,
    fields,
    lock,
    read_table,
    table,
    write_fields,
)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    source, target, policy = (tmp_path / name for name in ("old", "new", "policy"))
    for path in (
        source / ".git",
        source / "src",
        source / "tests",
        target / "src",
        target / "tests",
        policy / "compiler-pass-pipeline",
    ):
        path.mkdir(parents=True)
    atomic_write(policy / "compiler-pass-pipeline/SKILL.md", "规范\n")
    atomic_write(target / "tests/test_operator.py", "def test_add(): pass\n")
    atomic_write(target / "src/compute.py", "value = 1\n")
    manifest = tmp_path / "算子.md"
    atomic_write(
        manifest,
        table(
            engine.MANIFEST_HEADERS,
            (
                ("源码", "融合数学", "*", "tests/test_operator.py::test_add"),
                ("设备", "融合数学", "*", "tests/test_operator.py::test_add"),
            ),
        ),
    )
    monkeypatch.setattr(engine, "MANIFEST", manifest)
    monkeypatch.setattr(engine, "git", lambda *args: "test-head")
    monkeypatch.setattr(engine, "environment", lambda: {"Python": "test"})
    monkeypatch.setenv("NT_PASS_LOCK_ROOT", str(tmp_path / "设备锁"))
    return engine.initialize(tmp_path / "batch", source, target, policy)


def plan(batch, layer="源码", nodes=None):
    cfg = engine.config(batch)
    nodes = nodes or [
        ("tests/test_operator.py::test_add[triton-ascend]", "triton-ascend", "融合数学")
    ]
    p = batch / f"计划-{layer}.md"
    write_fields(
        p,
        "测试计划",
        {
            "验证层": layer,
            "回归分组": "核心",
            "附加筛选": "无",
            "请求后端": "triton-ascend",
            "测试摘要": engine.tree_digest(cfg["目标仓库"], "tests"),
            "清单摘要": engine.digest(Path(cfg["清单路径"]).read_bytes()),
            "计划节点数": len(nodes),
        },
        table(engine.PLAN_HEADERS, nodes),
    )
    return p


def fake_execution(monkeypatch, status="通过", reason="", fatal=False):
    def execute(command, cwd, env, log, timeout, **kwargs):
        atomic_write(log, "aicore execution times out 507014" if fatal else "日志\n")
        if "NT_PASS_EVIDENCE" in env:
            write_fields(
                Path(env["NT_PASS_EVIDENCE"]) / "观测.md",
                "观测",
                {
                    "测试结论": status,
                    "失败原因": reason,
                },
            )
        return engine.ProcessResult(0 if status == "通过" else 1, False, False, 0.01)

    monkeypatch.setattr(engine, "bounded_process", execute)

    def healthy(run, device, **kwargs):
        p = Path(run) / "fake-probe"
        write_fields(p / "健康.md", "模拟健康检查", {"芯片": "模拟910B4"})
        return p

    monkeypatch.setattr(engine, "health_probe", healthy)


def test_markdown_roundtrip_and_atomic_replacement(tmp_path):
    p = tmp_path / "中文.md"
    write_fields(p, "标题", {"内容": "中文|换行\n<&>"})
    assert fields(p)["内容"] == "中文|换行\n<&>"
    write_fields(p, "新标题", {"内容": "替换"})
    assert fields(p)["内容"] == "替换"
    assert list(tmp_path.glob(".写入-*")) == []


def test_bad_markdown_fails_closed(tmp_path):
    p = tmp_path / "坏.md"
    atomic_write(
        p,
        table(
            ("项目", "内容"),
            (("记录格式版本", "1"), ("状态", "通过"), ("状态", "失败")),
        ),
    )
    with pytest.raises(WorkflowError, match="重复字段"):
        fields(p)


def test_lock_excludes_other_process(tmp_path):
    p = tmp_path / "设备.lock"
    command = [
        sys.executable,
        "-c",
        "from mdio import lock; import sys\nwith lock(sys.argv[1]): pass",
        str(p),
    ]
    env = {**os.environ, "PYTHONPATH": str(engine.SCRIPTS)}
    with lock(p):
        result = subprocess.run(command, env=env, capture_output=True, timeout=5)
        assert result.returncode != 0
    assert (
        subprocess.run(command, env=env, capture_output=True, timeout=5).returncode == 0
    )


def test_timeout_kills_owned_process_group(tmp_path):
    log = tmp_path / "超时.log"
    result = engine.bounded_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        os.environ.copy(),
        log,
        0.1,
    )
    assert result.timed_out and result.returncode < 0 and result.elapsed < 4


def test_signal_is_preserved(tmp_path):
    result = engine.bounded_process(
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        tmp_path,
        os.environ.copy(),
        tmp_path / "signal.log",
        5,
    )
    assert result.returncode == -signal.SIGTERM


def test_scope_rejects_external_and_symlink(batch, tmp_path):
    cfg = engine.config(batch)
    external = tmp_path / "outside.py"
    atomic_write(external, "def test_x(): pass\n")
    (Path(cfg["目标仓库"]) / "tests/link.py").symlink_to(external)
    for node in ("../outside.py::test_x", "tests/link.py::test_x", "tests"):
        with pytest.raises(WorkflowError):
            engine.validate_selector(cfg["目标仓库"], node)


def test_collect_zero_is_error(batch, monkeypatch):
    def collect(command, cwd, env, log, timeout):
        atomic_write(log, "collected")
        atomic_write(
            env["NT_PASS_COLLECTION"],
            table(
                engine.PLAN_HEADERS,
                (("tests/test_operator.py::test_add[triton]", "triton", "融合数学"),),
            ),
        )
        return engine.ProcessResult(0, False, False, 0)

    monkeypatch.setattr(engine, "bounded_process", collect)
    with pytest.raises(WorkflowError, match="零个算子"):
        engine.collect(batch, "源码", "核心", "triton-ascend")


def test_execution_and_comparison_preserve_success(batch, monkeypatch):
    fake_execution(monkeypatch)
    p = plan(batch)
    before = engine.execute(batch, p, "基线")
    after = engine.execute(batch, p, "修改后")
    comparison = engine.compare_runs(batch, before, after)
    assert fields(comparison)["门禁"] == "通过"
    assert fields(after / "记录.md")["通过数"] == "1"


@pytest.mark.parametrize(
    "after_status,reason", (("失败", "新数值错误"), ("跳过", ""), ("未执行", ""))
)
def test_pass_cannot_turn_into_failure_or_missing(
    batch, monkeypatch, after_status, reason
):
    p = plan(batch)
    fake_execution(monkeypatch)
    before = engine.execute(batch, p, "基线")
    fake_execution(monkeypatch, after_status, reason)
    after = engine.execute(batch, p, "修改后")
    comparison = engine.compare_runs(batch, before, after)
    assert fields(comparison)["新增回归数"] == "1"


def test_changed_failure_not_treated_as_known(batch, monkeypatch):
    p = plan(batch)
    fake_execution(monkeypatch, "失败", "旧错误")
    before = engine.execute(batch, p, "基线")
    fake_execution(monkeypatch, "失败", "不同错误")
    after = engine.execute(batch, p, "修改后")
    assert fields(engine.compare_runs(batch, before, after))["门禁"] == "失败"


def test_known_failure_remains_visible(batch, monkeypatch):
    p = plan(batch)
    fake_execution(monkeypatch, "失败", "相同错误")
    before = engine.execute(batch, p, "基线")
    after = engine.execute(batch, p, "修改后")
    comparison = fields(engine.compare_runs(batch, before, after))
    assert comparison["门禁"] == "通过" and comparison["原有失败数"] == "1"


def test_environment_change_blocks_comparison(batch, monkeypatch):
    p = plan(batch)
    fake_execution(monkeypatch)
    before = engine.execute(batch, p, "基线")
    monkeypatch.setattr(engine, "environment", lambda: {"compiler": "changed"})
    after = engine.execute(batch, p, "修改后")
    with pytest.raises(WorkflowError, match="环境摘要"):
        engine.compare_runs(batch, before, after)


def test_test_change_invalidates_plan(batch, monkeypatch):
    p = plan(batch)
    cfg = engine.config(batch)
    atomic_write(
        Path(cfg["目标仓库"]) / "tests/test_operator.py",
        "def test_add(): assert True\n",
    )
    with pytest.raises(WorkflowError, match="测试或配置已变化"):
        engine.execute(batch, p, "基线")


def test_fatal_device_error_stops_queue(batch, monkeypatch):
    fake_execution(monkeypatch, "失败", "设备错误", fatal=True)
    p = plan(
        batch,
        "设备",
        [
            (
                "tests/test_operator.py::test_add[triton-ascend-1]",
                "triton-ascend",
                "融合数学",
            ),
            (
                "tests/test_operator.py::test_add[triton-ascend-2]",
                "triton-ascend",
                "融合数学",
            ),
        ],
    )
    directory = engine.execute(batch, p, "基线")
    results = read_table(directory / "记录.md", engine.RESULT_HEADERS)
    assert [r["结果"] for r in results] == ["失败", "阻塞"]
    assert fields(engine.device_paths("0")[1])["状态"] == "隔离"


def test_missing_observation_is_failure(tmp_path):
    atomic_write(tmp_path / "log", "")
    result = engine.ProcessResult(0, False, False, 0)
    assert (
        engine.assess_case(result, tmp_path / "missing", tmp_path / "log", "源码")[0]
        == "失败"
    )


def test_incomplete_execution_cannot_compare(batch, monkeypatch):
    fake_execution(monkeypatch)
    p = plan(batch)
    before = engine.execute(batch, p, "基线")
    after = engine.execute(batch, p, "修改后")
    state = fields(after / "记录.md")
    write_fields(after / "记录.md", "模拟进程中断", {**state, "状态": "执行中"})
    with pytest.raises(WorkflowError, match="未完整结束"):
        engine.compare_runs(batch, before, after)
    with pytest.raises(WorkflowError, match="遗留执行中"):
        engine.execute(batch, p, "修改后")
    engine.recover(batch)
    assert fields(after / "记录.md")["状态"] == "中断待核对"


def test_only_one_active_rule_and_no_finish_without_regression(batch):
    record = batch / "规则记录/M01.md"
    atomic_write(
        record, record.read_text() + "\n## 语义分析\n\n已有人工判断，必须保留。\n"
    )
    engine.begin_rule(batch, "M01")
    with pytest.raises(WorkflowError, match="当前规则"):
        engine.begin_rule(batch, "M02")
    with pytest.raises(WorkflowError, match="核心回归"):
        engine.finish_rule(batch, "M01", "已有覆盖", "不能只靠分析", [])
    engine.finish_rule(batch, "M01", "暂缓", "保留问题供后续处理", [])
    assert fields(batch / "规则记录/M01.md")["状态"] == "暂缓"
    assert "已有人工判断，必须保留。" in record.read_text()


def test_three_source_revision_limit(batch, monkeypatch):
    p = plan(batch)
    fake_execution(monkeypatch)
    engine.begin_rule(batch, "M01")
    cfg = engine.config(batch)
    for revision in range(3):
        atomic_write(Path(cfg["目标仓库"]) / "src/compute.py", f"value = {revision}\n")
        engine.execute(batch, p, "修改后")
    atomic_write(Path(cfg["目标仓库"]) / "src/compute.py", "value = 4\n")
    with pytest.raises(WorkflowError, match="三次"):
        engine.execute(batch, p, "修改后")


def test_other_device_owner_does_not_poison_device(batch, monkeypatch):
    p = plan(batch, "设备")
    with lock(engine.device_paths("0")[0]):
        with pytest.raises(WorkflowError, match="占用锁"):
            engine.execute(batch, p, "基线")
    assert not engine.device_paths("0")[1].exists()


@pytest.mark.parametrize("backend", ("npu", "ascend"))
def test_actual_ascend_driver_target_names(backend):
    validate_driver_target(SimpleNamespace(backend=backend, arch="Ascend910B4-1"))
    with pytest.raises(RuntimeError, match="不是 Ascend"):
        validate_driver_target(SimpleNamespace(backend="cuda", arch="sm_90"))


def test_environment_failure_does_not_quarantine_hardware(batch, monkeypatch):
    def missing_driver(run, device, **kwargs):
        raise WorkflowError("未安装 driver，未提交 kernel")

    monkeypatch.setattr(engine, "health_probe", missing_driver)
    with pytest.raises(WorkflowError, match="未安装"):
        engine.execute(batch, plan(batch, "设备"), "基线")
    assert not engine.device_paths("0")[1].exists()
    row = read_table(batch / "执行记录/执行-0001/记录.md", engine.RESULT_HEADERS)[0]
    assert row["结果"] == "阻塞"


def test_inherited_device_lock_survives_owner_crash(tmp_path):
    path = tmp_path / "继承.lock"
    owner = (
        "import os,subprocess,sys\nfrom mdio import lock\n"
        "with lock(sys.argv[1]) as f:\n"
        " subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)'],"
        "pass_fds=(f.fileno(),),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        " os._exit(0)\n"
    )
    subprocess.run(
        [sys.executable, "-c", owner, str(path)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(engine.SCRIPTS)},
        timeout=5,
    )
    with pytest.raises(WorkflowError, match="占用锁"):
        with lock(path):
            pass
    import time

    deadline = time.monotonic() + 5
    while True:
        try:
            with lock(path):
                break
        except WorkflowError:
            assert time.monotonic() < deadline
            time.sleep(0.05)


def test_unknown_inflight_device_blocks_other_batches(batch):
    write_fields(engine.device_paths("0")[1], "结果未知", {"状态": "执行中"})
    with pytest.raises(WorkflowError, match="设备已隔离"):
        engine.health_probe(batch, "0")


@pytest.mark.parametrize("fatal", (False, True))
def test_probe_distinguishes_environment_and_fatal_errors(batch, monkeypatch, fatal):
    def probe(command, cwd, env, log, timeout, **kwargs):
        atomic_write(log, "507014" if fatal else "driver is not installed")
        return engine.ProcessResult(1, False, False, 0)

    monkeypatch.setattr(engine, "bounded_process", probe)
    with pytest.raises(WorkflowError, match="健康检查未通过"):
        engine.health_probe(batch, "0")
    assert engine.device_paths("0")[1].exists() == fatal


def test_status_is_read_only_while_writer_holds_lock(batch, capsys):
    import workflow

    with lock(batch / "主控.lock"):
        assert workflow.main(["status", "--batch", str(batch)]) == 0
    assert "规则进度" in capsys.readouterr().out


def test_stubborn_process_does_not_hang_controller(tmp_path, monkeypatch):
    class Stubborn:
        pid = 9999999
        returncode = None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("stubborn", timeout)

    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: Stubborn())
    monkeypatch.setattr(engine.os, "killpg", lambda *args: None)
    result = engine.bounded_process(["ignored"], tmp_path, {}, tmp_path / "log", 0.1)
    assert result.timed_out and result.returncode is None
    assert result.elapsed < 1


def test_residual_process_blocks_even_explicit_recovery(batch):
    write_fields(
        engine.device_paths("0")[1],
        "残留宿主进程",
        {
            "状态": "隔离",
            "残留子进程PID": os.getpid(),
            "残留进程启动标识": engine.process_identity(os.getpid()),
        },
    )
    with pytest.raises(WorkflowError, match="残留进程"):
        engine.health_probe(batch, "0", clear=True)


def test_rerun_cannot_hide_earlier_new_failure(batch, monkeypatch):
    engine.begin_rule(batch, "M01")
    p = plan(batch)
    fake_execution(monkeypatch)
    before = engine.execute(batch, p, "基线")
    fake_execution(monkeypatch, "失败", "偶发新失败")
    engine.execute(batch, p, "修改后")
    fake_execution(monkeypatch)
    after = engine.execute(batch, p, "修改后")
    comparison = engine.compare_runs(batch, before, after)
    assert fields(comparison)["门禁"] == "通过"
    with pytest.raises(WorkflowError, match="不能用重跑通过掩盖"):
        engine.reject_hidden_retry_failures(
            batch, "M01", [comparison], fields(after / "记录.md")["源码摘要"]
        )


def test_old_source_failure_does_not_block_actual_fix(batch, monkeypatch):
    engine.begin_rule(batch, "M01")
    p = plan(batch)
    fake_execution(monkeypatch)
    before = engine.execute(batch, p, "基线")
    fake_execution(monkeypatch, "失败", "旧实现问题")
    engine.execute(batch, p, "修改后")
    cfg = engine.config(batch)
    atomic_write(Path(cfg["目标仓库"]) / "src/compute.py", "value = 2\n")
    fake_execution(monkeypatch)
    after = engine.execute(batch, p, "修改后")
    comparison = engine.compare_runs(batch, before, after)
    engine.reject_hidden_retry_failures(
        batch, "M01", [comparison], fields(after / "记录.md")["源码摘要"]
    )

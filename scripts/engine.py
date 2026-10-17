"""Durable local workflow with bounded subprocesses and Chinese evidence."""

import fnmatch
import importlib.metadata
import importlib.util
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mdio import (
    WorkflowError,
    atomic_write,
    digest,
    fields,
    lock,
    read_table,
    table,
    update_fields,
    write_fields,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
MANIFEST = ROOT / "references/算子回归清单.md"
MANIFEST_HEADERS = ("验证层", "分类", "核心模式", "测试入口")
PLAN_HEADERS = ("测试节点", "后端", "分类")
RESULT_HEADERS = ("测试节点", "后端", "分类", "结果", "原因摘要", "证据目录")
RULES = {
    "M01": ("CANN函数路由", "融合数学"),
    "M02": ("masked-load默认值", "逐元素索引,归约循环"),
    "M03": ("clamp分解", "逐元素索引,融合数学"),
    "M04": ("FP64精度语义", "逐元素索引,归约循环"),
    "M05": ("autotune-key", "逐元素索引,矩阵布局"),
    "M06": ("候选tile策略", "矩阵布局"),
    "M07": ("SDPA尾块", "融合数学,矩阵布局"),
    "M08": ("coreDim资源", "矩阵布局,归约循环"),
    "M09": ("多轴调优", "矩阵布局"),
    "M10": ("旧AOT包装", "暂缓"),
    "M11": ("独立后端路由", "逐元素索引"),
    "M12": ("宿主缓存异常", "逐元素索引"),
}
FATAL = re.compile(
    r"507014|aicore.*(?:tim(?:e[sd]?\s*out|eout))|device.*(?:lost|unhealthy)|ACL_ERROR_RT_AICORE",
    re.I,
)


@dataclass
class ProcessResult:
    returncode: int | None
    timed_out: bool
    interrupted: bool
    elapsed: float
    pid: int | None = None
    start_token: str = ""


def process_identity(pid):
    try:
        columns = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return "" if columns[0] == "Z" else columns[19]
    except (FileNotFoundError, ProcessLookupError):
        return ""


def bounded_process(command, cwd, env, log, timeout, pass_fds=()):
    if timeout <= 0:
        raise WorkflowError("执行时限必须大于零")
    started = time.monotonic()
    timed_out = interrupted = False
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        start_token = process_identity(process.pid)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if timed_out or interrupted:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                # Kill the original group even if the leader has exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # A process in uninterruptible kernel sleep may outlive SIGKILL.
                    pass
    return ProcessResult(
        process.returncode,
        timed_out,
        interrupted,
        time.monotonic() - started,
        process.pid,
        start_token,
    )


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20
    )
    if result.returncode:
        raise WorkflowError(f"Git 检查失败：{repo}：{result.stderr.strip()}")
    return result.stdout.strip()


def tree_digest(target, section=None):
    target = Path(target)
    directories = (
        [target / "src", target / "tests"] if section is None else [target / section]
    )
    files = [p for directory in directories for p in directory.rglob("*.py")]
    files += [
        p
        for p in (
            target / "pyproject.toml",
            target / "pytest.ini",
            target / "conftest.py",
        )
        if p.exists()
    ]
    return digest(
        "\n".join(
            f"{p.relative_to(target)}:{digest(p.read_bytes())}" for p in sorted(files)
        )
    )


def environment():
    packages = {}
    for package in (
        "torch",
        "torch-npu",
        "triton",
        "triton-ascend",
        "pytest",
        "sympy",
        "numpy",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "未安装"
    identity = {"解释器": sys.executable, "Python版本": sys.version, **packages}
    identity["工作流代码摘要"] = digest(
        "\n".join(
            f"{p.name}:{digest(p.read_bytes())}" for p in sorted(SCRIPTS.glob("*.py"))
        )
    )
    for key in (
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_LAUNCH_BLOCKING",
        "TRITON_CACHE_DIR",
        "NINETOOTHED_BACKEND",
        "NINETOOTHED_PLATFORM",
    ):
        identity[key] = os.environ.get(key, "未设置")
    # Record the imported namespace's sources without initializing a driver.
    spec = importlib.util.find_spec("triton")
    if spec and spec.origin:
        package = Path(spec.origin).parent
        for relative in (
            "__init__.py",
            "backends/ascend/compiler.py",
            "backends/ascend/driver.py",
        ):
            path = package / relative
            identity[f"Triton文件:{relative}"] = (
                digest(path.read_bytes()) if path.exists() else "不存在"
            )
    candidates = [
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path(
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
        ),
    ]
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        candidates.extend(Path(home) / name for name in ("version.cfg", "version.info"))
    for path in candidates:
        if path.is_file():
            identity[f"CANN版本文件:{path}"] = digest(path.read_bytes())
    return identity


def env_digest():
    return digest(repr(sorted(environment().items())))


def config(run):
    result = fields(Path(run) / "配置.md")
    for key in ("目标仓库", "源仓库", "规范目录", "清单路径"):
        if not Path(result[key]).exists():
            raise WorkflowError(f"缺少依赖：{key}：{result[key]}")
    return result


def validate_selector(target, node):
    filename, separator, function = node.partition("::")
    path = (Path(target) / filename).resolve()
    base = (Path(target) / "tests").resolve()
    if (
        not separator
        or not path.is_relative_to(base)
        or path.suffix != ".py"
        or not path.is_file()
    ):
        raise WorkflowError(f"测试入口不在允许的 tests/ 算子范围：{node}")
    if (
        not function.startswith("test_")
        or "::" in function
        or any(c in filename for c in "*?[")
    ):
        raise WorkflowError(f"测试入口必须精确到算子函数：{node}")
    return path


def manifest(config_values):
    rows = read_table(config_values["清单路径"], MANIFEST_HEADERS)
    for row in rows:
        validate_selector(config_values["目标仓库"], row["测试入口"])
        if row["验证层"] not in ("源码", "设备"):
            raise WorkflowError("清单中的验证层非法")
    return rows


def child_env(target, **extra):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("NT_PASS_"):
            env.pop(key)
    env.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
            "PYTHONPATH": os.pathsep.join(
                (str(SCRIPTS), str(Path(target) / "src"), str(target))
            ),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update({key: str(value) for key, value in extra.items()})
    return env


def pytest_command(*selectors):
    return [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-o",
        "junit_family=xunit1",
        "-p",
        "pytest_bridge",
        *selectors,
    ]


def initialize(output, source, target, policy):
    output, source, target, policy = map(
        lambda p: Path(p).resolve(), (output, source, target, policy)
    )
    if output.exists():
        raise WorkflowError(f"批次目录已经存在，不能覆盖：{output}")
    for path in (
        source / ".git",
        target / "tests",
        target / "src",
        policy / "compiler-pass-pipeline/SKILL.md",
    ):
        if not path.exists():
            raise WorkflowError(f"缺少依赖：{path}")
    source_commit = git(source, "rev-parse", "HEAD")
    target_commit = git(target, "rev-parse", "HEAD")
    output.mkdir(parents=True)
    write_fields(
        output / "配置.md",
        "迁移批次配置",
        {
            "源仓库": source,
            "目标仓库": target,
            "规范目录": policy,
            "清单路径": MANIFEST,
            "源提交": source_commit,
            "目标初始提交": target_commit,
            "源初始摘要": tree_digest(source),
            "规范摘要": policy_digest(policy),
        },
    )
    snapshot(output, "初始化，保留既有工作树")
    for rule, (name, group) in RULES.items():
        write_fields(
            output / "规则记录" / f"{rule}.md",
            f"{rule}：{name}",
            {
                "规则编号": rule,
                "名称": name,
                "状态": "暂缓" if group == "暂缓" else "待处理",
                "相关分类": group,
                "结论": "尚未处理",
                "回归比较": "无",
            },
        )
    summarize(output)
    return output


def policy_digest(policy):
    files = sorted(Path(policy).rglob("*.md"))
    return digest(
        "\n".join(f"{p.relative_to(policy)}:{digest(p.read_bytes())}" for p in files)
    )


def snapshot(run, reason):
    cfg = config(run)
    values = {
        "说明": reason,
        "目标提交": git(cfg["目标仓库"], "rev-parse", "HEAD"),
        "目标源码摘要": tree_digest(cfg["目标仓库"]),
        "测试摘要": tree_digest(cfg["目标仓库"], "tests"),
        "环境摘要": env_digest(),
        **environment(),
    }
    write_fields(
        Path(run) / "环境与源码快照.md",
        "环境与源码快照",
        values,
        "## 目标工作树既有状态\n\n```text\n"
        + git(cfg["目标仓库"], "status", "--short")
        + "\n```\n",
    )
    atomic_write(
        Path(run) / "补丁/初始工作树.diff",
        git(cfg["目标仓库"], "diff", "--binary") + "\n",
    )


def collect(run, layer, group, backend, match=None):
    run = Path(run)
    cfg = config(run)
    rows = [r for r in manifest(cfg) if r["验证层"] == layer]
    if not rows:
        raise WorkflowError("清单没有此验证层")
    selectors = sorted({r["测试入口"] for r in rows})
    name = f"计划-{layer}-{group}-{backend}" + (
        f"-{digest(match)[:8]}" if match else ""
    )
    evidence = run / "原始证据" / name
    evidence.mkdir(parents=True, exist_ok=True)
    collection = evidence / "收集.md"
    collection.unlink(missing_ok=True)
    env = child_env(cfg["目标仓库"], NT_PASS_COLLECTION=collection)
    command = pytest_command("--collect-only", "-q", *selectors)
    result = bounded_process(command, cfg["目标仓库"], env, evidence / "收集.log", 90)
    if (
        result.returncode != 0
        or result.timed_out
        or result.interrupted
        or not collection.exists()
    ):
        raise WorkflowError(f"测试收集失败，查看 {evidence / '收集.log'}")
    collected = read_table(collection, PLAN_HEADERS)
    selected = []
    for item in collected:
        validate_selector(cfg["目标仓库"], item["测试节点"])
        if item["后端"] not in ("triton", "triton-ascend") or item["分类"] == "未知":
            raise WorkflowError(f"无法明确归因的算子节点：{item}")
        if backend != "两者" and item["后端"] != backend:
            continue
        matching_rows = [
            r
            for r in rows
            if item["测试节点"].split("[")[0] == r["测试入口"]
            and item["分类"] == r["分类"]
        ]
        if not matching_rows:
            raise WorkflowError(f"算子不在分类清单中：{item['测试节点']}")
        if group == "核心" and not any(
            fnmatch.fnmatchcase(item["测试节点"], r["核心模式"]) for r in matching_rows
        ):
            continue
        if group not in ("核心", "完整") and item["分类"] != group:
            continue
        if match and not fnmatch.fnmatchcase(item["测试节点"], match):
            continue
        selected.append(item)
    if not selected:
        raise WorkflowError("实际选择了零个算子，不能标记成功")
    if len({x["测试节点"] for x in selected}) != len(selected):
        raise WorkflowError("清单包含重复节点")
    path = run / f"{name}.md"
    write_fields(
        path,
        "算子执行计划",
        {
            "验证层": layer,
            "回归分组": group,
            "请求后端": backend,
            "附加筛选": match or "无",
            "测试摘要": tree_digest(cfg["目标仓库"], "tests"),
            "清单摘要": digest(Path(cfg["清单路径"]).read_bytes()),
            "计划节点数": len(selected),
        },
        "## 实际测试节点\n\n"
        + table(PLAN_HEADERS, ([x[k] for k in PLAN_HEADERS] for x in selected)),
    )
    return path


def device_paths(device):
    if not re.fullmatch(r"[0-9]+", str(device)):
        raise WorkflowError("物理设备编号必须是非负整数")
    root = Path(
        os.environ.get(
            "NT_PASS_LOCK_ROOT", Path.home() / ".cache/ninetoothed-pass-workflow"
        )
    )
    return root / f"npu-{device}.lock", root / f"npu-{device}-设备状态.md"


def quarantine(device, reason, process=None):
    _, path = device_paths(device)
    values = {
        "状态": "隔离",
        "原因": reason,
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if path.exists():
        previous = fields(path)
        for key in ("残留子进程PID", "残留进程启动标识"):
            if key in previous:
                values[key] = previous[key]
    if process is not None and process.returncode is None:
        values.update(
            {"残留子进程PID": process.pid, "残留进程启动标识": process.start_token}
        )
    write_fields(
        path,
        "设备熔断状态",
        values,
    )


def device_environment(target, device, **kwargs):
    return child_env(target, ASCEND_RT_VISIBLE_DEVICES=str(device), **kwargs)


def health_probe(run, device, clear=False, pass_fds=()):
    cfg = config(run)
    _, marker = device_paths(device)
    if marker.exists():
        prior = fields(marker)
        pid = prior.get("残留子进程PID")
        if (
            pid
            and process_identity(pid)
            and process_identity(pid) == prior.get("残留进程启动标识")
        ):
            raise WorkflowError(f"仍有本工作流残留进程 {pid}，拒绝提交新的设备工作")
    if marker.exists() and fields(marker)["状态"] in {"隔离", "执行中"} and not clear:
        raise WorkflowError(f"设备已隔离：{marker}；完成外部恢复后使用 recover-device")
    evidence = Path(run) / "原始证据" / f"设备检查-{time.time_ns()}"
    evidence.mkdir(parents=True)
    command = [
        sys.executable,
        str(SCRIPTS / "device_probe.py"),
        str(evidence / "健康.md"),
    ]
    result = bounded_process(
        command,
        cfg["目标仓库"],
        device_environment(cfg["目标仓库"], device),
        evidence / "健康.log",
        40,
        pass_fds=pass_fds,
    )
    if (
        result.returncode
        or result.timed_out
        or result.interrupted
        or not (evidence / "健康.md").exists()
    ):
        text = (evidence / "健康.log").read_text(errors="replace")
        if (
            result.timed_out
            or result.interrupted
            or result.returncode is None
            or result.returncode < 0
            or FATAL.search(text)
        ):
            quarantine(
                device, f"健康检查出现设备致命错误或结果未知：{evidence}", result
            )
        raise WorkflowError(f"NPU 健康检查未通过：{evidence}")
    write_fields(
        marker, "设备状态", {"状态": "健康", "证据": evidence, "物理设备": device}
    )
    return evidence


def check_inputs(cfg):
    if tree_digest(cfg["源仓库"]) != cfg["源初始摘要"]:
        raise WorkflowError("旧仓库发生变化，请建立新批次，避免迁移依据失配")
    if policy_digest(cfg["规范目录"]) != cfg["规范摘要"]:
        raise WorkflowError("规范依赖发生变化，请建立新批次")
    if git(cfg["目标仓库"], "rev-parse", "HEAD") != cfg["目标初始提交"]:
        raise WorkflowError("目标 HEAD 已变化，请建立新批次；旧证据仍保留")


def normalize_failure(reason):
    return re.sub(r"0x[0-9a-fA-F]+", "<地址>", reason)


def assess_case(result, observation, log, layer):
    text = Path(log).read_text(errors="replace")
    fatal = layer == "设备" and (
        result.timed_out
        or result.interrupted
        or result.returncode is None
        or result.returncode < 0
        or bool(FATAL.search(text))
    )
    if result.interrupted:
        return "中断", "执行被中断，结果需要核对", fatal
    if result.timed_out:
        return "超时", "达到单例执行时限", fatal
    if result.returncode < 0:
        return "失败", f"进程收到信号 {-result.returncode}", fatal
    if not Path(observation).exists():
        return "失败", "pytest 未写出可验证的观测记录", fatal
    observed = fields(observation)
    status = observed["测试结论"]
    reason = normalize_failure(observed["失败原因"])
    if status == "通过" and result.returncode != 0:
        return "失败", "测试调用通过但进程退出异常", fatal
    if fatal:
        return "失败", reason or "检测到设备致命错误", True
    if status not in ("通过", "失败", "跳过", "未执行"):
        raise WorkflowError("观测结果包含未知状态")
    return status, reason, False


def execute(run, plan, phase, device="0", timeout=120, budget=1800):
    run, plan = Path(run).resolve(), Path(plan).resolve()
    if not plan.is_relative_to(run) or timeout <= 0 or budget <= 0:
        raise WorkflowError("计划必须在本批次内，时限/预算必须为正数")
    cfg = config(run)
    check_inputs(cfg)
    for path in (run / "执行记录").glob("执行-*/记录.md"):
        if fields(path)["状态"] == "执行中":
            raise WorkflowError("发现遗留执行中状态，请先 recover；不能自动重放 kernel")
    info = fields(plan)
    if tree_digest(cfg["目标仓库"], "tests") != info["测试摘要"]:
        raise WorkflowError("测试或配置已变化，请重新 collect，旧计划不可复用")
    if digest(Path(cfg["清单路径"]).read_bytes()) != info["清单摘要"]:
        raise WorkflowError("算子清单已变化，请重新 collect")
    rows = read_table(plan, PLAN_HEADERS)
    if len(rows) != int(info["计划节点数"]):
        raise WorkflowError("计划节点数不一致")
    allowed = {x["测试入口"] for x in manifest(cfg) if x["验证层"] == info["验证层"]}
    for row in rows:
        validate_selector(cfg["目标仓库"], row["测试节点"])
        if row["测试节点"].split("[")[0] not in allowed:
            raise WorkflowError("计划包含清单外入口")
    layer = info["验证层"]
    if layer == "设备" and any(r["后端"] != "triton-ascend" for r in rows):
        raise WorkflowError("设备主队列只接受 triton-ascend；旧 Triton 保留源码对照")
    active_rule = "仅回归"
    for path in (run / "规则记录").glob("M*.md"):
        rule = fields(path)
        if rule["状态"] == "处理中":
            active_rule = rule["规则编号"]
            if phase == "修改后":
                versions = rule.get("已尝试源码", "").splitlines()
                current = tree_digest(cfg["目标仓库"])
                if current not in versions:
                    if len(versions) >= int(rule.get("修正上限", "3")):
                        raise WorkflowError(
                            "本规则已达到三次不同源码的验证上限，请暂缓并记录根因"
                        )
                    rule["已尝试源码"] = "\n".join([*versions, current])
                    update_fields(path, f"规则 {active_rule}", rule)
    existing = list((run / "执行记录").glob("执行-*/记录.md"))
    serial = max((int(p.parent.name.split("-")[1]) for p in existing), default=0) + 1
    directory = run / "执行记录" / f"执行-{serial:04d}"
    directory.mkdir(parents=True)
    atomic_write(directory / "计划副本.md", plan.read_text())
    write_fields(directory / "环境.md", "本次执行环境", environment())
    state = {
        "执行编号": directory.name,
        "状态": "执行中",
        "阶段": phase,
        "规则编号": active_rule,
        "验证层": layer,
        "回归分组": info["回归分组"],
        "附加筛选": info["附加筛选"],
        "源码摘要": tree_digest(cfg["目标仓库"]),
        "测试摘要": info["测试摘要"],
        "环境摘要": env_digest(),
        "清单摘要": info["清单摘要"],
        "计划摘要": digest(plan.read_bytes()),
        "物理设备": device if layer == "设备" else "无",
        "单例时限秒": timeout,
        "批次预算秒": budget,
        "设备芯片": "未检查",
    }
    outcomes = []

    def persist():
        state["实际执行数"] = sum(r["结果"] != "阻塞" for r in outcomes)
        state["计划节点数"] = len(rows)
        state["通过数"] = sum(r["结果"] == "通过" for r in outcomes)
        state["失败数"] = sum(r["结果"] in ("失败", "超时", "中断") for r in outcomes)
        state["跳过数"] = sum(r["结果"] == "跳过" for r in outcomes)
        state["阻塞数"] = sum(r["结果"] == "阻塞" for r in outcomes)
        state["尚未记录结果数"] = len(rows) - len(outcomes)
        state["门禁"] = (
            "通过"
            if len(outcomes) == len(rows) and all(r["结果"] == "通过" for r in outcomes)
            else "未全部通过"
        )
        write_fields(
            directory / "记录.md",
            "算子执行记录",
            state,
            "## 逐例结果\n\n"
            + table(RESULT_HEADERS, ([r[k] for k in RESULT_HEADERS] for r in outcomes)),
        )

    persist()
    started = time.monotonic()
    stopped = None
    device_acquired = False
    case_started = False
    try:
        with (
            lock(device_paths(device)[0])
            if layer == "设备"
            else lock(directory / "源码执行.lock")
        ) as execution_lock:
            if layer == "设备":
                device_acquired = True
                probe = health_probe(run, device, pass_fds=(execution_lock.fileno(),))
                state["设备芯片"] = fields(probe / "健康.md")["芯片"]
                state["设备检查证据"] = str(probe)
            for index, row in enumerate(rows, 1):
                if stopped or time.monotonic() - started >= budget:
                    stopped = stopped or "达到本次执行预算"
                    outcomes.append(
                        {**row, "结果": "阻塞", "原因摘要": stopped, "证据目录": "无"}
                    )
                    persist()
                    continue
                case = directory / f"案例-{index:04d}"
                case.mkdir()
                env = child_env(
                    cfg["目标仓库"],
                    NT_PASS_EVIDENCE=case,
                    NT_PASS_TARGET=cfg["目标仓库"],
                    NT_PASS_BACKEND=row["后端"],
                    NT_PASS_LAYER=layer,
                    NT_PASS_NODE=row["测试节点"],
                )
                if layer == "设备":
                    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
                command = pytest_command(
                    "-q",
                    "--tb=short",
                    "--junitxml",
                    str(case / "pytest.xml"),
                    row["测试节点"],
                )
                write_fields(
                    case / "命令.md",
                    "案例执行上下文",
                    {
                        "测试节点": row["测试节点"],
                        "实际命令": shlex.join(command),
                        "工作目录": cfg["目标仓库"],
                        "请求后端": row["后端"],
                        "验证层": layer,
                        "物理设备": state["物理设备"],
                        "输入语义": "复用 tests/ 的固定输入与原有断言，不修改精度阈值",
                    },
                )
                print(
                    f"[{directory.name} {index}/{len(rows)}] {row['测试节点']}",
                    flush=True,
                )
                case_started = True
                if layer == "设备":
                    write_fields(
                        device_paths(device)[1],
                        "设备执行检查点",
                        {
                            "状态": "执行中",
                            "物理设备": device,
                            "批次": run,
                            "证据": case,
                            "主控PID": os.getpid(),
                        },
                    )
                result = bounded_process(
                    command,
                    cfg["目标仓库"],
                    env,
                    case / "pytest.log",
                    min(timeout, max(0.01, budget - (time.monotonic() - started))),
                    pass_fds=(execution_lock.fileno(),) if layer == "设备" else (),
                )
                outcome, reason, fatal = assess_case(
                    result, case / "观测.md", case / "pytest.log", layer
                )
                write_fields(
                    case / "进程.md",
                    "宿主进程结果",
                    {
                        "退出码": result.returncode,
                        "耗时秒": f"{result.elapsed:.3f}",
                        "是否超时": result.timed_out,
                        "是否中断": result.interrupted,
                    },
                )
                outcomes.append(
                    {**row, "结果": outcome, "原因摘要": reason, "证据目录": str(case)}
                )
                if fatal:
                    stopped = f"设备故障或执行结果未知，证据：{case}"
                    quarantine(device, stopped, result)
                elif layer == "设备":
                    write_fields(
                        device_paths(device)[1],
                        "设备执行已返回",
                        {
                            "状态": "健康",
                            "物理设备": device,
                            "最近执行证据": case,
                        },
                    )
                if result.interrupted:
                    stopped = "宿主执行被中断"
                persist()
            if (
                tree_digest(cfg["目标仓库"]) != state["源码摘要"]
                or env_digest() != state["环境摘要"]
            ):
                state["状态"] = "执行期间输入变化"
            else:
                state["状态"] = "阻塞" if stopped else "已结束"
    except BaseException as exc:
        state["状态"] = "中断待核对"
        state["中断原因"] = str(exc)
        for row in rows[len(outcomes) :]:
            outcomes.append(
                {**row, "结果": "阻塞", "原因摘要": str(exc), "证据目录": "无"}
            )
        if layer == "设备" and device_acquired and case_started:
            quarantine(device, f"执行异常，需核对：{directory}")
        persist()
        raise
    finally:
        state["总耗时秒"] = f"{time.monotonic() - started:.3f}"
        persist()
        summarize(run)
    return directory


def compare_runs(run, before, after):
    run = Path(run).resolve()
    before, after = (Path(p).resolve() for p in (before, after))
    for directory in (before, after):
        if not directory.is_relative_to(run / "执行记录"):
            raise WorkflowError("比较对象必须来自本批次的执行记录")
    baseline, candidate = fields(before / "记录.md"), fields(after / "记录.md")
    if baseline["阶段"] != "基线" or candidate["阶段"] != "修改后":
        raise WorkflowError("比较需要一个基线执行和一个修改后执行")
    for key in ("测试摘要", "环境摘要", "清单摘要", "计划摘要", "物理设备", "设备芯片"):
        if baseline[key] != candidate[key]:
            raise WorkflowError(f"比较条件变化：{key}；重新建立可比基线")
    if baseline["状态"] != "已结束" or candidate["状态"] != "已结束":
        raise WorkflowError("执行未完整结束，不能通过回归门禁")
    old, new = (read_table(p / "记录.md", RESULT_HEADERS) for p in (before, after))
    old = {(r["测试节点"], r["后端"]): r for r in old}
    new = {(r["测试节点"], r["后端"]): r for r in new}
    if old.keys() != new.keys():
        raise WorkflowError("前后测试节点不一致，不能用总数替代逐例比较")
    regressions = fixed = previous_failures = 0
    compared = []
    for key, row in new.items():
        prior = old[key]
        same_failure = (
            prior["结果"] == row["结果"] == "失败"
            and prior["原因摘要"] == row["原因摘要"]
        )
        if row["结果"] == prior["结果"] == "通过":
            verdict = "保持通过"
        elif row["结果"] == "通过" and prior["结果"] == "失败":
            verdict, fixed = "本轮修复", fixed + 1
        elif same_failure:
            verdict, previous_failures = "原有失败", previous_failures + 1
        else:
            verdict, regressions = "新增回归或验证缺失", regressions + 1
        compared.append((key[0], key[1], prior["结果"], row["结果"], verdict))
    path = Path(run) / "回归记录" / f"{before.name}-对比-{after.name}.md"
    write_fields(
        path,
        "修改前后回归比较",
        {
            "基线目录": before,
            "修改后目录": after,
            "门禁": "通过" if regressions == 0 else "失败",
            "新增回归数": regressions,
            "本轮修复数": fixed,
            "原有失败数": previous_failures,
            "验证层": candidate["验证层"],
            "回归分组": candidate["回归分组"],
            "附加筛选": candidate["附加筛选"],
            "修改后源码摘要": candidate["源码摘要"],
        },
        "## 逐例比较\n\n"
        + table(("测试节点", "后端", "修改前", "修改后", "中文结论"), compared)
        + "\n门禁通过仅代表无新增回归，不等于原有失败已经修复。\n",
    )
    summarize(run)
    return path


def begin_rule(run, rule):
    cfg = config(run)
    check_inputs(cfg)
    for path in (Path(run) / "规则记录").glob("M*.md"):
        current = fields(path)
        if current["状态"] == "处理中":
            raise WorkflowError(f"请先完成或暂缓当前规则：{current['规则编号']}")
    path = Path(run) / "规则记录" / f"{rule}.md"
    state = fields(path)
    if state["状态"] not in ("待处理", "暂缓"):
        raise WorkflowError("规则已经处理，不能覆盖原有结论；重新验证请建立新批次")
    state.update({"状态": "处理中", "开始源码摘要": tree_digest(cfg["目标仓库"])})
    update_fields(path, f"规则 {rule}", state)
    summarize(run)


FOCUS = {
    "M01": ("libdevice-fmod",),
    "M02": ("masked-copy", "row-reduction"),
    "M03": ("clamp",),
    "M04": ("float64",),
    "M05": ("matmul",),
    "M06": ("matmul",),
    "M07": ("attention",),
    "M08": ("matmul",),
    "M09": ("matmul",),
    "M11": ("add",),
    "M12": ("add",),
}


def reject_hidden_retry_failures(run, rule, comparisons, current_digest):
    baselines = {}
    for path in comparisons:
        info = fields(Path(fields(path)["基线目录"]) / "记录.md")
        key = tuple(info[k] for k in ("计划摘要", "环境摘要", "物理设备"))
        baselines[key] = {
            r["测试节点"]: r
            for r in read_table(
                Path(fields(path)["基线目录"]) / "记录.md", RESULT_HEADERS
            )
        }
    for path in (Path(run) / "执行记录").glob("执行-*/记录.md"):
        info = fields(path)
        if (
            info.get("规则编号") != rule
            or info["阶段"] != "修改后"
            or info["源码摘要"] != current_digest
        ):
            continue
        key = tuple(info[k] for k in ("计划摘要", "环境摘要", "物理设备"))
        if key not in baselines:
            continue
        for row in read_table(path, RESULT_HEADERS):
            before = baselines[key].get(row["测试节点"])
            known = (
                before
                and before["结果"] == row["结果"] == "失败"
                and before["原因摘要"] == row["原因摘要"]
            )
            if before and row["结果"] != "通过" and not known:
                raise WorkflowError(
                    f"当前源码曾有新增失败或验证缺失，不能用重跑通过掩盖：{path}"
                )


def finish_rule(run, rule, decision, reason, comparisons):
    cfg = config(run)
    check_inputs(cfg)
    path = Path(run) / "规则记录" / f"{rule}.md"
    state = fields(path)
    if state["状态"] != "处理中":
        raise WorkflowError("只能完成正在处理的规则")
    evidence = []
    if decision != "暂缓":
        layers, focused = set(), {"源码": set(), "设备": set()}
        current_digest = tree_digest(cfg["目标仓库"])
        reject_hidden_retry_failures(run, rule, comparisons, current_digest)
        for comparison in comparisons:
            comparison = Path(comparison).resolve()
            if not comparison.is_relative_to(Path(run).resolve() / "回归记录"):
                raise WorkflowError("回归记录不属于本批次")
            record = fields(comparison)
            if record["门禁"] != "通过" or record["修改后源码摘要"] != current_digest:
                raise WorkflowError("回归失败或源码已变化，不能完成规则")
            candidate = Path(record["修改后目录"])
            candidate_state = fields(candidate / "记录.md")
            if candidate_state["环境摘要"] != env_digest():
                raise WorkflowError("环境在回归后变化，请重新验证")
            rows = read_table(candidate / "记录.md", RESULT_HEADERS)
            if record["回归分组"] == "核心" and record["附加筛选"] == "无":
                target_set = {r["后端"] for r in rows}
                if record["验证层"] == "源码" and target_set == {
                    "triton",
                    "triton-ascend",
                }:
                    layers.add("源码核心")
                if record["验证层"] == "设备" and target_set == {"triton-ascend"}:
                    layers.add("设备核心")
            for token in FOCUS.get(rule, ()):
                relevant = [
                    r
                    for r in rows
                    if token in r["测试节点"] and r["后端"] == "triton-ascend"
                ]
                if relevant and all(r["结果"] == "通过" for r in relevant):
                    focused[record["验证层"]].add(token)
            evidence.append(str(comparison))
        if layers != {"源码核心", "设备核心"}:
            raise WorkflowError(
                "缺少当前源码的双 target 源码核心回归或 Ascend 设备核心回归"
            )
        needed = set(FOCUS.get(rule, ()))
        if not needed or any(focused[layer] != needed for layer in focused):
            raise WorkflowError(
                "相关规则的目标算子尚未在源码与设备层全部通过；保留处理中或暂缓"
            )
    state.update(
        {
            "状态": decision,
            "结论": reason,
            "回归比较": "\n".join(evidence) or "暂缓，未宣称完成",
        }
    )
    update_fields(path, f"规则 {rule}", state)
    summarize(run)


def recover(run):
    cfg = config(run)
    check_inputs(cfg)
    messages = []
    for path in sorted((Path(run) / "执行记录").glob("执行-*/记录.md")):
        record = fields(path)
        if record["状态"] == "执行中":
            if record["验证层"] == "设备":
                quarantine(record["物理设备"], f"上次执行中断，设备结果未知：{path}")
            record["状态"] = "中断待核对"
            body = path.read_text().split("## 逐例结果", 1)
            write_fields(
                path,
                "中断执行记录",
                record,
                "## 逐例结果" + body[1] if len(body) == 2 else "",
            )
            messages.append(str(path))
    current = tree_digest(cfg["目标仓库"])
    for path in (Path(run) / "执行记录").glob("执行-*/记录.md"):
        if fields(path)["源码摘要"] != current:
            messages.append(f"历史证据对应其他源码，不复用：{path.parent.name}")
    summarize(run)
    atomic_write(
        Path(run) / "恢复检查点.md",
        "# 恢复检查结果\n\n"
        + "\n".join(
            f"- {x}"
            for x in messages
            or ["未发现遗留执行中状态；执行前仍校验计划、环境与设备。"]
        )
        + "\n\n恢复只核对状态，不自动重放 kernel。\n",
    )
    return messages


def summary_text(run):
    run = Path(run)
    rules = [fields(p) for p in sorted((run / "规则记录").glob("M*.md"))]
    executions = []
    for path in sorted((run / "执行记录").glob("执行-*/记录.md")):
        record = fields(path)
        if "失败数" not in record:
            try:
                rows = read_table(path, RESULT_HEADERS)
            except WorkflowError:
                if int(record["实际执行数"]) != 0:
                    raise
                rows = []
            record["失败数"] = sum(r["结果"] in ("失败", "超时", "中断") for r in rows)
            record["跳过数"] = sum(r["结果"] == "跳过" for r in rows)
            record["阻塞数"] = sum(r["结果"] == "阻塞" for r in rows)
        executions.append(record)
    text = "# 昇腾迁移执行总览\n\n## 规则进度\n\n"
    text += table(
        ("规则", "名称", "状态", "中文结论"),
        ((r["规则编号"], r["名称"], r["状态"], r["结论"]) for r in rules),
    )
    text += "\n## 分层执行记录\n\n"
    text += table(
        (
            "执行编号",
            "阶段",
            "验证层",
            "分组",
            "状态",
            "通过/计划",
            "失败",
            "跳过",
            "阻塞",
            "完整记录",
        ),
        (
            (
                r["执行编号"],
                r["阶段"],
                r["验证层"],
                r["回归分组"],
                r["状态"],
                f"{r['通过数']}/{r['计划节点数']}",
                r["失败数"],
                r["跳过数"],
                r["阻塞数"],
                f"[查看记录](执行记录/{r['执行编号']}/记录.md)",
            )
            for r in executions
        ),
    )
    text += "\n源码、设备、构建和性能分别统计；源码通过不等于设备支持。执行器不修改编译器、不放宽断言、不自动迁移语义。\n"
    return text


def summarize(run):
    atomic_write(Path(run) / "执行总览.md", summary_text(run))

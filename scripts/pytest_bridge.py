"""Out-of-tree pytest instrumentation; never modify the compiler sources."""

import hashlib
import os
from pathlib import Path

import pytest
from mdio import atomic_write, table, write_fields

COLLECTION_HEADERS = ("测试节点", "后端", "分类")
SOURCE_GROUPS = {
    "elementwise-indexing": "逐元素索引",
    "reduction-control-flow": "归约循环",
    "dot-linalg": "矩阵布局",
    "composite": "融合数学",
}
NAMES = {
    "逐元素索引": "add broadcast-add expand-copy pad-view elementwise-cast masked-copy raw-load source-index source-store",
    "归约循环": "atomic-sum row-reduction reduction-loop scan conditional-loop",
    "矩阵布局": "matmul matmul-transpose addmm conv2d layout-transfer",
    "融合数学": "softmax attention libdevice-fmod dropout",
}


def classification(item):
    params = getattr(getattr(item, "callspec", None), "params", {})
    backend = params.get("runtime_contract", params.get("target_contract", "未知"))
    group = SOURCE_GROUPS.get(params.get("category"))
    if group is None:
        group = next(
            (k for k, names in NAMES.items() if params.get("name") in names.split()),
            "未知",
        )
    return str(backend), group


def pytest_configure(config):
    config._workflow_reports = []
    config._workflow_artifacts = []
    config._workflow_calls = 0
    config._workflow_restore = []
    evidence = os.environ.get("NT_PASS_EVIDENCE")
    if evidence is None:
        return
    import ninetoothed
    import ninetoothed.compiler.runtime as runtime_module
    from ninetoothed.compiler.driver import Compiler
    from ninetoothed.compiler.runtime import Handle

    root = Path(os.environ["NT_PASS_TARGET"]).resolve()
    if Path(ninetoothed.__file__).resolve() != root / "src/ninetoothed/__init__.py":
        raise pytest.UsageError("导入的 ninetoothed 不是所选目标工作树")
    expected = os.environ["NT_PASS_BACKEND"]
    source_only = os.environ["NT_PASS_LAYER"] == "源码"
    original_compile, original_call = Compiler.compile, Handle.__call__

    def record(compilation):
        artifact = compilation.artifact
        actual = artifact.backend.value
        if actual != expected:
            raise AssertionError(f"实际产物后端错误：请求 {expected}，得到 {actual}")
        trace = tuple(compilation.pass_trace)
        if f"ssa.{expected}.optimize_schedule" not in trace:
            raise AssertionError("实际 pass trace 缺少目标 schedule pass")
        fingerprint = hashlib.sha256(artifact.primary_source.encode()).hexdigest()
        row = (actual, compilation.target.platform.name, fingerprint, ", ".join(trace))
        if row not in config._workflow_artifacts:
            config._workflow_artifacts.append(row)
            number = len(config._workflow_artifacts)
            atomic_write(
                Path(evidence) / f"产物-{number:03d}.py", artifact.primary_source
            )
            atomic_write(
                Path(evidence) / f"最终SSA-{number:03d}.txt",
                str(compilation.kernel.ssa),
            )

    def checked_compile(self, request):
        result = original_compile(self, request)
        record(result)
        return result

    def checked_call(self, *args, **kwargs):
        if source_only:
            raise AssertionError("源码验证意外执行了 kernel")
        record(self._compilation)
        import torch

        for value in (*args, *kwargs.values()):
            if isinstance(value, torch.Tensor) and value.device.type != "npu":
                raise AssertionError(f"设备验证遇到非 NPU 输入：{value.device}")
        config._workflow_calls += 1
        return original_call(self, *args, **kwargs)

    Compiler.compile, Handle.__call__ = checked_compile, checked_call
    config._workflow_restore = [
        (Compiler, "compile", original_compile),
        (Handle, "__call__", original_call),
    ]
    if source_only:

        def forbid_materialization(*args, **kwargs):
            raise AssertionError("源码验证禁止材料化或初始化设备")

        config._workflow_restore.extend(
            [
                (Compiler, "materialize", Compiler.materialize),
                (runtime_module, "materialize", runtime_module.materialize),
            ]
        )
        Compiler.materialize = forbid_materialization
        runtime_module.materialize = forbid_materialization


def pytest_collection_modifyitems(config, items):
    expected = os.environ.get("NT_PASS_NODE")
    if expected and (len(items) != 1 or items[0].nodeid != expected):
        raise pytest.UsageError("执行节点与计划不一致，拒绝执行额外案例")
    output = os.environ.get("NT_PASS_COLLECTION")
    if output:
        atomic_write(
            output,
            "# 实际收集的算子节点\n\n"
            + table(
                COLLECTION_HEADERS, ((x.nodeid, *classification(x)) for x in items)
            ),
        )


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    config = item.config
    if not os.environ.get("NT_PASS_EVIDENCE"):
        return
    passed = any(r.when == "call" and r.passed for r in config._workflow_reports)
    if not passed:
        return
    if not config._workflow_artifacts:
        raise AssertionError("测试未观察到任何实际编译产物")
    if os.environ["NT_PASS_LAYER"] == "源码":
        import torch

        if hasattr(torch, "npu") and torch.npu.is_initialized():
            raise AssertionError("纯源码案例初始化了 NPU runtime")
    if os.environ["NT_PASS_LAYER"] == "设备":
        if not config._workflow_calls:
            raise AssertionError("设备测试未观察到实际 handle 调用")
        import torch

        torch.npu.synchronize()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    item.config._workflow_reports.append(outcome.get_result())


def pytest_sessionfinish(session, exitstatus):
    output = os.environ.get("NT_PASS_EVIDENCE")
    if not output:
        return
    config = session.config
    reports = config._workflow_reports
    failed = [r for r in reports if r.failed]
    skipped = [r for r in reports if r.skipped]
    passed = any(r.when == "call" and r.passed for r in reports)
    status = "失败" if failed else "跳过" if skipped else "通过" if passed else "未执行"
    reasons = []
    for report in failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        reasons.append(getattr(crash, "message", str(report.longrepr)))
    properties = [row for report in reports for row in report.user_properties]
    write_fields(
        Path(output) / "观测.md",
        "实际编译与执行观测",
        {
            "测试结论": status,
            "失败原因": "\n".join(dict.fromkeys(reasons)),
            "pytest 退出码": int(exitstatus),
            "实际 handle 调用次数": config._workflow_calls,
        },
        "## 实际产物\n\n"
        + table(
            ("后端", "平台", "源码摘要", "实际 pass 顺序"), config._workflow_artifacts
        )
        + "\n## 测试附属记录\n\n"
        + table(("字段", "原始值"), properties),
    )


def pytest_unconfigure(config):
    for owner, name, value in getattr(config, "_workflow_restore", []):
        setattr(owner, name, value)

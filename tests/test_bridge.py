import sys
from types import ModuleType, SimpleNamespace

import pytest
import pytest_bridge


@pytest.fixture
def configured_bridge(tmp_path, monkeypatch):
    root = tmp_path / "target"
    package = ModuleType("ninetoothed")
    package.__file__ = str(root / "src/ninetoothed/__init__.py")
    compiler_package = ModuleType("ninetoothed.compiler")
    driver = ModuleType("ninetoothed.compiler.driver")
    runtime = ModuleType("ninetoothed.compiler.runtime")

    class Compiler:
        def compile(self, request):
            return request

        def materialize(self, *args):
            return "不应调用"

    class Handle:
        launched = False

        def __call__(self, *args, **kwargs):
            self.launched = True

    driver.Compiler, runtime.Handle = Compiler, Handle
    runtime.materialize = lambda *args: "不应调用"
    package.compiler = compiler_package
    compiler_package.driver, compiler_package.runtime = driver, runtime
    for name, module in {
        "ninetoothed": package,
        "ninetoothed.compiler": compiler_package,
        "ninetoothed.compiler.driver": driver,
        "ninetoothed.compiler.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("NT_PASS_TARGET", str(root))
    monkeypatch.setenv("NT_PASS_BACKEND", "triton-ascend")
    monkeypatch.setenv("NT_PASS_EVIDENCE", str(tmp_path / "evidence"))
    config = SimpleNamespace()
    yield config, Compiler, Handle, runtime
    pytest_bridge.pytest_unconfigure(config)


def test_wrong_actual_backend_cannot_launch(configured_bridge, monkeypatch):
    config, compiler, handle_class, _ = configured_bridge
    monkeypatch.setenv("NT_PASS_LAYER", "设备")
    pytest_bridge.pytest_configure(config)
    wrong = SimpleNamespace(
        artifact=SimpleNamespace(backend=SimpleNamespace(value="triton"))
    )
    handle = handle_class()
    handle._compilation = wrong
    with pytest.raises(AssertionError, match="实际产物后端错误"):
        handle()
    assert not handle.launched
    with pytest.raises(AssertionError, match="实际产物后端错误"):
        compiler().compile(wrong)


def test_source_only_cannot_materialize(configured_bridge, monkeypatch):
    config, compiler, _, runtime = configured_bridge
    monkeypatch.setenv("NT_PASS_LAYER", "源码")
    pytest_bridge.pytest_configure(config)
    with pytest.raises(AssertionError, match="禁止材料化"):
        compiler().materialize(None)
    with pytest.raises(AssertionError, match="禁止材料化"):
        runtime.materialize(None)


def test_plugin_rejects_expanded_node_set(monkeypatch):
    monkeypatch.setenv("NT_PASS_NODE", "tests/test_add.py::test[npu]")
    with pytest.raises(pytest.UsageError, match="额外案例"):
        pytest_bridge.pytest_collection_modifyitems(
            None,
            [
                SimpleNamespace(nodeid="tests/test_add.py::test[npu]"),
                SimpleNamespace(nodeid="tests/test_add.py::test[cuda]"),
            ],
        )

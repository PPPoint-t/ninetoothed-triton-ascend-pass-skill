"""Bounded subprocess entry point for a real NPU health check."""

import importlib.metadata
import sys
from pathlib import Path

from mdio import write_fields


def validate_driver_target(target):
    if target.backend not in {"npu", "ascend"} or not str(
        target.arch
    ).lower().startswith("ascend"):
        raise RuntimeError(f"active driver 不是 Ascend：{target}")


def main():
    import torch
    import torch_npu  # noqa: F401
    import triton
    from triton.runtime import driver

    if not torch.npu.is_available():
        raise RuntimeError("没有可用 NPU")
    torch.npu.set_device(0)
    active = driver.active.get_current_target()
    validate_driver_target(active)
    x = torch.arange(8, dtype=torch.float32, device="npu:0")
    y = x + 1
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), torch.arange(8, dtype=torch.float32) + 1)
    write_fields(
        Path(sys.argv[1]),
        "NPU 健康检查",
        {
            "状态": "通过",
            "芯片": torch.npu.get_device_name(0),
            "逻辑设备": "npu:0",
            "实际驱动": str(active),
            "Triton 导入版本": triton.__version__,
            "Triton 模块路径": triton.__file__,
            "torch-npu 版本": importlib.metadata.version("torch-npu"),
        },
    )


if __name__ == "__main__":
    main()

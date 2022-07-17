# -*- coding: utf-8 -*-

"""Common functions for TVM."""

from typing import Dict, List, Tuple, Union
from functools import lru_cache
import inspect
import platform
from pathlib import Path
import tvm
from tvm import relay, autotvm
import onnx


def check_tvm_device_exists(device_type: str) -> bool:
    """Check whether device type exists."""
    try:
        return tvm.device(device_type).exist
    except ValueError:
        return False


@lru_cache(maxsize=None)
def list_tvm_targets() -> List[str]:
    """Get available TVM targets."""
    return [
        x
        for x in tvm.target.Target.list_kinds()
        if check_tvm_device_exists(x)
    ]


def is_flop_limit_supported() -> bool:
    """Check whether flop limit is supported."""
    sig = inspect.signature(autotvm.LocalRunner.__init__)
    return "max_flop_limit" in sig.parameters


def load_onnx_model(model_path: Path) -> \
        Tuple[tvm.IRModule, Dict[str, tvm.nd.NDArray]]:
    """Load ONNX model."""
    model = onnx.load(str(model_path))
    shape_dict = {
        x.name: [y.dim_value for y in x.type.tensor_type.shape.dim]
        for x in model.graph.input
    }
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    return mod, params


def get_device_name(device_type: str, device_idx: int = 0) -> Union[str, None]:
    """Get device name."""
    try:
        device = tvm.device(device_type, device_idx)
        if not device.exist:
            return None
        if device_type.split()[0] == "llvm":
            return platform.processor()
        return device.device_name or ""
    except ValueError:
        return None

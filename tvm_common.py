# -*- coding: utf-8 -*-

"""Common functions for TVM."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
import tvm
from tvm import relay
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

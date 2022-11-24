#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TVM model compiler."""

from typing import Union
import sys
import argparse
import logging
from pathlib import Path
import tvm
from tvm import autotvm, relay
from tvm_common import load_onnx_model

LOG = logging.getLogger(__file__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TVM tuner"
    )
    parser.add_argument(
        "-t", "--target",
        help="TVM target (default: %(default)s)",
        default="llvm",
        type=str,
    )
    parser.add_argument(
        "-O", "--opt",
        help="Optimization level (default: %(default)d)",
        default=3,
        type=int,
        choices=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--tuner-log",
        help="Tuner log",
        type=Path,
    )
    parser.add_argument(
        "model_path",
        help="Path to model file (ONNX)",
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Output path",
        type=Path,
    )
    return parser.parse_args()


class TunerContext:
    """Context manager for tuner log."""

    def __init__(self, tuner_log: Union[Path, None]) -> None:
        """Create TunerContext."""
        if tuner_log is not None:
            self.ctx = autotvm.apply_history_best(str(tuner_log))
        else:
            self.ctx = None

    def __enter__(self) -> "TunerContext":
        """Enter context."""
        if self.ctx is not None:
            self.ctx.__enter__()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Leave context."""
        if self.ctx is not None:
            self.ctx.__exit__(*args, **kwargs)


def main(
    model_path: Path,
    output_path: Path,
    target: str,
    opt: int,
    tuner_log: Union[Path, None],
) -> int:
    """Run CLI."""
    target = tvm.target.Target(target)
    LOG.info("Compile for target %s", str(target))
    LOG.info("Loading model...")
    mod, params = load_onnx_model(model_path)
    LOG.info("Compiling model...")
    with TunerContext(tuner_log):
        with tvm.transform.PassContext(opt_level=opt):
            lib = relay.build(
                mod,
                target=target,
                params=params,
            )
    lib.export_library(str(output_path))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    LOG.setLevel(logging.INFO)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(1)

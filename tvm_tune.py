#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TVM tuner."""

from typing import Any, Dict, List, Union
import os
import sys
import argparse
import logging
import multiprocessing
from enum import Enum
from pathlib import Path
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm_common import load_onnx_model, is_flop_limit_supported


LOG = logging.getLogger(__file__)


class TunerKind(Enum):
    """Tuner kind enum."""

    XGB = "xgb"
    GA = "ga"
    RANDOM = "random"
    GRID_SEARCH = "grid_search"

    def __str__(self):
        """Convert to string."""
        return self.value

    def __repr__(self):
        """Print representation."""
        return str(self)


def argparse_int_list(val: str) -> List[int]:
    """Parse comma separated ints."""
    try:
        return [int(x) for x in val.split(",")]
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid comma separated list: {val}") from exc


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
        help="Optimization level (default: %(default)s)",
        default=3,
        type=int,
        choices=[0, 1, 2, 3]
    )
    parser.add_argument(
        "-k", "--kind",
        dest="tuner_kind",
        help="Tuner type (default: %(default)s)",
        default=TunerKind.GRID_SEARCH,
        type=TunerKind,
        choices=list(TunerKind),
    )
    parser.add_argument(
        "-i", "--tasks",
        dest="task_idx",
        help="Tasks to run (comma separated indices)",
        default=None,
        type=argparse_int_list,
    )
    parser.add_argument(
        "-n", "--num-iter",
        help="Number of iterations",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-s", "--early-stopping",
        help="Early stopping",
        default=None,
        type=int,
    )
    if is_flop_limit_supported():
        parser.add_argument(
            "--max-tflops",
            help="TFLOPS limit",
            default=None,
            type=int,
        )
    parser.add_argument(
        "--measure-num",
        help="Number of measurements",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--measure-repeats",
        help="Repeats of measurements",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--measure-min-time",
        help="Minimum measurement length in ms",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--flush-cpu",
        help="Flush CPU cache before measurements",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--timeout-builder",
        help="Timeout for builder in seconds",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--timeout-runner",
        help="Timeout for runner in seconds",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--enable-transfer-learning",
        help="Enable transfer learning",
        dest="transfer_learning",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "model_path",
        help="Path to model file (ONNX)",
        type=Path,
    )
    parser.add_argument(
        "tuner_log",
        help="Tune log output file",
        type=Path,
    )
    return parser.parse_args()


def extract_tasks(model_path: Path, target: tvm.target.Target, opt: int,
                  op_list: List[str]) -> List[autotvm.task.Task]:
    """Extract tasks from model."""
    mod, params = load_onnx_model(model_path)
    with tvm.transform.PassContext(opt_level=opt):
        return autotvm.task.extract_from_program(
            mod["main"],
            target=target,
            params=params,
            ops=[
                relay.op.get(op)
                for op in op_list
            ]
        )


def tune_kernels(
    tasks: List[autotvm.task.Task],
    measure_option: Dict[str, Any],
    log_filename: Path,
    tuner: TunerKind = TunerKind.GRID_SEARCH,
    early_stopping: Union[int, None] = None,
    n_trial: Union[int, None] = None,
    transfer_learning: bool = False,
) -> None:
    """Tune TVM kernels."""
    for i, task in enumerate(tasks):
        prefix = f"[Task {i + 1:2d}/{len(tasks):2d}] "
        if tuner == TunerKind.XGB:
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == TunerKind.GA:
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == TunerKind.RANDOM:
            tuner_obj = RandomTuner(task)
        elif tuner == TunerKind.GRID_SEARCH:
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError(f"Invalid tuner: {tuner}")
        size = len(task.config_space)
        if n_trial is not None:
            size = min(size, n_trial)
        if transfer_learning and log_filename.is_file():
            tuner_obj.load_history(
                autotvm.record.load_from_file(str(log_filename))
            )
        tuner_obj.tune(
            n_trial=size,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(size, prefix=prefix),
                autotvm.callback.log_to_file(str(log_filename)),
            ],
        )


def get_task_info(task: autotvm.task.Task) -> str:
    """Get TVM task info."""
    func_name = task.name
    config_space = task.config_space
    return f"{func_name}, config space: {len(config_space):n}"


def main(
    model_path: Path,
    tuner_log: Path,
    target: str,
    opt: int,
    tuner_kind: TunerKind,
    task_idx: Union[List[int], None],
    num_iter: Union[int, None],
    early_stopping: Union[int, None],
    measure_num: int,
    measure_repeats: int,
    measure_min_time: int,
    timeout_builder: int,
    timeout_runner: int,
    flush_cpu: bool,
    transfer_learning: bool,
    max_tflops: Union[float, None] = None,
) -> int:
    """Run CLI."""
    target = tvm.target.Target(target)
    LOG.info("Extracting tasks for target %s...", str(target))
    tasks = extract_tasks(
        model_path=model_path,
        target=target,
        opt=opt,
        op_list=[
            "nn.conv2d_transpose",
            "nn.conv2d",
        ]
    )
    LOG.info("Found %d tasks", len(tasks))
    for i, task in enumerate(tasks):
        print(f"{i}: {get_task_info(task)}")
    if task_idx is not None:
        tasks = [
            x
            for i, x in enumerate(tasks)
            if i in task_idx
        ]
        LOG.info("Will tune %d kernels (idx: %s)", len(tasks), ", ".join(
            [str(x) for x in task_idx]
        ))
    else:
        LOG.info("Will tune all %d kernels", len(tasks))
    runner_opt = {}
    if max_tflops is not None:
        runner_opt.update({"max_flop_limit": max_tflops * 1e12})
    tuning_options = {
        "log_filename": tuner_log,
        "tuner": tuner_kind,
        "n_trial": num_iter,
        "early_stopping": early_stopping,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=timeout_builder),
            runner=autotvm.LocalRunner(
                number=measure_num,
                repeat=measure_repeats,
                timeout=timeout_runner,
                min_repeat_ms=measure_min_time,
                enable_cpu_cache_flush=flush_cpu,
                **runner_opt,
            ),
        ),
        "transfer_learning": transfer_learning,
    }
    tune_kernels(tasks, **tuning_options)
    return 0


if __name__ == "__main__":
    num_threads = multiprocessing.cpu_count()
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    LOG.setLevel(logging.INFO)
    try:
        sys.exit(main(**vars(parse_args())))
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        logging.shutdown()

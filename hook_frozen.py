# -*- coding: utf-8 -*-

"""Hacks for frozen script."""

import os
import sys
import multiprocessing
import platform
import pickle
from patch_utils import patch_fn, get_full_name

assert getattr(sys, "frozen", False)

script_dir = os.path.dirname(__file__)
sources_path = os.path.join(script_dir, "sources.bin")
with open(sources_path, "rb") as f:
    known_sources = pickle.load(f)


if platform.system() == "Windows":
    paths = os.environ["PATH"].split(";")
    if script_dir not in paths and f"\"{script_dir}\"" not in paths:
        os.environ["PATH"] += f";\"{script_dir}\""


def patched_source(get_source, obj, *args, **kwargs):
    """Get function source inside frozen script."""
    fn_name = get_full_name(obj)
    if fn_name in known_sources:
        return known_sources[fn_name]
    return get_source(obj, *args, **kwargs)


patch_fn("inspect", "getsource", patched_source)

try:
    multiprocessing.freeze_support()

    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        # pylint: disable=protected-access
        mod_name = sys.argv[2]
        sys.argv = [sys.argv[0]] + sys.argv[3:]
        import runpy
        runpy._run_module_as_main(mod_name)
        sys.exit()
except KeyboardInterrupt:
    sys.exit(1)

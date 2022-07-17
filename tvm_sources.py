# -*- coding: utf-8 -*-

"""Hacks TVM to extract dynamic sources."""

import importlib
import inspect
from patch_utils import patch_fn, get_full_name


def get_tvm_sources():
    """Extract TVM sources."""
    sources = {}

    def patched_script(script, fn, *args, **kwargs):
        fn_name = get_full_name(fn)
        source = None
        try:
            source = inspect.getsource(fn)
        except IOError:
            pass
        if source is not None:
            sources[fn_name] = source
        return script(fn, *args, **kwargs)

    patch_fn("tvm.te.hybrid", "script", patched_script)
    importlib.import_module("tvm.relay")
    return sources

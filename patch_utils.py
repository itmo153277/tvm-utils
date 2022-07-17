# -*- coding: utf-8 -*-

"""Utility functions for patching."""

import importlib


def get_full_name(obj):
    """Get full object name."""
    return obj.__module__ + "." + obj.__qualname__


def patch_fn(pkg_name, fn_name, func):
    """Patch function."""
    orig_obj = importlib.import_module(pkg_name)
    mod = orig_obj
    obj_name = None
    for name in fn_name.split("."):
        mod = orig_obj
        orig_obj = getattr(orig_obj, name)
        obj_name = name

    def patched_fn(*args, **kwargs):
        return func(orig_obj, *args, **kwargs)

    setattr(mod, obj_name, patched_fn)

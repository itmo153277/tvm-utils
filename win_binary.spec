# -*- mode: python ; coding: utf-8 -*-

"""Spec for windows binaries."""


import os
import shutil
import atexit
import sys
import pickle
import importlib
import tempfile


def import_module_from_path(path, mod_name):
    """Import module by absolute path."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    sys.path.append(os.path.dirname(path))
    spec.loader.exec_module(mod)
    sys.path = sys.path[:-1]
    return mod


tmpdir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(tmpdir))

block_cipher = None
data_files = []
bin_files = []
common_excludes = ["setuptools", "distutils", "numpy.distutils"]

get_tvm_sources = import_module_from_path(
    os.path.join(os.getcwd(), "tvm_sources.py"),
    "tvm_sources",
).get_tvm_sources
known_sources = get_tvm_sources()
known_sources_path = os.path.join(tmpdir, "sources.bin")
with open(known_sources_path, "wb") as f:
    pickle.dump(known_sources, f, protocol=pickle.HIGHEST_PROTOCOL)
data_files.append((known_sources_path, "."))


def add_dep_files(pkg_name, files, target_list):
    """Add files from dependency."""
    pkg_imp = importlib.import_module(pkg_name)
    source_dir = os.path.dirname(os.path.realpath(pkg_imp.__file__))
    target_dir = pkg_name.replace(".", "/")
    for filename in files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if not os.path.isdir(source_path):
            target_path = os.path.dirname(target_path)
        target_list.append((source_path, target_path))


add_dep_files("tvm", ["tvm.dll", "tvm_runtime.dll"], bin_files)
add_dep_files("tvm.relay", ["std"], data_files)
add_dep_files("xgboost", ["VERSION"], data_files)
add_dep_files("xgboost", ["lib/xgboost.dll"], bin_files)

tvm_tune_analysis = Analysis(
    ["tvm_tune.py"],
    pathex=[],
    binaries=bin_files,
    datas=data_files,
    hiddenimports=["patch_utils",
                   "tvm.exec.popen_worker"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["hook_frozen.py"],
    excludes=common_excludes + [],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
tvm_compile_analysis = Analysis(
    ["tvm_compile.py"],
    pathex=[],
    binaries=bin_files,
    datas=data_files,
    hiddenimports=["patch_utils",
                   "tvm.exec.popen_worker"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["hook_frozen.py"],
    excludes=common_excludes + [],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
MERGE(
    (tvm_tune_analysis, "tvm_tune", "tvm_tune"),
    (tvm_compile_analysis, "tvm_compile", "tvm_tcompile"),
)

tvm_tune_pyz = PYZ(
    tvm_tune_analysis.pure,
    tvm_tune_analysis.zipped_data,
    cipher=block_cipher
)
tvm_compile_pyz = PYZ(
    tvm_compile_analysis.pure,
    tvm_compile_analysis.zipped_data,
    cipher=block_cipher
)

tvm_tune_exe = EXE(
    tvm_tune_pyz,
    tvm_tune_analysis.scripts,
    [],
    exclude_binaries=True,
    name="tvm_tune",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='tvm-logo-square.ico',
)
tvm_compile_exe = EXE(
    tvm_compile_pyz,
    tvm_compile_analysis.scripts,
    [],
    exclude_binaries=True,
    name="tvm_compile",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='tvm-logo-square.ico',
)

coll = COLLECT(
    tvm_tune_exe,
    tvm_tune_analysis.binaries,
    tvm_tune_analysis.zipfiles,
    tvm_tune_analysis.datas,
    tvm_compile_exe,
    tvm_compile_analysis.binaries,
    tvm_compile_analysis.zipfiles,
    tvm_compile_analysis.datas,
    strip=False,
    upx=False,
    name="tvm-utils",
)

# -*- coding: utf-8 -*-

__all__ = [
    "get_compile_args",
    "get_cache_version",
    "get_header_dirs",
    "find_librebound",
]

import os
import sys
import sysconfig
import pkg_resources

import rebound

from .rebound_pymc3_version import __version__


def get_compile_args(compiler):
    opts = ["-std=c++11", "-O2", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    return opts


def get_cache_version():
    if "dev" in __version__:
        return ()
    return tuple(map(int, __version__.split(".")))


def get_header_dirs():
    this_path = os.path.dirname(
        os.path.abspath(
            pkg_resources.resource_filename(__name__, "theano_helpers.h")
        )
    )
    rebound_path = os.path.dirname(
        os.path.abspath(
            pkg_resources.resource_filename("rebound", "rebound.h")
        )
    )
    return [this_path, rebound_path]


def find_librebound():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix is None:
        suffix = ".so"
    path = os.path.dirname(os.path.dirname(rebound.__file__))
    path = os.path.join(path, "librebound" + suffix)
    if not os.path.exists(path):
        raise RuntimeError("can't find librebound")
    return path

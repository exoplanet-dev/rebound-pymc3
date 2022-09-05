# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["IntegrateOp"]

import pkg_resources

import numpy as np

from aesara_theano_fallback import aesara
from aesara_theano_fallback.graph import basic, op, params_type
from aesara_theano_fallback import tensor as aet

from .build_utils import (
    get_compile_args,
    get_cache_version,
    get_header_dirs,
    get_librebound_path,
    get_librebound_name,
)


class IntegrateOp(op.ExternalCOp):
    params_type = params_type.ParamsType(
        t=aesara.scalar.float64,
        dt=aesara.scalar.float64,
        integrator=aesara.scalar.int32,
    )
    __props__ = ("t", "dt", "integrator")
    func_file = "./integrate.cc"
    func_name = "APPLY_SPECIFIC(integrate)"
    _INTEGRATORS = {
        "ias15": 0,
        "whfast": 1,
        "sei": 2,
        "leapfrog": 4,
        "none": 7,
        "janus": 8,
        "mercurius": 9,
    }

    def __init__(self, t=0.0, dt=0.1, integrator="ias15", **kwargs):
        self.t = float(t)
        self.dt = float(dt)
        self.integrator = self._INTEGRATORS.get(integrator.lower(), None)
        if self.integrator is None:
            raise ValueError("unknown integrator {0}".format(integrator))
        self.integrator = np.int32(self.integrator)
        super().__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_headers(self, c_compiler):
        return [
            "theano_helpers.h",
            "vector",
            "array",
            "numeric",
            "algorithm",
        ]

    def c_header_dirs(self, c_compiler):
        return [
            pkg_resources.resource_filename(__name__, "")
        ] + get_header_dirs()

    def c_compile_args(self, c_compiler):
        return get_compile_args(c_compiler)

    def c_libraries(self, c_compiler):
        return [get_librebound_name()]

    def c_lib_dirs(self, c_compiler):
        return [get_librebound_path()]

    def perform(self, node, inputs, outputs):
        assert 0

    def make_node(self, masses, initial_coords, times):
        in_args = [
            aet.as_tensor_variable(masses),
            aet.as_tensor_variable(initial_coords),
            aet.as_tensor_variable(times),
        ]
        dtype = in_args[0].dtype
        out_args = [
            aet.TensorType(dtype=dtype, broadcastable=[False] * 3)(),
            aet.TensorType(dtype=dtype, broadcastable=[False] * 5)(),
        ]
        return basic.Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return (
            list(shapes[2]) + list(shapes[0]) + [6],
            list(shapes[2]) + list(shapes[0]) + [7] + list(shapes[0]) + [6],
        )

    def grad(self, inputs, gradients):
        masses, initial_coords, times = inputs
        coords, jac = self(*inputs)
        bcoords = gradients[0]
        if not isinstance(gradients[1].type, aesara.gradient.DisconnectedType):
            raise ValueError(
                "can't propagate gradients with respect to Jacobian"
            )

        # (time, num, 6) * (time, num, 7, num, 6) -> (num, 7)
        grad = aet.sum(bcoords[:, None, None, :, :] * jac, axis=(0, 3, 4))
        return grad[:, 0], grad[:, 1:], aet.zeros_like(times)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

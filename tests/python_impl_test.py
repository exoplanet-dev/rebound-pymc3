# -*- coding: utf-8 -*-

import numpy as np
from exoplanet.theano_ops.test_tools import InferShapeTester

from aesara_theano_fallback import aesara
from aesara_theano_fallback import tensor as aet

from rebound_pymc3.python_impl import ReboundOp


class TestRebound(InferShapeTester):
    def __init__(self):
        super().__init__()
        self.op_class = ReboundOp
        self.op = ReboundOp()

    def get_args(self):
        m_val = np.array([1.3, 1e-3, 1e-5])
        x_val = np.zeros((3, 6))
        x_val[1, 0] = 15.0
        x_val[1, 4] = 0.4
        x_val[2, 0] = 100.0
        x_val[2, 4] = 0.2
        t = np.linspace(100, 1000, 12)

        m = aet.dvector()
        x = aet.dmatrix()

        f = aesara.function([m, x], self.op(m, x, t)[0])

        return t, f, [m, x], [m_val, x_val]

    def test_basic(self):
        _, f, _, in_args = self.get_args()
        f(*in_args)

    def test_infer_shape(self):
        t, f, args, arg_vals = self.get_args()
        self._compile_and_check(
            args, self.op(*(list(args) + [t])), arg_vals, self.op_class
        )

    def test_grad(self):
        t, _, _, in_args = self.get_args()
        func = lambda *args: self.op(*(list(args) + [t]))[0]  # NOQA
        aesara.gradient.verify_grad(func, in_args, n_tests=1, rng=np.random)

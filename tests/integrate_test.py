# -*- coding: utf-8 -*-

import time
import pytest
import numpy as np

from aesara_theano_fallback import aesara
from aesara_theano_fallback import tensor as aet

from rebound_pymc3.test_tools import InferShapeTester
from rebound_pymc3.python_impl import ReboundOp
from rebound_pymc3.integrate import IntegrateOp


class TestIntegrate(InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = IntegrateOp
        self.op = IntegrateOp()

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


@pytest.mark.parametrize("kwargs", [dict(), dict(integrator="whfast")])
def test_consistent_results(kwargs):
    m = np.array([1.3, 1e-3, 1e-5])
    x = np.zeros((3, 6))
    x[1, 0] = 15.0
    x[1, 4] = 0.4
    x[2, 0] = 100.0
    x[2, 4] = 0.2
    t = np.linspace(100, 1000, 12)

    v1, j1 = aesara.function([], ReboundOp(**kwargs)(m, x, t))()
    v2, j2 = aesara.function([], IntegrateOp(**kwargs)(m, x, t))()
    j1 = np.moveaxis(j1, -1, 1)
    j1 = np.moveaxis(j1, -1, 1)

    assert np.allclose(v1, v2)
    assert np.allclose(j1, j2)


def test_performance(K=10):
    m = np.array([1.3, 1e-3, 1e-5])
    x = np.zeros((3, 6))
    x[1, 0] = 15.0
    x[1, 4] = 0.4
    x[2, 0] = 100.0
    x[2, 4] = 0.2
    t_tensor = aet.dvector()
    t = np.linspace(100, 1000, 100)

    f1 = aesara.function([t_tensor], ReboundOp()(m, x, t_tensor))
    f2 = aesara.function([t_tensor], IntegrateOp()(m, x, t_tensor))

    f1(t)
    f2(t)

    strt = time.time()
    for k in range(K):
        f1(t)
    time1 = time.time() - strt

    strt = time.time()
    for k in range(K):
        f2(t)
    time2 = time.time() - strt

    assert time1 > time2

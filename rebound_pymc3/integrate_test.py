# -*- coding: utf-8 -*-

import time
import pytest
import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .python_impl import ReboundOp
from .integrate import IntegrateOp


class TestIntegrate(utt.InferShapeTester):
    def setUp(self):
        super(TestIntegrate, self).setUp()
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

        m = tt.dvector()
        x = tt.dmatrix()

        f = theano.function([m, x], self.op(m, x, t)[0])

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
        utt.verify_grad(func, in_args, n_tests=1)


@pytest.mark.parametrize("kwargs", [dict(), dict(integrator="whfast")])
def test_consistent_results(kwargs):
    m = np.array([1.3, 1e-3, 1e-5])
    x = np.zeros((3, 6))
    x[1, 0] = 15.0
    x[1, 4] = 0.4
    x[2, 0] = 100.0
    x[2, 4] = 0.2
    t = np.linspace(100, 1000, 12)

    v1, j1 = theano.function([], ReboundOp(**kwargs)(m, x, t))()
    v2, j2 = theano.function([], IntegrateOp(**kwargs)(m, x, t))()
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
    t_tensor = tt.dvector()
    t = np.linspace(100, 1000, 100)

    f1 = theano.function([t_tensor], ReboundOp()(m, x, t_tensor))
    f2 = theano.function([t_tensor], IntegrateOp()(m, x, t_tensor))

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

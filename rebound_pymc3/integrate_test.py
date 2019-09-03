# -*- coding: utf-8 -*-

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

    # def test_grad(self):
    #     t, _, _, in_args = self.get_args()
    #     func = lambda *args: self.op(*(list(args) + [t]))[0]  # NOQA
    #     utt.verify_grad(func, in_args, n_tests=1)


@pytest.mark.parametrize("kwargs", [dict(), dict(integrator="whfast")])
def test_consistent_results(kwargs):
    m = np.array([1.3, 1e-3, 1e-5])
    x = np.zeros((3, 6))
    x[1, 0] = 15.0
    x[1, 4] = 0.4
    x[2, 0] = 100.0
    x[2, 4] = 0.2
    t = np.linspace(100, 1000, 12)

    results1 = ReboundOp()(m, x, t)[0].eval()
    results2 = IntegrateOp()(m, x, t)[0].eval()

    assert np.allclose(results1, results2)

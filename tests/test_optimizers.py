import unittest

from tiny_optim.demo import (
    loss_and_grads,
    make_data,
    make_matrix_data,
    matrix_loss_and_grads,
)
from tiny_optim.optimizers import (
    Adam,
    Momentum,
    Muon,
    SGD,
    _matmul,
    _transpose,
    newton_schulz5,
)


def run_quadratic(opt, steps=200):
    # minimize f(p) = (p0 - 1)^2 + (p1 + 2)^2, minimum at [1, -2]
    params = [0.0, 0.0]
    for _ in range(steps):
        grads = [2 * (params[0] - 1), 2 * (params[1] + 2)]
        params = opt.step(params, grads)
    return params


class TestOptimizers(unittest.TestCase):
    def test_sgd_reduces_loss(self):
        xs, ys = make_data()
        params = [0.0, 0.0]
        before, grads = loss_and_grads(params, xs, ys)
        params = SGD(lr=0.05).step(params, grads)
        after, _ = loss_and_grads(params, xs, ys)
        self.assertLess(after, before)

    def test_zero_gradient_is_noop(self):
        for opt in (SGD(), Momentum(), Adam()):
            out = opt.step([1.0, -2.0], [0.0, 0.0])
            self.assertEqual(out, [1.0, -2.0])

    def test_all_converge_near_minimum(self):
        for opt in (SGD(lr=0.1), Momentum(lr=0.05, mu=0.9), Adam(lr=0.05)):
            p = run_quadratic(opt)
            self.assertAlmostEqual(p[0], 1.0, delta=0.05)
            self.assertAlmostEqual(p[1], -2.0, delta=0.05)


class TestMuon(unittest.TestCase):
    def test_newton_schulz_orthogonalizes(self):
        # wide matrix: after NS5, X X^T should be close to the identity.
        # The quintic coefficients are tuned for 5 steps: singular values
        # land in a band around 1 rather than exactly on it, so assert the
        # band (and a large improvement over the raw matrix), not exactness.
        m = [
            [0.5, -1.0, 2.0, 0.25, 1.0],
            [1.5, 0.0, -0.5, 2.0, -1.0],
            [-0.25, 1.0, 1.0, 0.75, 0.5],
        ]
        out = newton_schulz5(m)
        gram = _matmul(out, _transpose(out))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(gram[i][j], 1.0 if i == j else 0.0,
                                       delta=0.35)

    def test_muon_zero_gradient_is_noop(self):
        params = [[1.0, 2.0], [3.0, 4.0]]
        grads = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(Muon().step(params, grads), params)

    def test_muon_reduces_matrix_loss(self):
        xs, ys = make_matrix_data()
        W = [[0.0] * 3 for _ in range(4)]
        before, _ = matrix_loss_and_grads(W, xs, ys)
        opt = Muon(lr=0.05, momentum=0.95)
        for _ in range(100):
            _, grads = matrix_loss_and_grads(W, xs, ys)
            W = opt.step(W, grads)
        after, _ = matrix_loss_and_grads(W, xs, ys)
        self.assertLess(after, 0.25 * before)


if __name__ == "__main__":
    unittest.main()

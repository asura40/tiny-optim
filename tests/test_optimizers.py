import unittest

from tiny_optim.demo import loss_and_grads, make_data
from tiny_optim.optimizers import Adam, Momentum, SGD


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


if __name__ == "__main__":
    unittest.main()

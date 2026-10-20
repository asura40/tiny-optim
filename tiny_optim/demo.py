"""Fit y = w*x + b to synthetic data with each optimizer and compare."""

import random

from .optimizers import Adam, Momentum, SGD


def make_data(n=64, w_true=3.0, b_true=2.0, noise=0.1, seed=0):
    rng = random.Random(seed)
    xs = [rng.uniform(-2, 2) for _ in range(n)]
    ys = [w_true * x + b_true + rng.gauss(0, noise) for x in xs]
    return xs, ys


def loss_and_grads(params, xs, ys):
    w, b = params
    n = len(xs)
    loss = 0.0
    gw = 0.0
    gb = 0.0
    for x, y in zip(xs, ys):
        err = w * x + b - y
        loss += err * err
        gw += 2 * err * x
        gb += 2 * err
    return loss / n, [gw / n, gb / n]


def train(opt, steps=300):
    xs, ys = make_data()
    params = [0.0, 0.0]
    for _ in range(steps):
        _, grads = loss_and_grads(params, xs, ys)
        params = opt.step(params, grads)
    loss, _ = loss_and_grads(params, xs, ys)
    return params, loss


def main():
    configs = [
        ("SGD lr=0.05", SGD(lr=0.05)),
        ("Momentum lr=0.05 mu=0.9", Momentum(lr=0.05, mu=0.9)),
        ("Adam lr=0.05", Adam(lr=0.05)),
    ]
    print("target: w=3.0, b=2.0")
    for name, opt in configs:
        (w, b), loss = train(opt)
        print(f"{name:26s} final loss={loss:.6f}  w={w:.3f}  b={b:.3f}")


if __name__ == "__main__":
    main()

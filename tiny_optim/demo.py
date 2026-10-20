"""Fit y = w*x + b to synthetic data with each optimizer and compare,
then fit a small matrix regression with Muon vs (flattened) Momentum."""

import random

from .optimizers import Adam, Momentum, Muon, SGD


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


# ---------------- matrix regression (for Muon) ----------------

W_TRUE = [
    [1.0, -2.0, 0.5],
    [0.0, 1.5, -1.0],
    [2.0, 0.5, 1.0],
    [-1.0, 0.0, 2.0],
]  # 4x3


def make_matrix_data(n=48, noise=0.05, seed=1):
    rng = random.Random(seed)
    xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
    ys = [
        [sum(w * x for w, x in zip(w_row, x)) + rng.gauss(0, noise)
         for w_row in W_TRUE]
        for x in xs
    ]
    return xs, ys


def matrix_loss_and_grads(W, xs, ys):
    n = len(xs)
    rows, cols = len(W), len(W[0])
    loss = 0.0
    grads = [[0.0] * cols for _ in range(rows)]
    for x, y in zip(xs, ys):
        for i in range(rows):
            pred = sum(W[i][j] * x[j] for j in range(cols))
            err = pred - y[i]
            loss += err * err
            for j in range(cols):
                grads[i][j] += 2 * err * x[j]
    scale = 1.0 / (n * rows)
    return loss * scale, [[g * scale for g in row] for row in grads]


def _flatten(m):
    return [v for row in m for v in row]


def _unflatten(flat, rows, cols):
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


def train_matrix(opt, steps=100, matrix=False):
    xs, ys = make_matrix_data()
    rows, cols = len(W_TRUE), len(W_TRUE[0])
    W = [[0.0] * cols for _ in range(rows)]
    for _ in range(steps):
        _, grads = matrix_loss_and_grads(W, xs, ys)
        if matrix:
            W = opt.step(W, grads)
        else:
            W = _unflatten(opt.step(_flatten(W), _flatten(grads)), rows, cols)
    loss, _ = matrix_loss_and_grads(W, xs, ys)
    return W, loss


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

    print("\nmatrix regression: learn a 4x3 weight matrix")
    _, loss_mom = train_matrix(Momentum(lr=0.05, mu=0.9), steps=300)
    _, loss_muon = train_matrix(Muon(lr=0.05, momentum=0.95), steps=300,
                                matrix=True)
    print(f"{'Momentum (flattened)':26s} final loss={loss_mom:.6f}")
    print(f"{'Muon lr=0.05':26s} final loss={loss_muon:.6f}")


if __name__ == "__main__":
    main()

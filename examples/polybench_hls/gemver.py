# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32
import allo.ir.types as T

import sys

def gemver_np(A, u1, u2, v1, v2, x, y, w, z, alpha, beta):
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in range(N):
        for j in range(N):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in range(N):
        x[i] = x[i] + z[i]

    for i in range(N):
        for j in range(N):
            w[i] = w[i] + alpha * A[i, j] * x[j]


def gemver(concrete_type, n, alpha=10, beta=10):
    def kernel_gemver[
        T: (int32, int32), N: int32
    ](
        A: "T[N, N]",
        u1: "T[N]",
        u2: "T[N]",
        v1: "T[N]",
        v2: "T[N]",
        x: "T[N]",
        y: "T[N]",
        w: "T[N]",
        z: "T[N]",
    ):
        for i, j in allo.grid(N, N):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

        for i, j in allo.grid(N, N):
            x[i] = x[i] + beta * A[j, i] * y[j]

        for i in allo.grid(N):
            x[i] = x[i] + z[i]

        for i, j in allo.grid(N, N):
            w[i] = w[i] + alpha * A[i, j] * x[j]

    s0 = allo.customize(kernel_gemver, instantiate=[concrete_type, n])
    return s0

def test_gemver(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "mini"
    N = psize["gemver"][test_psize]["N"]
    concrete_type = int32
    alpha = 10
    beta = 10
    s0 = gemver(concrete_type, N, alpha, beta)
    if print_hls:
        # printing hls instead
        mod = s0.build(target="vhls")
        print(mod)
        return
    mod = s0.build()

    # generate input data
    A = np.random.randint(1, 10, size=(N,N)).astype(np.int32)
    u1 = np.random.randint(1, 10, size=(N)).astype(np.int32)
    u2 = np.random.randint(1, 10, size=(N)).astype(np.int32)
    v1 = np.random.randint(1, 10, size=(N)).astype(np.int32)
    v2 = np.random.randint(1, 10, size=(N)).astype(np.int32)
    x = np.random.randint(1, 10, size=(N)).astype(np.int32)
    y = np.random.randint(1, 10, size=(N)).astype(np.int32)
    w = np.random.randint(1, 10, size=(N)).astype(np.int32)
    z = np.random.randint(1, 10, size=(N)).astype(np.int32)
    A_golden = A.copy()
    y_golden = y.copy()
    x_golden = x.copy()
    w_golden = w.copy()

    gemver_np(A_golden, u1, u2, v1, v2, x_golden, y_golden, w_golden, z, alpha, beta)
    mod(A, u1, u2, v1, v2, x, y, w, z)

    np.testing.assert_allclose(A, A_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(y, y_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(x, x_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(w, w_golden, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_gemver(print_hls=True)
    else:
        pytest.main([__file__])

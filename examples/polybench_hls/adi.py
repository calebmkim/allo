# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
import os
import json
import numpy as np
import allo.ir.types as T
from allo.ir.types import int32, index

import sys


def adi_np(u, v, p, q, TSTEPS, N):
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = int(-mul1 / 2.0)
    b = int(1.0 + mul1)
    c = int(a)
    d = int(-mul2 / 2.0)
    e = int(1.0 + mul2)
    f = int(d)

    for t in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            v[0][i] = 1.0
            p[i][0] = 0.0
            q[i][0] = v[0][i]
            for j in range(1, N - 1):
                p[i][j] = int(-c / (a * p[i][j - 1] + b))
                q[i][j] = int((
                    -d * u[j][i - 1]
                    + (1.0 + 2.0 * d) * u[j][i]
                    - f * u[j][i + 1]
                    - a * q[i][j - 1]
                ) / (a * p[i][j - 1] + b))

            v[N - 1][i] = 1.0
            for j in range(N - 2, 0, -1):
                v[j][i] = int(p[i][j] * v[j + 1][i] + q[i][j])
        for i in range(1, N - 1):
            u[i][0] = 1.0
            p[i][0] = 0.0
            q[i][0] = u[i][0]
            for j in range(1, N - 1):
                p[i][j] = int(-f / (d * p[i][j - 1] + e))
                q[i][j] = int((
                    -a * v[i - 1][j]
                    + (1.0 + 2.0 * a) * v[i][j]
                    - c * v[i + 1][j]
                    - d * q[i][j - 1]
                ) / (d * p[i][j - 1] + e))
            u[i][N - 1] = int(1.0)
            for j in range(N - 2, 0, -1):
                u[i][j] = int(p[i][j] * u[i][j + 1] + q[i][j])


def adi(ttype, TSTEPS, N):
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = int(-mul1 / 2.0)
    b = int(1.0 + mul1)
    c = int(a)
    d = int(-mul2 / 2.0)
    e = int(1.0 + mul2)
    f = int(d)

    def kernel_adi[
        T: (int32, int32), TSTEPS: int32, N: int32
    ](u: "T[N, N]", v: "T[N, N]", p: "T[N, N]", q: "T[N, N]"):
        for t in range(1, TSTEPS + 1):
            for i in range(1, N - 1):
                v[0, i] = 1
                p[i, 0] = 0
                q[i, 0] = v[0, i]
                for j in range(1, N - 1):
                    p[i, j] = int(-c / int(a * p[i, j - 1] + b))
                    q[i, j] = int((
                        int(-d * u[j, i - 1])
                        + int((1 + 2 * d) * u[j, i])
                        - int(f * u[j, i + 1])
                        - int(a * q[i, j - 1])
                    ) / int(a * p[i, j - 1] + b))

                v[N - 1, i] = int(1.0)
                for j_rev in range(N - 1):
                    j: index = N - 2 - j_rev
                    v[j, i] = int(p[i, j] * v[j + 1, i] + q[i, j])
            for i in range(1, N - 1):
                u[i, 0] = 1
                p[i, 0] = 0
                q[i, 0] = u[i, 0]
                for j in range(1, N - 1):
                    p[i, j] = int(-f / int(d * p[i, j - 1] + e))
                    q[i, j] = int(int(
                        -a * v[i - 1, j]
                        + (1 + 2 * a) * v[i, j]
                        - c * v[i + 1, j]
                        - d * q[i, j - 1]
                    ) / int(d * p[i, j - 1] + e))
                u[i, N - 1] = 1
                for j_rev in range(N - 1):
                    j: index = N - 2 - j_rev
                    u[i, j] = int(p[i, j] * u[i, j + 1] + q[i, j])

    s = allo.customize(kernel_adi, instantiate=[ttype, TSTEPS, N])
    return s


def test_adi(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use mini problem size
    test_psize = "mini"

    TSTEPS = psize["adi"][test_psize]["TSTEPS"]
    N = psize["adi"][test_psize]["N"]

    u = np.random.randint(0, 100, (N, N)).astype(np.int32)
    v = np.random.randint(0, 100, (N, N)).astype(np.int32)
    p = np.random.randint(0, 100, (N, N)).astype(np.int32)
    q = np.random.randint(0, 100, (N, N)).astype(np.int32)

    u_ref = u.copy()
    v_ref = v.copy()
    p_ref = p.copy()
    q_ref = q.copy()

    adi_np(u_ref, v_ref, p_ref, q_ref, TSTEPS, N)

    s = adi(int32, TSTEPS, N)
    if print_hls:
        # printing hls instead
        mod = s.build(target="vhls")
        print(mod)
        return
    mod = s.build()
    mod(u, v, p, q)

    np.testing.assert_allclose(u, u_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v, v_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(p, p_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(q, q_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_adi(print_hls=True)
    else:
        pytest.main([__file__])

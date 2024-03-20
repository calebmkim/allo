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


def atax_np(A, x):
    out_Ax = np.dot(A, x)
    y = np.dot(A.T, out_Ax)
    return y


def atax(concrete_type, m, n):
    def stage_M[
        T: (int32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", out_Ax: "T[M]"):
        for m in allo.grid(M):
            for r in allo.reduction(N):
                out_Ax[m] += A[m, r] * x[r]

    def stage_N[
        T: (int32, int32), M: int32, N: int32
    ](A: "T[M, N]", out_Ax: "T[M]", y: "T[N]"):
        for n in allo.grid(N):
            for k in allo.reduction(M):
                y[n] += A[k, n] * out_Ax[k]

    def kernel_atax[
        T: (int32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", y: "T[N]"):
        out_Ax: T[M] = 0
        stage_M[T, M, N](A, x, out_Ax)
        stage_N[T, M, N](A, out_Ax, y)

    sch0 = allo.customize(stage_M, instantiate=[concrete_type, m, n])
    sch0.reorder("r", "m")
    sch0.pipeline("m")
    # unroll factor 39

    sch1 = allo.customize(stage_N, instantiate=[concrete_type, m, n])
    sch1.reorder("k", "n")
    sch1.pipeline("n")
    # unroll factor 41

    sch = allo.customize(kernel_atax, instantiate=[concrete_type, m, n])
    sch.compose(sch0)
    sch.compose(sch1)

    return sch


def test_atax(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["atax"][test_psize]["M"]
    N = psize["atax"][test_psize]["N"]
    concrete_type = int32
    sch = atax(concrete_type, M, N)
    if print_hls:
        # printing hls instead
        mod = sch.build(target="vhls")
        print(mod)
        return
    mod = sch.build()
    A = np.random.randint(100, size=(M, N))
    x = np.random.randint(100, size=N)
    y = np.zeros((N,), dtype=np.int32)
    y_ref = atax_np(A, x)
    mod(A, x, y)
    assert np.allclose(y, y_ref, atol=1e-5, rtol=1e-3)

if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_atax(print_hls=True)
    else:
        pytest.main([__file__])




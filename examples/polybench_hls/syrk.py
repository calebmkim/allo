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

def syrk_np(A, C, alpha, beta):
    for i in range(A.shape[0]):
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(A.shape[1]):
            for j in range(i + 1):
                C[i, j] += alpha * A[i, k] * A[j, k]


def syrk(concrete_type, M, N, alpha=15, beta=12):
    def update_C[T: (int32, int32), N: int32](Cin: "T[N, N]", Cout: "T[N, N]"):
        for i0, j0 in allo.grid(N, N, name="update"):
            if j0 <= i0:
                Cout[i0, j0] = beta * Cin[i0, j0]
            else:
                Cout[i0, j0] = Cin[i0, j0]

    def compute_sum[
        T: (int32, int32), N: int32, M: int32
    ](A: "T[N, M]", A_copy: "T[N, M]", Cin: "T[N, N]", Cout: "T[N, N]"):
        buffer: T[N, N]
        for i0, j0 in allo.grid(N, N, name="load"):
            buffer[i0, j0] = Cin[i0, j0]
        for i1, k1, j1 in allo.grid(N, M, N, name="sum"):
            if j1 <= i1:
                buffer[i1, j1] += alpha * A[i1, k1] * A_copy[j1, k1]
        for i2, j2 in allo.grid(N, N, name="store"):
            Cout[i2, j2] = buffer[i2, j2]

    def kernel_syr2k[
        T: (int32, int32), N: int32, M: int32
    ](A: "T[N, M]", A_copy: "T[N, M]", Cin: "T[N, N]", Cout: "T[N, N]"):
        C: T[N, N]
        update_C[T, N](Cin, C)
        compute_sum[T, N, M](A, A_copy, C, Cout)

    sch0 = allo.customize(update_C, instantiate=[concrete_type, N])
    sch0.pipeline("i0")
    sch0.partition(sch0.Cin, dim=2)
    sch0.partition(sch0.Cout, dim=2)

    sch1 = allo.customize(compute_sum, instantiate=[concrete_type, N, M])
    sch1.pipeline("i0")
    sch1.partition(sch1.buffer, dim=2)
    sch1.pipeline("k1")
    sch1.partition(sch1.A_copy, dim=1)
    sch1.pipeline("i2")
    sch1.partition(sch1.Cout, dim=2)

    sch = allo.customize(kernel_syr2k, instantiate=[concrete_type, N, M])
    sch.compose(sch0)
    sch.compose(sch1)
    return sch


def test_syrk(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["syrk"][test_psize]["M"]
    N = psize["syrk"][test_psize]["N"]
    concrete_type = int32
    alpha = 15
    beta = 12
    sch = syrk(concrete_type, M, N, alpha, beta)
    A = np.random.randint(0, 10, (N, M)).astype(np.int32)
    C = np.random.randint(0, 10, (N, N)).astype(np.int32)
    C_golden = np.copy(C)
    syrk_np(A, C_golden, alpha, beta)
    if print_hls:
        # printing hls instead
        mod = sch.build(target="vhls")
        print(mod)
        return
    mod = sch.build()
    mod(A.copy(), A.copy(), C.copy(), C)
    np.testing.assert_allclose(C, C_golden)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_syrk(print_hls=True)
    else:
        pytest.main([__file__])

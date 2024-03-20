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

def seidel_2d_np(A, TSTEPS):
    for t in range(TSTEPS):
        for i in range(1, A.shape[0] - 1):
            for j in range(1, A.shape[1] - 1):
                A[i, j] = int(
                    A[i - 1, j - 1]
                    + A[i - 1, j]
                    + A[i - 1, j + 1]
                    + A[i, j - 1]
                    + A[i, j]
                    + A[i, j + 1]
                    + A[i + 1, j - 1]
                    + A[i + 1, j]
                    + A[i + 1, j + 1]
                ) / 9


def seidel_2d(concrete_type, TSTEPS, N):
    def kernel_seidel_2d[T: (int32, int32), TSTEPS: int32, N: int32](A: "T[N, N]"):
        for t in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    A[i, j] = int((
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9)

    s = allo.customize(kernel_seidel_2d, instantiate=[concrete_type, TSTEPS, N])
    return s


def test_seidel_2d(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    TSTEPS = psize["seidel_2d"][test_psize]["TSTEPS"]
    N = psize["seidel_2d"][test_psize]["N"]
    concrete_type = int32
    s = seidel_2d(concrete_type, TSTEPS, N)
    if print_hls:
        # printing hls instead
        mod = s.build(target="vhls")
        print(mod)
        return
    mod = s.build()
    # functional correctness test
    A = np.random.randint(10, size=(N, N)).astype(np.int32)
    A_ref = A.copy()
    mod(A)
    seidel_2d_np(A_ref, TSTEPS)
    np.testing.assert_allclose(A, A_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_seidel_2d(print_hls=True)
    else:
        pytest.main([__file__])

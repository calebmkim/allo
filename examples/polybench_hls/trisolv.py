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

def trisolv_np(L, x, b):
    N = L.shape[0]
    for i in range(N):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] = int(x[i] / L[i, i])


def trisolv(concrete_type, n):
    def kernel_trisolv[
        T: (int32, int32), N: int32
    ](L: int32[N, N], b: int32[N], x: int32[N]):
        for i in range(N):
            x[i] = b[i]
            for j in range(i):
                x[i] -= L[i, j] * x[j]
            x[i] = int(x[i] / L[i, i])

    s0 = allo.customize(kernel_trisolv, instantiate=[concrete_type, n])
    return s0


def test_trisolv(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"

    # generate input data
    N = psize["trisolv"][test_psize]["N"]
    L = np.random.randint(1, 10, size=(N, N)).astype(np.int32)
    b = np.random.randint(1, 10, size=(N,)).astype(np.int32)

    # run reference
    x_ref = np.zeros_like(b)
    trisolv_np(L, x_ref, b)

    # run allo
    x = np.zeros_like(b)
    s0 = trisolv(int32, N)
    if print_hls:
        # printing hls instead
        mod = s0.build(target="vhls")
        print(mod)
        return
    mod = s0.build()
    mod(L, b, x)

    # verify
    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_trisolv(print_hls=True)
    else:
        pytest.main([__file__])

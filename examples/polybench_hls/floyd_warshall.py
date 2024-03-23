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


def floyd_warshall_np(path):
    N = path.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j]) + 1


def floyd_warshall(concrete_type, N):
    def kernel_floyd_warshall(path: int32[N, N]):
        for k, i, j in allo.grid(N, N, N):
            path_: int32 = path[i, k] + path[k, j]
            if path[i, j] >= path_:
                path[i, j] = path_

    s0 = allo.customize(kernel_floyd_warshall)
    return s0


def test_floyd_warshall(print_hls=False):
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["floyd_warshall"][test_psize]["N"]
    concrete_type = int32
    s0 = floyd_warshall(concrete_type, N)
    if print_hls:
        # printing hls instead
        mod = s0.build(target="vhls")
        print(mod)
        return
    mod = s0.build()
    path = np.random.randint(1, 10, size=(N, N)).astype(np.int32)
    path_ref = path.copy()
    floyd_warshall_np(path_ref)
    mod(path)
    np.testing.assert_allclose(path, path_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    if "-hls" in sys.argv:
        test_floyd_warshall(print_hls=True)
    else:
        pytest.main([__file__])

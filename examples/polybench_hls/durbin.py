# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32
import allo.ir.types as T

def durbin_np(r, y):
    z = np.zeros_like(y).astype(np.int32)
    N = r.shape[0]
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]
    for k in range(1, N):
        beta = (1 - alpha * alpha) * beta
        sum_ = 0.0
        for i in range(k):
            sum_ = sum_ + r[k - i - 1] * y[i]
        alpha = -1.0 * (r[k] + sum_)
        # alpha = alpha / beta
        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]
        for i in range(k):
            y[i] = z[i]
        y[k] = alpha

def durbin(concrete_type, n):
    def kernel_durbin[T: (int32, int32), N: int32](r: "T[N]", y: "T[N]"):
        y[0] = -r[0]
        beta: T = 1
        alpha: T = -r[0]

        for k in range(1, N):
            beta = (1 - alpha * alpha) * beta
            sum_: T = 0

            z: T[N] = 0
            for i in range(k):
                sum_ = sum_ + r[k - i - 1] * y[i]

            alpha = -1 * (r[k] + sum_)
            # alpha = alpha / beta # unstable

            for i in range(k):
                z[i] = y[i] + alpha * y[k - i - 1]

            for i in range(k):
                y[i] = z[i]

            y[k] = alpha

    s = allo.customize(kernel_durbin, instantiate=[concrete_type, n])
    return s


def test_durbin():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "mini"
    N = psize["durbin"][test_psize]["N"]
    concrete_type = int32
    r = np.random.randint(1, 10, size=(N,)).astype(np.int32)
    y = np.random.randint(1, 10, size=(N,)).astype(np.int32)
    y_golden = y.copy().astype(np.int32)
    durbin_np(r, y_golden)
    s = durbin(concrete_type, N)
    mod = s.build()
    mod(r.copy(), y)
    np.testing.assert_allclose(y, y_golden, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])

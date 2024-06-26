# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import int32


def covariance_np(data, mean, cov, M, N):
    for j in range(M):
        mean[j] = 0.0
        for i in range(N):
            mean[j] += data[i, j]
        mean[j] /= int(N)
    for i in range(N):
        for j in range(M):
            data[i, j] -= mean[j]

    for i in range(M):
        for j in range(M):
            cov[i, j] = 0.0
            for k in range(N):
                cov[i, j] += data[k, i] * data[k, j]
            cov[i, j] /= int(N - 1)
            cov[j, i] = cov[i, j]


def covariance(type, m, n):
    def kernel_covariance[
        T: (int32, int32), M: int32, N: int32
    ](data: "T[N, M]", mean: "T[M]", cov: "T[M, M]"):
        # Compute mean
        for x in allo.grid(M):
            total: int32 = 0.0
            for k in allo.grid(N):
                total += data[k, x]
            mean[x] = int(total / N)

        # Compute covariance
        for i, j in allo.grid(M, M):
            covariance: int32 = 0
            for p in allo.grid(N):
                covariance += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
            cov[i, j] = int(covariance / (N - 1))

    s = allo.customize(kernel_covariance, instantiate=[type, m, n])
    return s


def test_covariance():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["covariance"][test_psize]["M"]
    N = psize["covariance"][test_psize]["N"]
    mod = covariance(int32, M, N)
    data = np.random.randint(-10, 10, (N, M)).astype(np.int32)
    mean = np.zeros((M,), dtype=np.int32)
    cov = np.zeros((M, M), dtype=np.int32)
    mean_ref = np.zeros((M,), dtype=np.int32)
    cov_ref = np.zeros((M, M), dtype=np.int32)
    covariance_np(data.copy(), mean_ref, cov_ref, M, N)
    mod(data, mean, cov)
    np.testing.assert_allclose(mean, mean_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-5, atol=1e-5)

def test_covariance():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["covariance"][test_psize]["M"]
    N = psize["covariance"][test_psize]["N"]
    s = covariance(int32, M, N)
    mod = s.build()
    data = np.random.randint(-10, 10, (N, M)).astype(np.int32)
    mean = np.zeros((M,), dtype=np.int32)
    cov = np.zeros((M, M), dtype=np.int32)
    mean_ref = np.zeros((M,), dtype=np.int32)
    cov_ref = np.zeros((M, M), dtype=np.int32)
    covariance_np(data.copy(), mean_ref, cov_ref, M, N)
    mod(data, mean, cov)
    np.testing.assert_allclose(mean, mean_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-5, atol=1e-5)

def print_covariance():
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["covariance"][test_psize]["M"]
    N = psize["covariance"][test_psize]["N"]
    s = covariance(int32, M, N)
    mod = s.build(target="vhls")
    print(mod)

if __name__ == "__main__":
    pytest.main([__file__])

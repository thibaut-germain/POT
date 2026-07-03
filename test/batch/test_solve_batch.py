"""Tests for module batch"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Paul Krzakala <paul.krzakala@gmail.com>
#         Sonia Mazelet <sonia.mazelet@polytechnique.edu>
#         Thibaut Germain <thibaut.germain.pro@gmail.com>
#
# License: MIT License

import numpy as np
from ot.batch import (
    solve_batch,
    solve_sample_batch,
    dist_batch,
    loss_linear_samples_batch,
    loss_linear_batch,
    bregman_projection_batch,
    bregman_log_projection_batch,
    proximal_bregman_log_plan_batch,
)

from ot import solve
import pytest
from ot.backend import torch


@pytest.mark.parametrize("solver", ["sinkhorn", "log_sinkhorn"])
@pytest.mark.parametrize("reg_type", ["kl", "entropy"])
def test_sinkhorn_solve_batch(solver, reg_type):
    """Check that solve_batch gives the same results as solve for each instance in the batch."""
    batchsize = 4
    n = 16
    rng = np.random.RandomState(0)

    M = rng.rand(batchsize, n, n)

    reg = 0.1
    max_iter = 10000
    tol = 1e-5

    res = solve_batch(
        M,
        a=None,
        b=None,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        solver=solver,
        reg_type=reg_type,
        grad="detach",
    )
    plan_batch = res.plan
    values_batch = res.value_linear

    for i in range(batchsize):
        M_i = M[i]
        res_i = solve(
            M_i, a=None, b=None, reg=reg, max_iter=max_iter, tol=tol, reg_type=reg_type
        )
        plan_i = res_i.plan
        value_i = res_i.value_linear
        np.testing.assert_allclose(plan_i, plan_batch[i], atol=1e-05)
        np.testing.assert_allclose(value_i, values_batch[i], atol=1e-4)


def test_proximal_solve_batch():
    """Check that proximal_bregman_log_plan_batch gives the same results as solve for each instance in the batch."""
    batchsize = 3
    n = 5
    d = 7
    rng = np.random.RandomState(0)
    C = rng.rand(batchsize, n, d)

    exact_plan = np.zeros((batchsize, n, d))
    exact_value = np.zeros(batchsize)
    for i in range(batchsize):
        C_i = C[i]
        res_i = solve(C_i, reg=None, tol=1e-5)
        exact_plan[i] = res_i.plan
        exact_value[i] = res_i.value_linear

    configs = [
        {"reg": None, "solver": "proximal"},
        {"reg": None, "solver": "log_sinkhorn"},
        {"reg": None, "solver": "sinkhorn"},
        {"reg": 0, "solver": "proximal"},
        {"reg": 1e-2, "solver": "proximal"},
    ]

    for config in configs:
        res = solve_batch(C, max_iter=10000, tol=1e-5, grad="detach", **config)
        plan = res["T"]
        value = res["value_linear"]
        np.testing.assert_allclose(plan, exact_plan, atol=1e-5)
        np.testing.assert_allclose(value, exact_value, atol=1e-4)


@pytest.mark.parametrize("inner_iter", [1, 5, 10])
def test_proximal_bregman_log_plan_batch(inner_iter):
    batchsize = 3
    n = 5
    d = 7
    rng = np.random.RandomState(0)
    C = rng.rand(batchsize, n, d)
    res = proximal_bregman_log_plan_batch(
        C, reg=1e-2, max_iter=10000, tol=1e-5, inner_iter=inner_iter, grad="detach"
    )
    plan = res["T"]
    for i in range(batchsize):
        C_i = C[i]
        res_i = solve(C_i, reg=None, tol=1e-5)
        plan_i = res_i.plan
        np.testing.assert_allclose(plan_i, plan[i], atol=1e-5)


def test_bregman_batch():
    batchsize = 4
    d = 2
    n = 4
    rng = np.random.RandomState(0)
    X = rng.rand(batchsize, n, d)
    M = dist_batch(X, X)
    K = np.exp(-M / 0.01)
    log_K = -M / 0.01
    res = bregman_projection_batch(K, max_iter=50, tol=1e-10)
    plan = res["T"]
    res_log = bregman_log_projection_batch(log_K, max_iter=50, tol=1e-10)
    plan_log = res_log["T"]
    np.testing.assert_allclose(plan, plan_log, atol=1e-3)


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "minkowski", "kl"])
@pytest.mark.parametrize("solver", ["proximal", "sinkhorn", "log_sinkhorn"])
def test_sample_solve_batch(metric, solver):
    """Check that all functions run without error."""

    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    X = rng.rand(batchsize, n, d)
    if metric == "kl":
        X = np.abs(X) + 1e-6
        X = X / np.sum(X, axis=-1, keepdims=True)
    M = dist_batch(X, X, metric=metric)
    is_positive = M >= 0
    np.testing.assert_equal(is_positive.all(), True)

    # Solve sample batch
    res = solve_sample_batch(
        X, X, reg=0.1, max_iter=10, tol=1e-5, metric=metric, solver=solver
    )

    # Compute loss
    loss = res.value_linear  # loss given by solver
    loss2 = loss_linear_batch(M, res.plan)  # recompute loss from plan
    loss3 = loss_linear_samples_batch(
        X, X, res.plan, metric=metric
    )  # recompute loss from plan and samples
    np.testing.assert_allclose(loss, loss2, atol=1e-5)
    np.testing.assert_allclose(loss, loss3, atol=1e-5)


@pytest.mark.skipif(not torch, reason="torch not installed")
@pytest.mark.parametrize("grad", ["detach", "envelope", "autodiff", "last_step"])
def test_gradients_torch(grad):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    for solver in ["proximal", "sinkhorn", "log_sinkhorn"]:
        X = torch.randn((batchsize, n, d), requires_grad=True)
        M = dist_batch(X, X)
        res = solve_batch(M, reg=0.1, max_iter=10, tol=1e-5, grad=grad, solver=solver)
        loss = res.value_linear.sum()
        loss_plan = res.plan.sum()
        if grad == "detach":
            assert loss.grad == None
        elif grad == "envelope":
            loss.backward()
            assert X.grad is not None
        elif grad in ["autodiff", "last_step"]:
            loss_plan.backward()
            assert X.grad is not None


def test_backend(nx):
    """Check that all gradient methods run without error."""
    batchsize = 2
    n = 4
    d = 2
    X = np.random.randn(batchsize, n, d)
    X = nx.from_numpy(X)
    M = dist_batch(X, X)
    for solver in ["proximal", "sinkhorn", "log_sinkhorn"]:
        solve_batch(M, reg=0.1, max_iter=10, tol=1e-5, solver=solver)
        solve_sample_batch(X, X, reg=0.1, max_iter=10, tol=1e-5, solver=solver)


def test_metric_default_parameters():
    """Check that all functions with default parameters run without error."""

    batchsize = 2
    n = 4
    d = 2
    rng = np.random.RandomState(0)
    X = rng.rand(batchsize, n, d)
    M = dist_batch(X, X)
    is_positive = M >= 0
    np.testing.assert_equal(is_positive.all(), True)

    # Solve batch
    res = solve_batch(M, reg=0.1, max_iter=10, tol=1e-5)

    # Solve sample batch
    res = solve_sample_batch(X, X, reg=0.1)

    # Compute loss
    loss_linear_batch(M, res.plan)  # recompute loss from plan
    loss_linear_samples_batch(X, X, res.plan)  # recompute loss from plan and samples
    assert np.isfinite(loss_linear_batch(M, res.plan)).all()
    assert np.isfinite(loss_linear_samples_batch(X, X, res.plan)).all()

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from collections import deque
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, PyTree

from flows.anderson_acceleration_step import compute_fixed_point_residual
from functionals.functional import Potential
from geometry.G_matrix import G_matrix
from parametric_model.parametric_model import ParametricModel


def _update_lifo_history(history, item, max_size: int):
    """
    Update a bounded LIFO history where index 0 is always the newest entry.
    """
    if isinstance(history, deque):
        updated = deque(history, maxlen=max_size)
    elif history is None:
        updated = deque(maxlen=max_size)
    else:
        updated = deque(history, maxlen=max_size)
    updated.appendleft(item)
    return updated


def _tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_sub(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: x - y, a, b)


def _tree_scale(a: PyTree, scalar: float) -> PyTree:
    return jax.tree.map(lambda x: scalar * x, a)


def _tree_axpy(y: PyTree, alpha: float, x: PyTree) -> PyTree:
    return jax.tree.map(lambda yi, xi: yi + alpha * xi, y, x)


def _tree_lincomb(vectors: List[PyTree], coeffs: List[float], template: PyTree) -> PyTree:
    out = jax.tree.map(lambda x: jnp.zeros_like(x), template)
    for vec, coeff in zip(vectors, coeffs):
        out = _tree_axpy(out, float(coeff), vec)
    return out


def _tree_inf_norm(tree: PyTree) -> float:
    leaves = jax.tree.leaves(tree)
    if len(leaves) == 0:
        return 0.0
    return float(max([jnp.max(jnp.abs(leaf)) for leaf in leaves]))


def _build_orthonormal_basis(
    residual_differences: List[PyTree],
    param_differences: List[PyTree],
    G_mat: G_matrix,
    z_samples: Array,
    params: PyTree,
    tol: float,
    restart_eta: float,
) -> Tuple[List[PyTree], List[PyTree], List[float]]:
    """
    Build (q_i, u_i) orthonormalized pairs from stored differences.

    Inputs are expected in LIFO order (newest first). We process oldest->newest.
    """
    if residual_differences is None or param_differences is None:
        return [], [], []

    q_basis: List[PyTree] = []
    u_basis: List[PyTree] = []
    w_basis: List[float] = []

    for q_raw, u_raw in zip(reversed(residual_differences), reversed(param_differences)):
        q = q_raw
        u = u_raw
        raw_dx_inf = _tree_inf_norm(u_raw)
        s_proj = []
        for q_i, u_i in zip(q_basis, u_basis):
            s_ij = G_mat.inner_product(q, q_i, z_samples, params=params)
            s_ij = float(s_ij)
            s_proj.append(s_ij)
            q = _tree_axpy(q, -s_ij, q_i)
            u = _tree_axpy(u, -s_ij, u_i)

        s_jj_sq = G_mat.inner_product(q, q, z_samples, params=params)
        s_jj = float(jnp.sqrt(jnp.maximum(s_jj_sq, 0.0)))
        if s_jj <= tol:
            continue

        # Algorithm 3.1 with C = 1.
        w_j = raw_dx_inf / s_jj
        for s_ij, w_i in zip(s_proj, w_basis):
            w_j += abs(s_ij / s_jj) * w_i

        if w_j > restart_eta:
            # Restart: flush Q and U for subsequent iterations.
            q_basis = []
            u_basis = []
            w_basis = []
            continue

        q_basis.append(_tree_scale(q, 1.0 / s_jj))
        u_basis.append(_tree_scale(u, 1.0 / s_jj))
        w_basis.append(w_j)

    return q_basis, u_basis, w_basis


def aa_tgs_step(
    parametric_model: ParametricModel,
    current_params: PyTree,
    param_history: Optional[List[PyTree]],
    residual_history: Optional[List[PyTree]],
    param_diff: Optional[List[PyTree]],
    residual_diff: Optional[List[PyTree]],
    G_mat: G_matrix,
    potential: Potential,
    z_samples: Array,
    step_size: float = 0.01,
    memory_size: int = 5,
    relaxation: float = 1.0,
    anderson_tol: float = 1e-6,
    solver: str = "cg",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    l2_reg_gamma: float = 1e-6,
    restart_eta: float = float("inf"),
    graphdef: Optional[nnx.GraphDef] = None,
    outer_iter: Optional[int] = None,
    run_id: Optional[str] = None,
    method_name: Optional[str] = None,
):
    """
    AA-TGS(m) step with bounded LIFO histories.

    Notes:
      - beta_j is fixed to step_size.
      - `relaxation` and `l2_reg_gamma` are accepted for interface compatibility.
    """
    del relaxation
    del l2_reg_gamma
    del solver_maxiter

    if current_params is None or graphdef is None:
        graphdef, current_params = nnx.split(parametric_model)

    beta_j = step_size
    eps = max(float(anderson_tol), 1e-12)

    if param_history is None:
        f_0 = compute_fixed_point_residual(
            parametric_model,
            current_params,
            G_mat,
            potential,
            z_samples,
            step_size,
            solver,
            solver_tol,
            regularization,
            outer_iter=outer_iter,
            phase="bootstrap_f0",
            run_id=run_id,
            method_name=method_name,
        )
        x_1 = _tree_axpy(current_params, beta_j, f_0)
        f_1 = compute_fixed_point_residual(
            parametric_model,
            x_1,
            G_mat,
            potential,
            z_samples,
            step_size,
            solver,
            solver_tol,
            regularization,
            outer_iter=outer_iter,
            phase="bootstrap_f1",
            run_id=run_id,
            method_name=method_name,
        )

        delta_x_0 = _tree_sub(x_1, current_params)
        delta_f_0 = _tree_sub(f_1, f_0)

        return (
            deque([x_1, current_params], maxlen=memory_size + 1),
            deque([f_1, f_0], maxlen=memory_size + 1),
            deque([delta_x_0], maxlen=memory_size),
            deque([delta_f_0], maxlen=memory_size),
        )

    f_j = residual_history[0]

    # Separate current difference from the previous AA window.
    if (
        param_diff is not None
        and residual_diff is not None
        and len(param_diff) > 0
        and len(residual_diff) > 0
    ):
        u = param_diff[0]
        q = residual_diff[0]
        prev_param_diffs = list(param_diff)[1:]
        prev_residual_diffs = list(residual_diff)[1:]
    else:
        # Fallback after a restart flush: reconstruct current diff from histories.
        u = _tree_sub(param_history[0], param_history[1])
        q = _tree_sub(residual_history[0], residual_history[1])
        prev_param_diffs = []
        prev_residual_diffs = []

    q_basis_prev, u_basis_prev, w_basis_prev = _build_orthonormal_basis(
        prev_residual_diffs,
        prev_param_diffs,
        G_mat,
        z_samples,
        current_params,
        eps,
        restart_eta,
    )

    # Current raw differences: u = x_j - x_{j-1}, q = f_j - f_{j-1}
    raw_dx_inf = _tree_inf_norm(u)

    # Orthogonalize current pair against existing basis.
    s_proj_curr = []
    for q_i, u_i in zip(q_basis_prev, u_basis_prev):
        s_ij = G_mat.inner_product(q, q_i, z_samples, params=current_params)
        s_ij = float(s_ij)
        s_proj_curr.append(s_ij)
        u = _tree_axpy(u, -s_ij, u_i)
        q = _tree_axpy(q, -s_ij, q_i)

    s_jj_sq = G_mat.inner_product(q, q, z_samples, params=current_params)
    s_jj = float(jnp.sqrt(jnp.maximum(s_jj_sq, 0.0)))

    q_basis = list(q_basis_prev)
    u_basis = list(u_basis_prev)
    restart_now = False
    w_j = 0.0
    if s_jj > eps:
        # Compute w_j using Algorithm 3.1 (C = 1). Restart is applied after update.
        w_j = raw_dx_inf / s_jj
        for s_ij, w_i in zip(s_proj_curr, w_basis_prev):
            w_j += abs(s_ij / s_jj) * w_i
        restart_now = w_j > restart_eta

        q_basis.append(_tree_scale(q, 1.0 / s_jj))
        u_basis.append(_tree_scale(u, 1.0 / s_jj))

    eta = [
        float(G_mat.inner_product(q_i, f_j, z_samples, params=current_params))
        for q_i in q_basis
    ]

    proj_u = _tree_lincomb(u_basis, eta, current_params)
    proj_q = _tree_lincomb(q_basis, eta, f_j)

    f_j_minus_qeta = _tree_sub(f_j, proj_q)
    aa_delta = _tree_add(_tree_scale(proj_u, -1.0), _tree_scale(f_j_minus_qeta, beta_j))

    # Descent-direction safeguard:
    # keep AA correction only if it is aligned with the baseline fixed-point direction f_j.
    # Otherwise, fall back to the plain step and restart memory.
    alignment = float(
        G_mat.inner_product(aa_delta, f_j, z_samples, params=current_params)
    )
    if (not jnp.isfinite(alignment)) or alignment <= 0.0:
        aa_delta = _tree_scale(f_j, beta_j)
        restart_now = True

    x_next = _tree_add(current_params, aa_delta)
    f_next = compute_fixed_point_residual(
        parametric_model,
        x_next,
        G_mat,
        potential,
        z_samples,
        step_size,
        solver,
        solver_tol,
        regularization,
        outer_iter=outer_iter,
        phase="residual_eval",
        run_id=run_id,
        method_name=method_name,
    )

    delta_x_next = _tree_sub(x_next, current_params)
    delta_f_next = _tree_sub(f_next, f_j)

    new_param_history = _update_lifo_history(param_history, x_next, memory_size + 1)
    new_residual_history = _update_lifo_history(residual_history, f_next, memory_size + 1)
    new_param_diff = _update_lifo_history(param_diff, delta_x_next, memory_size)
    new_residual_diff = _update_lifo_history(residual_diff, delta_f_next, memory_size)

    # Restart affects the next iteration, so flush stored Q/U surrogates here.
    if restart_now:
        new_param_diff = deque(maxlen=memory_size)
        new_residual_diff = deque(maxlen=memory_size)

    return (
        new_param_history,
        new_residual_history,
        new_param_diff,
        new_residual_diff,
    )

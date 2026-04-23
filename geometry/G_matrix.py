import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit, vmap, grad, flatten_util
from typing import Dict, Any, Optional
import time
from jaxtyping import PyTree, Array
from functools import partial
from jax.scipy.sparse.linalg import gmres
from flax import nnx
from geometry.lin_alg_solvers import minres, reg_cg


class G_matrix:
    """
    Computation of G matrix
    """

    def __init__(self, mapping: nnx.Module):
        """
        Initialize G matrix computation

        Args:
            mapping: Neural ODE model nnx.Module instance
        """

        self.mapping = mapping
        self._linear_solve_context = {
            "run_id": None,
            "method_name": None,
            "outer_iter": None,
            "phase": "unspecified",
        }
        self._big_solve_records = []

    def clear_big_solve_records(self) -> None:
        self._big_solve_records = []

    def get_big_solve_records(self) -> list[dict]:
        return list(self._big_solve_records)

    def set_linear_solve_context(
        self,
        *,
        run_id: Optional[str] = None,
        method_name: Optional[str] = None,
        outer_iter: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> None:
        if run_id is not None:
            self._linear_solve_context["run_id"] = run_id
        if method_name is not None:
            self._linear_solve_context["method_name"] = method_name
        if outer_iter is not None:
            self._linear_solve_context["outer_iter"] = int(outer_iter)
        if phase is not None:
            self._linear_solve_context["phase"] = phase

    def _extract_solver_stats(
        self, backend_info: Any, fallback_maxiter: int
    ) -> tuple[Optional[int], Optional[bool], Optional[float], bool]:
        iterations = None
        converged = None
        residual_norm = None
        iterations_estimated = False
        if isinstance(backend_info, dict):
            iterations = backend_info.get(
                "iterations",
                backend_info.get("num_iters", backend_info.get("niter")),
            )
            converged = backend_info.get(
                "success", backend_info.get("converged", backend_info.get("ok"))
            )
            residual_norm = backend_info.get(
                "norm_res", backend_info.get("residual_norm", backend_info.get("residual"))
            )
        elif isinstance(backend_info, (int, float)):
            # JAX scipy solvers often return integer status codes (not iteration counts).
            # We treat 0 as converged and leave iteration count unknown.
            converged = bool(backend_info == 0)
        if iterations is not None:
            try:
                iterations = int(iterations)
            except (TypeError, ValueError):
                iterations = None
        if converged is not None:
            converged = bool(converged)
        if residual_norm is not None:
            try:
                residual_norm = float(residual_norm)
            except (TypeError, ValueError):
                residual_norm = None
        if iterations is None:
            iterations = int(fallback_maxiter)
            iterations_estimated = True
        return iterations, converged, residual_norm, iterations_estimated

    # @partial(jit, static_argnums=(0,))
    def mvp(
        self, z_samples: Array, eta: PyTree, params: Optional[PyTree] = None
    ) -> PyTree:
        """
        Computation of G eta
        Args:
            z_samples: (Bs,d) Samples from reference density
            eta: PyTree with same GraphDef as mapping
            parms: PyTree where the G matrix is computed at
        Return:
            G(theta) eta : PyTree
        """

        if params is None:

            _, params = nnx.split(self.mapping)

        def single_sample_contribution(z: Array) -> PyTree:

            # Define the flow map

            def flow_map(p):

                return self.mapping(z.reshape(1, -1), params=p)

            # Step 1: Compute \partial_{theta}T @ eta using Jvp

            jvp_result = jax.jvp(flow_map, (params,), (eta,))[1]

            # Step 2: Compute \partial_{\theta}T @ jvp_result

            _, vjp_fn = jax.vjp(flow_map, params)

            result = vjp_fn(jvp_result)[0]

            return result

        # Vectorize over all samples

        contributions = vmap(single_sample_contribution)(z_samples)

        return jax.tree.map(lambda x: jnp.mean(x, axis=0), contributions)

    # @partial(jit,static_argnums = (0,6))
    def solve_system(
        self,
        z_samples: Array,
        b: PyTree,
        params: Optional[PyTree] = None,
        tol: float = 1e-5,
        maxiter: int = 10,
        method: str = "minres",
        regularization: float = 1e-6,
        x0: Optional[PyTree] = None,
    ) -> PyTree:
        """
        Solve G(theta) x = b using conjugate gradient method

        Args:
            z_samples: (Bs,d) Samples from reference density
            b: PyTree with same GraphDef as mapping
            parms: PyTree where the G matrix is computed at
            tol: Tolerance for CG solver
            maxiter: Maximum number of iterations for CG solver
            method: Method to use for solving the linear system ("cg" or "gmres")
            regularization: Regularization parameter for the CG solver
            x0: Initial guess for the solution

        Returns:
            x: PyTree solution to G(theta)x = b
        """
        if method not in ["cg", "gmres", "minres"]:
            raise ValueError(f"Unknown method: {method}")
        if method == "cg":
            solver = lambda matvec, b, tol, maxiter, x0: reg_cg(
                matvec, b, epsilon=regularization, tol=tol, maxiter=maxiter, x0=x0
            )
        elif method == "gmres":
            solver = gmres
        elif method == "minres":
            solver = minres
        if params is None:
            _, params = nnx.split(self.mapping)
        # Define the linear operator for G(theta)
        matvec = lambda eta: self.mvp(z_samples, eta, params)
        # Use Jax inbuilts methods cg or gmres.
        t0 = time.perf_counter()
        x, backend_info = solver(matvec, b, tol=tol, maxiter=maxiter, x0=x0)
        elapsed_sec = time.perf_counter() - t0
        iterations, converged, residual_norm, iterations_estimated = self._extract_solver_stats(
            backend_info, fallback_maxiter=maxiter
        )
        info = {
            "iterations": iterations,
            "iterations_estimated": bool(iterations_estimated),
            "converged": converged,
            "residual_norm": residual_norm,
            "elapsed_sec": float(elapsed_sec),
            "backend_info": backend_info,
        }
        record = {
            "run_id": self._linear_solve_context["run_id"],
            "method": self._linear_solve_context["method_name"],
            "outer_iter": self._linear_solve_context["outer_iter"],
            "phase": self._linear_solve_context["phase"],
            "solver": method,
            "tol": float(tol),
            "maxiter": int(maxiter),
            "regularization": float(regularization),
            "iterations": iterations,
            "iterations_estimated": bool(iterations_estimated),
            "converged": converged,
            "residual_norm": residual_norm,
            "elapsed_sec": float(elapsed_sec),
            "gmvp_count": None,
        }
        self._big_solve_records.append(record)
        # x,info = minres(matvec, b, tol=tol, maxiter=maxiter,x0 = x0)
        return x, info

    def inner_product(
        self, x: PyTree, y: PyTree, z_samples: Array, params: Optional[PyTree] = None
    ) -> float:
        """
        Compute the inner product <x,y>_G = x^T G y

        Args:
            x: PyTree with same GraphDef as mapping
            y: PyTree with same GraphDef as mapping
            z_samples: (Bs,d) Samples from reference density
            parms: PyTree where the G matrix is computed at

        Returns:
            inner_product: Scalar value of the inner product
        """

        Gy = self.mvp(z_samples, y, params)

        # Compute inner product using tree utilities

        # leaves_x, treedef = jax.tree.flatten(x)

        # leaves_Gy, _ = jax.tree.flatten(Gy)

        # inner_product = sum([jnp.vdot(a, b) for a, b in zip(leaves_x, leaves_Gy)])
        inner_product = sum(
            jax.tree.leaves(jax.tree.map(lambda a, b: jnp.vdot(a, b), x, Gy))
        )
        return inner_product

    def metric_derivative_quadratic_form(
        self, z_samples: Array, eta: PyTree, params: Optional[PyTree] = None
    ) -> PyTree:
        """
        Compute the gradient of the G matrix in the direction eta. We are returning the PyTree

        [neta^T \\partial_{theta_k} G(theta) neta]_{k=1}^{N_params}

        Args:
            z_samples: (Bs,d) Samples from reference density
            eta: PyTree with same GraphDef as mapping
            parms: PyTree where the G matrix is computed at

        Returns:
            grad_G: PyTree with same GraphDef as mapping
        """

        if params is None:
            _, params = nnx.split(self.mapping)

        # Stop gradients for eta
        eta_sg = jax.lax.stop_gradient(eta)
        # Compute the gradient of the inner product <eta, G eta> w.r.t. params
        grad_fn = lambda p: self.inner_product(eta_sg, eta_sg, z_samples, p)
        grad_G = jax.grad(grad_fn)(params)

        return grad_G

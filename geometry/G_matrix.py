import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit, vmap, grad, flatten_util
from typing import Dict, Any, Optional
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

    @partial(jit, static_argnums=(0,))
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
        x, info = solver(matvec, b, tol=tol, maxiter=maxiter, x0=x0)
        # verify solution
        b_verif = self.mvp(z_samples, x, params)
        # Residual relative error
        residual = sum(
            jax.tree.leaves(
                jax.tree.map(
                    lambda a, b: jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-8),
                    b_verif,
                    b,
                )
            )
        )
        info = {"error": residual}
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

        [neta^T \partial_{theta_k} G(theta) neta]_{k=1}^{N_params}

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

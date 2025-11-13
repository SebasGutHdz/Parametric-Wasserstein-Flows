import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Optional
from jaxtyping import Array
from flax import nnx
from functools import partial


def euler_method(f: Callable, t_list: Array, y0: Array, history=False) -> Array:
    """
    Euler method for solving ODEs.
    Inputs:
        f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
        t_list: jax array, a list of time points at which to evaluate the ODE
        y0: tensor [bs,d], initial value of y at t0
    Outputs:
    """
    n_steps = len(t_list) - 1
    if history:
        y_h = [y0]
        ys = y_h[-1]
    else:
        ys = y0
    for i in range(n_steps):
        dt = t_list[i + 1] - t_list[i]
        ts = t_list[i]
        y_next = ys + dt * f(ts, ys)  # Euler step [Bs, d]
        if history:
            y_h.append(y_next)
            ys = y_h[-1]
        else:
            ys = y_next
    if history:
        return jnp.array(y_h)
    else:
        return ys


def heun_method(f: Callable, t_list: Array, y0: Array, history=False) -> Array:
    """
    Heun's method for solving ODEs.
    Inputs:
        f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
        t_list: jax array, a list of time points at which to evaluate the ODE
        y0: tensor [bs,d], initial value of y at t0
    Outputs:
        y_h: jax array, the solution of the ODE at the time points in t_list
    """
    n_steps = len(t_list) - 1
    if history:
        y_h = [y0]
        ys = y_h[-1]
    else:
        ys = y0
    for i in range(n_steps):
        dt = t_list[i + 1] - t_list[i]
        ts = t_list[i]
        y_mid = ys + dt * f(ts, ys)  # Euler step [Bs, d]
        y_next = ys + dt / 2 * (
            f(ts, ys) + f(t_list[i + 1], y_mid)
        )  # Heun step [Bs, d]
        if history:
            y_h.append(y_next)
            ys = y_h[-1]
        else:
            ys = y_next
    if history:
        return jnp.array(y_h)
    else:
        return ys


def string_2_solver(solver_str: str) -> Callable:
    """
    Converts a string to a diffrax solver.
    Inputs:
        solver_str: str, the name of the solver
    Outputs:
        solver: diffrax solver, the corresponding diffrax solver
    """
    if solver_str == "euler":
        return EulerSolver()  # euler_method
    elif solver_str == "heun":
        return HeunSolver()  # heun_method
    else:
        raise ValueError(
            f"Solver {solver_str} not recognized. Available solvers: euler, heun."
        )


class ODESolver(nnx.Module):
    """
    Base class for ODE solvers.
    """

    def __init__(self):

        pass

    def step(self):
        """Perform one integration step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the step method")

    def __call__(
        self, f: Callable, t_list: Array, y0: Array, history: bool = False
    ) -> Array:
        """
        Solve the ODE dy/dt = f(t, y) from t_list[0] to t_list[-1] with initial condition y0.
        Args:
            f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
            t_list: jax array, a list of time points at which to evaluate the ODE
            y0: tensor [bs,d], initial value of y at t0
            history: bool, if True, return the solution at all time points in t_list, else return only the final value
        Returns:
            y: jax array, the solution of the ODE at the time points in t_list if history is True, else the solution at t_list[-1]
        """
        solution_history = [y0]

        for i in range(len(t_list) - 1):
            y_new = self.step(f, t_list, i, solution_history)

            solution_history.append(y_new)
        if history:

            return jnp.array(solution_history).transpose(
                (1, 0, 2)
            )  # Shape (batch_size, time_steps, dim)
        else:
            return solution_history[-1]


class EulerSolver(ODESolver):
    """
    Euler method for solving ODEs.
    """

    def __init__(self):
        super().__init__()

    def step(
        self, f: Callable, t_list: Array, step_index: int, solution_history: list
    ) -> Array:
        """
        Perform one Euler integration step.
        Args:
            f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
            t_list: jax array, a list of time points at which to evaluate the ODE
            step_index: int, the index of the current time step
            solution_history: list of jax arrays, the history of solutions up to the current time step
        Returns:
            y_new: jax array, the solution at the next time point
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        y_current = solution_history[-1]
        y_new = y_current + dt * f(t_current, y_current)  # Euler step
        return y_new


class HeunSolver(ODESolver):
    """
    Heun's method for solving ODEs.
    """

    def __init__(self):
        super().__init__()

    def step(
        self, f: Callable, t_list: Array, step_index: int, solution_history: list
    ) -> Array:
        """
        Perform one Heun integration step.
        Args:
            f: Callable , the function defining the ODE dy/dt = f(t, y), where t is a float and y is a tensor of shape [bs, d]
            t_list: jax array, a list of time points at which to evaluate the ODE
            step_index: int, the index of the current time step
            solution_history: list of jax arrays, the history of solutions up to the current time step
        Returns:
            y_new: jax array, the solution at the next time point
        """
        dt = t_list[step_index + 1] - t_list[step_index]
        t_current = t_list[step_index]
        y_current = solution_history[-1]

        # Predictor step (Euler)
        y_predictor = y_current + dt * f(t_current, y_current)

        # Corrector step
        y_new = y_current + (dt / 2) * (
            f(t_current, y_current) + f(t_list[step_index + 1], y_predictor)
        )

        return y_new

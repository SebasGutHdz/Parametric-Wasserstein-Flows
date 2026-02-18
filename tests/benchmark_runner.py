#!/usr/bin/env python3
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import gc
import itertools
import json
import multiprocessing as mp
import queue
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.training import orbax_utils
from jax.scipy.special import logsumexp
from num2tex import num2tex
from tqdm.auto import tqdm

# TODO: proper installation
root_path = Path.cwd().parent.absolute()
import sys

sys.path.append(str(root_path))


from flows.anderson_acceleration import anderson_method
from flows.gradient_flow import run_gradient_flow
from flows.memoryless_qn import memoryless_qn_method
from functionals.functional import Potential
from functionals.functions import get_gaussian_potential, styblinski_tang_potential_fn
from functionals.internal_functional_class import InternalPotential
from functionals.linear_funcitonal_class import LinearPotential
from geometry.G_matrix import G_matrix
from parametric_model.parametric_model import ParametricModel


METHOD_ALIASES = {
    "gradient_flow": "gradient_flow",
    "gradient-flow": "gradient_flow",
    "baseline": "gradient_flow",
    "anderson": "anderson",
    "anderson_acceleration": "anderson",
    "memoryless_qn": "memoryless_qn",
    "memoryless-qn": "memoryless_qn",
    "mqn": "memoryless_qn",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as infile:
        config = json.load(infile)

    for key in ["distributions", "common_params", "methods", "plotting"]:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if not isinstance(config["methods"], dict):
        raise ValueError("config['methods'] must be a dict")

    normalized_methods = {}
    for raw_name, method_cfg in config["methods"].items():
        canonical = METHOD_ALIASES.get(raw_name, raw_name)
        if canonical not in {"gradient_flow", "anderson", "memoryless_qn"}:
            raise ValueError(f"Unknown method in config: {raw_name}")
        normalized_methods[canonical] = method_cfg or {}
    config["methods"] = normalized_methods

    plotting_cfg = config["plotting"]
    if not isinstance(plotting_cfg, dict):
        raise ValueError("config['plotting'] must be a dict")
    if "style_channels" in plotting_cfg and not isinstance(
        plotting_cfg["style_channels"], list
    ):
        raise ValueError("plotting.style_channels must be a list")
    if "style_values" in plotting_cfg and not isinstance(
        plotting_cfg["style_values"], dict
    ):
        raise ValueError("plotting.style_values must be a dict")

    legacy_plotting_keys = {"colors", "linestyles", "linewidths", "markers"}
    present_legacy = sorted(k for k in legacy_plotting_keys if k in plotting_cfg)
    if present_legacy:
        raise ValueError(
            "Legacy plotting keys are no longer supported: "
            + ", ".join(present_legacy)
            + ". Use plotting.method_colormaps/style_channels/style_values."
        )

    parallel_cfg = dict(config.get("parallel", {}))
    max_workers = int(parallel_cfg.get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("parallel.max_workers must be >= 1")

    gpu_ids_raw = parallel_cfg.get("gpu_ids", [])
    if not isinstance(gpu_ids_raw, list):
        raise ValueError("parallel.gpu_ids must be a list of non-negative integers")
    gpu_ids = [int(g) for g in gpu_ids_raw]
    if any(g < 0 for g in gpu_ids):
        raise ValueError("parallel.gpu_ids must contain only non-negative integers")

    disable_preallocate = bool(parallel_cfg.get("disable_preallocate", True))
    mem_fraction = parallel_cfg.get("mem_fraction", None)
    if mem_fraction is not None:
        mem_fraction = float(mem_fraction)
        if not (0.0 < mem_fraction <= 1.0):
            raise ValueError("parallel.mem_fraction must be in (0, 1]")

    config["parallel"] = {
        "max_workers": max_workers,
        "gpu_ids": gpu_ids,
        "disable_preallocate": disable_preallocate,
        "mem_fraction": mem_fraction,
    }
    return config


def method_grid(method_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not method_cfg:
        return [{}]

    keys = list(method_cfg.keys())
    values = [v if isinstance(v, list) else [v] for v in method_cfg.values()]
    combos = []
    for product in itertools.product(*values):
        combos.append(dict(zip(keys, product, strict=False)))
    return combos


def sanitize_component(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip().lower())


def distribution_grid(distributions_cfg: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in distributions_cfg:
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, dict) and "name" in item:
            out.append(item)
        else:
            raise ValueError(f"Invalid distribution entry: {item}")
    return out


def build_model(
    common: dict[str, Any], seed: int
) -> tuple[ParametricModel, dict[str, Any]]:
    dim = int(common["dimension"])
    n_hidden = int(common["n_hidden"])
    width_hidden = int(common["width_hidden"])
    model_cfg = {
        "parametric_map": common.get("parametric_map", "node"),
        "rhs_model": common.get("rhs_model", "mlp"),
        "activation_fn": common.get("activation_fn", "tanh"),
        "time_dependent": bool(common.get("time_dependent", True)),
        "solver": common.get("ode_solver", "euler"),
        "dt0": float(common.get("dt0", 0.01)),
        "ref_density": common.get("ref_density", "gaussian"),
        "scale_factor": float(common.get("scale_factor", 1.0)),
        "architecture": [dim, n_hidden, width_hidden],
        "seed": seed,
    }
    model = ParametricModel(
        parametric_map=model_cfg["parametric_map"],
        rhs_model=model_cfg["rhs_model"],
        architecture=model_cfg["architecture"],
        activation_fn=model_cfg["activation_fn"],
        time_dependent=model_cfg["time_dependent"],
        solver=model_cfg["solver"],
        dt0=model_cfg["dt0"],
        ref_density=model_cfg["ref_density"],
        scale_factor=model_cfg["scale_factor"],
        key=jax.random.PRNGKey(seed),
    )
    return model, model_cfg


def potential_double_banana(x: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
    x_shifted = x - shift
    x1 = x_shifted[..., 0]
    log_density = 2.0 * (jnp.linalg.norm(x_shifted, axis=-1) - 3.0) ** 2
    log_density -= logsumexp(
        jnp.stack((-2.0 * (x1 - 3.0) ** 2, -2.0 * (x1 + 3.0) ** 2), axis=-1),
        axis=-1,
    )
    return log_density


def build_problem(
    distribution_cfg: dict[str, Any], common: dict[str, Any]
) -> tuple[Potential, dict[str, Any]]:
    dist_name = distribution_cfg["name"].lower()
    dim = int(common["dimension"])

    if dist_name == "gaussian":
        mean_value = float(distribution_cfg.get("mean_value", 2.0))
        mean = jnp.full((dim,), mean_value)
        if "sigma_diag" in distribution_cfg:
            sigma_diag = jnp.asarray(distribution_cfg["sigma_diag"], dtype=jnp.float32)
            if sigma_diag.shape[0] != dim:
                raise ValueError("gaussian sigma_diag length must match dimension")
        elif "sigma_min" in distribution_cfg and "sigma_max" in distribution_cfg:
            sigma_diag = jnp.linspace(
                distribution_cfg["sigma_min"],
                distribution_cfg["sigma_max"],
                dim,
                endpoint=True,
            )
            sigma_diag = jnp.roll(sigma_diag, 1)
        else:
            raise ValueError(
                'Need to specify either the diagonal entries of the covariance matrix ("sigma_diag") or their range ("sigma_min" and "sigma_max")'
                f"Got {distribution_cfg}"
            )
        sigma_inv = jnp.diag(1.0 / sigma_diag)
        if "orth_seed" in distribution_cfg:
            rs = distribution_cfg["orth_seed"]
            key = jax.random.key(rs)
            U = jax.random.orthogonal(key, dim)
            sigma_inv = U.T @ sigma_inv @ U
        potential_fn = get_gaussian_potential(mean, sigma_inv)
    elif dist_name in {"double-banana", "double_banana", "double banana"}:
        shift_2d = distribution_cfg.get("shift", [0.0, 10.0])
        shift = jnp.zeros((dim,), dtype=jnp.float32)
        shift = shift.at[0].set(float(shift_2d[0]))
        if dim >= 2:
            shift = shift.at[1].set(float(shift_2d[1]))

        def potential_fn(x: jnp.ndarray) -> jnp.ndarray:
            return potential_double_banana(x, shift)

    elif dist_name in {"styblinski-tang", "styblinski_tang", "st"}:

        def potential_fn(x: jnp.ndarray) -> jnp.ndarray:
            return styblinski_tang_potential_fn(x, d=dim)

    else:
        raise ValueError(f"Unknown distribution: {distribution_cfg['name']}")

    linear_potential = LinearPotential(potential_fn=potential_fn, coeff=1.0)
    internal_potential = InternalPotential(
        functional="entropy", coeff=1.0, method="exact", prob_dim=dim
    )
    potential = Potential(
        linear=linear_potential, internal=internal_potential, interaction=None
    )
    return potential, {"distribution": distribution_cfg}


def build_plot_potential_2d(distribution_cfg: dict[str, Any]) -> LinearPotential | None:
    name = distribution_cfg["name"].lower()
    if name == "gaussian":
        mean_value = float(distribution_cfg.get("mean_value", 2.0))
        mean = jnp.array([mean_value, mean_value], dtype=jnp.float32)
        sigma_first = float(distribution_cfg.get("sigma_first", 1000.0))
        sigma_second = float(distribution_cfg.get("sigma_second", 10.0))
        sigma_inv = jnp.diag(jnp.array([sigma_first, sigma_second], dtype=jnp.float32))
        fn = get_gaussian_potential(mean, sigma_inv)
        return LinearPotential(potential_fn=fn, coeff=1.0)

    if name in {"double-banana", "double_banana", "double banana"}:
        shift_2d = distribution_cfg.get("shift", [0.0, 10.0])
        shift = jnp.array([float(shift_2d[0]), float(shift_2d[1])], dtype=jnp.float32)

        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return potential_double_banana(x, shift)

        return LinearPotential(potential_fn=fn, coeff=1.0)

    if name in {"styblinski-tang", "styblinski_tang", "st"}:

        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return styblinski_tang_potential_fn(x, d=2)

        return LinearPotential(potential_fn=fn, coeff=1.0)

    return None


def sample_reference(
    common: dict[str, Any], seed: int, n_samples: int | None = None
) -> jnp.ndarray:
    dim = int(common["dimension"])
    n = int(n_samples if n_samples is not None else common.get("plot_n_samples", 300))
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n, dim))


def save_model_checkpoint(final_model: ParametricModel, ckpt_dir: Path) -> None:
    import shutil

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)

    state = nnx.state(final_model)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    abspath = ckpt_dir.absolute()
    checkpointer.save(abspath, state, save_args=save_args)


def run_single(
    method: str,
    method_params: dict[str, Any],
    distribution_cfg: dict[str, Any],
    common: dict[str, Any],
    run_seed: int,
    checkpoint_root: Path,
    run_id: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    diagnostic_sample_size: int | None = None,
) -> dict[str, Any]:
    model, model_cfg = build_model(common, run_seed)
    potential, potential_meta = build_problem(distribution_cfg, common)
    g_mat = G_matrix(model)

    n_samples = int(common["N_samples"])
    max_iterations = int(common.get("max_iterations", 300))
    stepsize = float(common["stepsize"])
    tolerance = float(common.get("tolerance", 1e-4))
    solver = common.get("linear_solver", "cg")
    z_samples = sample_reference(
        common, run_seed + 13, n_samples=int(common.get("eval_samples", 300))
    )

    t0 = time.perf_counter()

    if method == "gradient_flow":
        history = run_gradient_flow(
            model,
            z_samples,
            g_mat,
            potential,
            N_samples=n_samples,
            h=float(method_params.get("stepsize", stepsize)),
            solver=str(method_params.get("solver", solver)),
            max_iterations=int(method_params.get("max_iterations", max_iterations)),
            tolerance=float(method_params.get("tolerance", tolerance)),
            regularization=float(method_params.get("regularization", 1e-6)),
            progress_every=int(
                method_params.get("progress_every", common.get("progress_every", 100))
            ),
            plot_intermediate=bool(method_params.get("plot_intermediate", False)),
            verbose=bool(method_params.get("verbose", False)),
            use_tqdm=bool(method_params.get("use_tqdm", False)),
            progress_callback=progress_callback,
            diagnostic_sample_size=diagnostic_sample_size,
        )
        final_model = history["final_parametric_model"]
        energies = np.asarray(history["energy_history"], dtype=np.float64)
        riem_grad = np.asarray(history["riemann_grad_norm_history"], dtype=np.float64)

    elif method == "anderson":
        graphdef, init_params = nnx.split(model)
        final_params, history = anderson_method(
            parametric_model=model,
            batch_size=n_samples,
            test_data_set=z_samples,
            G_mat=g_mat,
            potential=potential,
            initial_params=init_params,
            n_iterations=int(method_params.get("max_iterations", max_iterations)),
            step_size=float(method_params.get("stepsize", stepsize)),
            memory_size=int(method_params.get("memory_size", 8)),
            relaxation=float(method_params.get("relaxation", 1.8)),
            anderson_tol=float(method_params.get("anderson_tol", 1e-6)),
            solver=str(method_params.get("solver", solver)),
            solver_tol=float(method_params.get("solver_tol", tolerance)),
            solver_maxiter=int(method_params.get("solver_maxiter", 50)),
            regularization=float(method_params.get("regularization", 1e-6)),
            convergence_tol=float(method_params.get("tolerance", tolerance)),
            plot_intermediate=False,
            plot_frequency=int(
                method_params.get("progress_every", common.get("progress_every", 100))
            ),
            save_param_trajectory=False,
            regularization_factor_gamma=float(
                method_params.get("regularization", 1e-3)
            ),
            regularization_method_gamma=str(
                method_params.get("regularization_kind", "l2")
            ),
            ensure_descent=bool(method_params.get("ensure_descent", True)),
            verbose=bool(method_params.get("verbose", False)),
            progress_callback=progress_callback,
            diagnostic_sample_size=diagnostic_sample_size,
        )
        final_model = nnx.merge(graphdef, final_params)
        energies = np.asarray(history["energies"], dtype=np.float64)
        riem_grad = np.asarray(history["riemann_grad_history"], dtype=np.float64)

    elif method == "memoryless_qn":
        graphdef, init_params = nnx.split(model)
        final_params, history = memoryless_qn_method(
            parametric_model=model,
            batch_size=n_samples,
            test_data_set=z_samples,
            G_mat=g_mat,
            potential=potential,
            initial_params=init_params,
            n_iterations=int(method_params.get("max_iterations", max_iterations)),
            step_size=float(method_params.get("stepsize", stepsize)),
            solver=str(method_params.get("solver", "cg")),
            solver_tol=float(method_params.get("solver_tol", tolerance)),
            solver_maxiter=int(method_params.get("solver_maxiter", 50)),
            solver_regularization=float(
                method_params.get("solver_regularization", 1e-6)
            ),
            ensure_descent=bool(method_params.get("ensure_descent", False)),
            hessian_update_strategy=str(
                method_params.get("hessian_update_strategy", "BFGS")
            ),
            regularization_strategy=str(
                method_params.get("regularization_strategy", "Li-Fukushima")
            ),
            spectral_scaling=bool(method_params.get("spectral_scaling", True)),
            convergence_tol=float(method_params.get("tolerance", tolerance)),
            plot_intermediate=False,
            plot_frequency=int(
                method_params.get("progress_every", common.get("progress_every", 100))
            ),
            save_param_trajectory=False,
            verbose=bool(method_params.get("verbose", False)),
            progress_callback=progress_callback,
            diagnostic_sample_size=diagnostic_sample_size,
        )
        final_model = nnx.merge(graphdef, final_params)
        energies = np.asarray(history["energies"], dtype=np.float64)
        riem_grad = np.asarray(history["riemann_grad_history"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported method: {method}")

    runtime_sec = time.perf_counter() - t0

    ckpt_relpath = Path("model_checkpoints") / run_id
    save_model_checkpoint(final_model, checkpoint_root / ckpt_relpath)

    return {
        "method": method,
        "method_params": method_params,
        "distribution": distribution_cfg,
        "common": common,
        "model_config": model_cfg,
        "problem_meta": potential_meta,
        "energy_history": energies,
        "riemann_grad_history": riem_grad,
        "runtime_sec": runtime_sec,
        "model_ckpt_relpath": str(ckpt_relpath),
    }


def initialize_h5(path: Path, config: dict[str, Any]) -> None:
    filtered_config = {
        key: value
        for key, value in config.items()
        if key not in {"plotting", "parallel"}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs["config_json"] = json.dumps(filtered_config)
        h5.attrs["run_count"] = 0
        h5.create_group("runs")


def append_run_to_h5(path: Path, run_id: str, run: dict[str, Any]) -> None:
    with h5py.File(path, "a") as h5:
        runs_grp = h5["runs"]
        if run_id in runs_grp:
            del runs_grp[run_id]
        grp = runs_grp.create_group(run_id)
        grp.attrs["method"] = run["method"]
        grp.attrs["distribution"] = run["distribution"]["name"]
        grp.attrs["method_params_json"] = json.dumps(
            run["method_params"], sort_keys=True
        )
        grp.attrs["model_config_json"] = json.dumps(run["model_config"], sort_keys=True)
        grp.attrs["common_json"] = json.dumps(run["common"], sort_keys=True)
        grp.attrs["runtime_sec"] = float(run["runtime_sec"])
        grp.attrs["model_ckpt_relpath"] = run["model_ckpt_relpath"]
        grp.create_dataset("energy_history", data=run["energy_history"])
        grp.create_dataset("riemann_grad_history", data=run["riemann_grad_history"])
        h5.attrs["run_count"] = int(h5.attrs.get("run_count", 0)) + 1
        h5.flush()


def load_experiment_config_from_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        raw = h5.attrs.get("config_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    loaded = json.loads(str(raw))
    if not isinstance(loaded, dict):
        raise ValueError("Invalid config_json in .h5: expected dict")
    return {
        key: value
        for key, value in loaded.items()
        if key not in {"plotting", "parallel"}
    }


def make_plot_config(
    current_config: dict[str, Any], experiment_config_h5: dict[str, Any]
) -> dict[str, Any]:
    config_for_plot = dict(experiment_config_h5)
    config_for_plot["plotting"] = dict(current_config.get("plotting", {}))
    if "methods" not in config_for_plot:
        config_for_plot["methods"] = dict(current_config.get("methods", {}))
    if "common_params" not in config_for_plot:
        config_for_plot["common_params"] = dict(current_config.get("common_params", {}))
    if "distributions" not in config_for_plot:
        config_for_plot["distributions"] = []
    return config_for_plot


def load_runs_from_h5(path: Path) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as h5:
        out: list[dict[str, Any]] = []
        for run_id in sorted(h5["runs"].keys()):
            grp = h5["runs"][run_id]
            out.append(
                {
                    "run_id": run_id,
                    "method": str(grp.attrs["method"]),
                    "distribution": str(grp.attrs["distribution"]),
                    "method_params": json.loads(str(grp.attrs["method_params_json"])),
                    "model_config": json.loads(str(grp.attrs["model_config_json"])),
                    "common": json.loads(str(grp.attrs["common_json"])),
                    "runtime_sec": float(grp.attrs["runtime_sec"]),
                    "model_ckpt_relpath": str(grp.attrs["model_ckpt_relpath"]),
                    "energy_history": np.asarray(
                        grp["energy_history"][:], dtype=np.float64
                    ),
                    "riemann_grad_history": np.asarray(
                        grp["riemann_grad_history"][:], dtype=np.float64
                    ),
                }
            )
    return out


KEY_LABELS = {
    "relaxation": r"\beta",
    "regularization": r"\lambda",
    "stepsize": r"h",
    "hessian_update_strategy": r"\mathrm{HS}",
    "regularization_kind": r"\mathrm{RK}",
    "regularization_strategy": r"\mathrm{RS}",
    "spectral_scaling": r"\mathrm{SS}",
    "memory_size": r"\mathrm{M}",
    "ensure_descent": r"\mathrm{ED}",
    "solver_maxiter": r"\mathrm{SMI}",
    "solver_tol": r"\mathrm{ST}",
    "anderson_tol": r"\mathrm{AT}",
    "max_iterations": r"\mathrm{MI}",
}

VALUE_LABELS = {
    "Li-Fukushima": r"\mathrm{LiF}",
    "Powell": r"\mathrm{Pwl}",
    "preconvex": r"\mathrm{PC}",
    "adaptive": r"\mathrm{adp}",
}


def get_varying_keys_in_order(
    method_runs: list[dict[str, Any]],
    key_order: list[str] | None = None,
) -> list[str]:
    if not method_runs:
        return []

    ordered_keys: list[str] = []
    seen: set[str] = set()
    if key_order is not None:
        for key in key_order:
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)
    for run in method_runs:
        for key in run["method_params"].keys():
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

    varying: list[str] = []
    for key in ordered_keys:
        values = [run["method_params"].get(key, None) for run in method_runs]
        if len(set(values)) > 1:
            varying.append(key)
    return varying


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _value_order_for_param(
    method_cfg: dict[str, Any],
    key: str,
    method_runs: list[dict[str, Any]],
) -> list[Any]:
    ordered_values: list[Any] = []
    if key in method_cfg:
        cfg_values = method_cfg[key]
        cfg_list = cfg_values if isinstance(cfg_values, list) else [cfg_values]
        for value in cfg_list:
            _append_unique(ordered_values, value)
    for run in method_runs:
        if key in run["method_params"]:
            _append_unique(ordered_values, run["method_params"][key])
    return ordered_values


def _method_param_order(
    method_cfg: dict[str, Any], method_runs: list[dict[str, Any]]
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for key in method_cfg.keys():
        if key not in seen:
            order.append(key)
            seen.add(key)
    for run in method_runs:
        for key in run["method_params"].keys():
            if key not in seen:
                order.append(key)
                seen.add(key)
    return order


def _sample_colormap(method: str, cmap_name: str, count: int) -> list[Any]:
    if count <= 0:
        return []
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        tqdm.write(
            f"[warn] unknown colormap '{cmap_name}' for {method}; using 'viridis'"
        )
        cmap = plt.get_cmap("viridis")

    if count == 1:
        return [cmap(0.6)]
    return [cmap(t) for t in np.linspace(0.15, 0.9, count)]


DEFAULT_STYLE_CHANNELS = ["color", "linestyle", "linewidth", "marker"]

DEFAULT_STYLE_VALUES = {
    "linestyle": ["-", "--", ":", "-."],
    "linewidth": [2.2, 1.8, 1.4, 1.1],
    "marker": ["o", "^", "s", "D", "x", "P", "*", "v"],
    "alpha": [0.9, 0.75, 0.6, 0.45],
}


def _values_for_style_channel(plotting_cfg: dict[str, Any], channel: str) -> list[Any]:
    style_values = plotting_cfg.get("style_values", {})
    if channel in style_values:
        values = style_values[channel]
        if not isinstance(values, list):
            values = [values]
        if not values:
            raise ValueError(f"plotting.style_values['{channel}'] must not be empty")
        return values

    if channel in DEFAULT_STYLE_VALUES:
        return DEFAULT_STYLE_VALUES[channel]

    raise ValueError(
        f"No default values for style channel '{channel}'. "
        f"Add plotting.style_values['{channel}']."
    )


def build_method_style_plan(
    config: dict[str, Any],
    plotting_cfg: dict[str, Any],
    method: str,
    method_runs: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    methods_cfg = config.get("methods", {})
    method_cfg_raw = (
        methods_cfg.get(method, {}) if isinstance(methods_cfg, dict) else {}
    )
    method_cfg = method_cfg_raw if isinstance(method_cfg_raw, dict) else {}

    param_order = _method_param_order(method_cfg, method_runs)
    varying_keys = get_varying_keys_in_order(method_runs, param_order)

    channels = plotting_cfg.get("style_channels", DEFAULT_STYLE_CHANNELS)
    if not isinstance(channels, list):
        raise ValueError("plotting.style_channels must be a list")
    if not channels:
        raise ValueError("plotting.style_channels must not be empty")

    channel_value_maps: dict[str, dict[str, Any]] = {}
    method_colormaps = plotting_cfg.get("method_colormaps", {})
    cmap_name = str(method_colormaps.get(method, "viridis"))

    for channel_idx, channel in enumerate(channels):
        param_key = (
            varying_keys[channel_idx] if channel_idx < len(varying_keys) else None
        )

        if channel == "color":
            if param_key is None:
                colors = _sample_colormap(method, cmap_name, 1)
                channel_value_maps[channel] = {
                    "param_key": None,
                    "value_map": {},
                    "default": colors[0],
                }
                continue

            value_order = _value_order_for_param(method_cfg, param_key, method_runs)
            if len(value_order) > 1:
                colors = _sample_colormap(method, cmap_name, len(value_order))
                channel_value_maps[channel] = {
                    "param_key": param_key,
                    "value_map": {
                        value: colors[i] for i, value in enumerate(value_order)
                    },
                    "default": colors[0],
                }
            else:
                colors = _sample_colormap(method, cmap_name, 1)
                channel_value_maps[channel] = {
                    "param_key": param_key,
                    "value_map": {},
                    "default": colors[0],
                }
            continue

        channel_values = _values_for_style_channel(plotting_cfg, channel)
        value_map: dict[Any, Any] = {}
        if param_key is not None:
            value_order = _value_order_for_param(method_cfg, param_key, method_runs)
            if len(value_order) > 1:
                value_map = {
                    value: channel_values[i % len(channel_values)]
                    for i, value in enumerate(value_order)
                }

        channel_value_maps[channel] = {
            "param_key": param_key,
            "value_map": value_map,
            "default": channel_values[0],
        }

    return varying_keys, channel_value_maps


def latex_key(key: str) -> str:
    if key in KEY_LABELS:
        return KEY_LABELS[key]
    return rf"\mathrm{{{key.replace('_', r'\_')}}}"


def latex_value(value: Any) -> str:
    if isinstance(value, bool):
        return r"\mathrm{T}" if value else r"\mathrm{F}"
    if isinstance(value, (int, float)):
        return num2tex(value)
    if isinstance(value, str) and value in VALUE_LABELS:
        return VALUE_LABELS[value]
    return rf"\mathrm{{{str(value).replace('_', r'\_')}}}"


def build_method_label_latex(
    method: str,
    params: dict[str, Any],
    varying_keys: list[str],
) -> str:
    method_tex = rf"\mathrm{{{method.replace('_', r'\_')}}}"
    if not varying_keys:
        return rf"${method_tex}$"
    parts = [
        f"{latex_key(k)}={latex_value(params[k])}" for k in varying_keys if k in params
    ]
    return rf"${method_tex}\;|\;" + r",\;".join(parts) + "$"


def style_for_run(
    method_params: dict[str, Any],
    channel_value_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    style: dict[str, Any] = {}
    for channel, spec in channel_value_maps.items():
        param_key = spec["param_key"]
        value_map = spec["value_map"]
        default_value = spec["default"]
        if param_key is None:
            style[channel] = default_value
            continue
        param_value = method_params.get(param_key, None)
        style[channel] = value_map.get(param_value, default_value)
    return style


SCATTER_STYLE_KWARGS = {
    "alpha",
    "c",
    "cmap",
    "color",
    "edgecolors",
    "facecolors",
    "linestyle",
    "linestyles",
    "linewidth",
    "linewidths",
    "marker",
    "norm",
    "plotnonfinite",
    "rasterized",
    "vmax",
    "vmin",
    "zorder",
}


def filter_scatter_style_kwargs(
    style_kwargs: dict[str, Any],
    method: str,
    warned: set[tuple[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in style_kwargs.items():
        scatter_key = "linewidths" if key == "linewidth" else key
        if scatter_key in SCATTER_STYLE_KWARGS:
            out[scatter_key] = value
            continue
        warning_key = (method, key)
        if warning_key not in warned:
            tqdm.write(
                f"[warn] style kwarg '{key}' for method '{method}' is not "
                "supported by scatter; ignoring"
            )
            warned.add(warning_key)
    return out


def dynamic_legend_layout(labels: list[str], fig_width: float) -> tuple[int, float]:
    n = max(1, len(labels))
    max_chars = max((len(lbl) for lbl in labels), default=20)
    ncol = int(n**0.5)

    # Keep total legend width inside figure width.
    # TODO: 0.035 char width found empirically. find a right way to compute it
    est_col_width = 0.035 * (max_chars + 1)
    ncol = min(int(0.95 * fig_width / est_col_width), ncol)

    nrows = int(np.ceil(n / ncol))
    bottom = float(np.clip(0.06 + 0.05 * nrows, 0.08, 0.35))
    return ncol, bottom


def save_live_convergence_plot(
    energy_history: list[float],
    riemann_grad_history: list[float],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if energy_history:
        e = np.asarray(energy_history, dtype=np.float64)
        axes[0].plot(e, color="#1f77b4", linewidth=1.8)
        if np.all(e > 0):
            axes[0].set_yscale("log")
    if riemann_grad_history:
        g = np.asarray(riemann_grad_history, dtype=np.float64)
        axes[1].plot(g, color="#d62728", linewidth=1.8)
        if np.all(g > 0):
            axes[1].set_yscale("log")

    axes[0].set_title("Energy")
    axes[1].set_title("Riemannian gradient norm")
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("iteration")
    axes[0].set_ylabel("energy")
    axes[1].set_ylabel("grad norm")
    axes[0].grid(True)
    axes[1].grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_live_scatter_plot(
    scatter_samples: np.ndarray,
    distribution_cfg: dict[str, Any],
    plotting_cfg: dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(plotting_cfg.get("scatter_figsize", [8, 8])))

    x = scatter_samples[:, 0]
    y = scatter_samples[:, 1]
    ax.scatter(
        x,
        y,
        s=float(plotting_cfg.get("scatter_size", 20)),
        alpha=float(plotting_cfg.get("method_alpha", 0.55)),
        color=plotting_cfg.get("diagnostic_color", "#444444"),
    )

    try:
        plot_pot = build_plot_potential_2d(distribution_cfg)
        if plot_pot is not None:
            low = np.min(scatter_samples[:, :2], axis=0)
            high = np.max(scatter_samples[:, :2], axis=0)
            margin = 0.2 * np.maximum(high - low, 1e-3)
            x_bds = jnp.array([low[0] - margin[0], high[0] + margin[0]])
            y_bds = jnp.array([low[1] - margin[1], high[1] + margin[1]])
            plot_pot.plot_function(
                fig=fig,
                ax=ax,
                x_bds=x_bds,
                y_bds=y_bds,
                fill=False,
                levels=20,
                alpha=0.5,
            )
    except Exception:
        pass

    ax.set_title("Samples")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_convergence_plots(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plotting_cfg : dict = config.get("plotting", {})
    distributions = sorted({run["distribution"] for run in runs})
    guess_min = plotting_cfg.get('guess_min', False)
    for distribution in distributions:
        dist_runs = [run for run in runs if run["distribution"] == distribution]
        method_runs_map: dict[str, list[dict[str, Any]]] = {}
        for run in dist_runs:
            method_runs_map.setdefault(run["method"], []).append(run)
        style_plan_map: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {
            method: build_method_style_plan(config, plotting_cfg, method, method_runs)
            for method, method_runs in method_runs_map.items()
        }

        fig, axes = plt.subplots(
            1, 2, figsize=tuple(plotting_cfg.get("figsize", [16, 6]))
        )

        if guess_min:
            e_min = min(jnp.min(run["energy_history"]) for run in dist_runs) - 1e-6
        else:
            e_min = 0.

        used_labels: dict[str, int] = {}
        for run in dist_runs:
            m = run["method"]
            varying_keys, channel_value_maps = style_plan_map[m]
            style = style_for_run(run["method_params"], channel_value_maps)
            label = build_method_label_latex(m, run["method_params"], varying_keys)
            if label in used_labels:
                used_labels[label] += 1
                label = label[:-1] + rf"\;\mathrm{{(run\ {used_labels[label]})}}$"
            else:
                used_labels[label] = 1
            axes[0].plot(
                run["energy_history"] - e_min,
                label=label,
                **style,
            )
            axes[1].plot(
                run["riemann_grad_history"],
                label=label,
                **style,
            )

        if guess_min:
            axes[0].set_yscale('log')
        axes[1].set_yscale('log')

        axes[0].set_title(f"Energy history ({distribution})")
        axes[1].set_title(f"Riemannian gradient history ({distribution})")
        axes[0].set_xlabel("iteration")
        axes[1].set_xlabel("iteration")
        axes[0].set_ylabel("energy")
        axes[1].set_ylabel("riemann grad norm")
        axes[0].grid(True)
        axes[1].grid(True)
        handles, labels = axes[1].get_legend_handles_labels()
        ncol, bottom = dynamic_legend_layout(labels, fig.get_size_inches()[0])
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=True,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=bottom)
        dist_slug = sanitize_component(distribution)
        out = output_dir / f"run_all__{dist_slug}__all_methods__convergence.pdf"
        fig.savefig(out)
        plt.close(fig)


def restore_model_from_run(
    run: dict[str, Any], checkpoint_root: Path
) -> ParametricModel:
    seed = int(run["model_config"].get("seed", 0))
    model = ParametricModel(
        parametric_map=run["model_config"]["parametric_map"],
        rhs_model=run["model_config"]["rhs_model"],
        architecture=run["model_config"]["architecture"],
        activation_fn=run["model_config"]["activation_fn"],
        time_dependent=run["model_config"]["time_dependent"],
        solver=run["model_config"]["solver"],
        dt0=run["model_config"]["dt0"],
        ref_density=run["model_config"]["ref_density"],
        scale_factor=run["model_config"]["scale_factor"],
        key=jax.random.PRNGKey(seed),
    )
    template_state = nnx.state(model)
    ckpt_path = checkpoint_root / run["model_ckpt_relpath"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for run {run.get('run_id', '<unknown>')}: {ckpt_path}"
        )
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(template_state)
    restored_state = checkpointer.restore(
        str(ckpt_path.absolute()),
        item=template_state,
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)
    return model


def generate_samples(
    model: ParametricModel, dim: int, n_samples: int, seed: int
) -> np.ndarray:
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (n_samples, dim))
    x = model(z)
    return np.asarray(x)


def get_dist_cfg(config: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in distribution_grid(config["distributions"]):
        if entry["name"] == name:
            return entry
    raise ValueError(f"Distribution config not found for {name}")


def save_scatter_plots(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    checkpoint_root: Path,
    output_dir: Path,
) -> None:
    plotting_cfg = config.get("plotting", {})
    common = config["common_params"]
    dim = int(common["dimension"])
    n_samples = int(common.get("plot_n_samples", 300))
    distributions = sorted({run["distribution"] for run in runs})

    for distribution in distributions:
        dist_runs = [run for run in runs if run["distribution"] == distribution]
        method_runs_map: dict[str, list[dict[str, Any]]] = {}
        for run in dist_runs:
            method_runs_map.setdefault(run["method"], []).append(run)
        style_plan_map: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {
            method: build_method_style_plan(config, plotting_cfg, method, method_runs)
            for method, method_runs in method_runs_map.items()
        }
        warned_scatter_kwargs: set[tuple[str, str]] = set()

        gf_runs = [run for run in dist_runs if run["method"] == "gradient_flow"]
        if not gf_runs:
            print(f"[warn] skip scatter for {distribution}: no gradient_flow run")
            continue
        gf_baseline = gf_runs[0]
        gf_varying_keys, gf_channel_maps = style_plan_map["gradient_flow"]
        gf_style = style_for_run(gf_baseline["method_params"], gf_channel_maps)
        gf_scatter_style = filter_scatter_style_kwargs(
            gf_style,
            "gradient_flow",
            warned_scatter_kwargs,
        )
        gf_model = restore_model_from_run(gf_baseline, checkpoint_root)
        gf_samples = generate_samples(gf_model, dim, n_samples, seed=123)

        methods = sorted(
            {run["method"] for run in dist_runs if run["method"] != "gradient_flow"}
        )
        for method in methods:
            method_runs = [run for run in dist_runs if run["method"] == method]
            if not method_runs:
                continue

            fig, ax = plt.subplots(
                figsize=tuple(plotting_cfg.get("scatter_figsize", [8, 8]))
            )
            method_varying_keys, method_channel_maps = style_plan_map[method]
            used_labels: dict[str, int] = {}

            gf_label = build_method_label_latex(
                "gradient_flow", gf_baseline["method_params"], gf_varying_keys
            )
            ax.scatter(
                gf_samples[:, 0],
                gf_samples[:, 1],
                s=float(plotting_cfg.get("scatter_size", 20)),
                alpha=float(plotting_cfg.get("gf_alpha", 0.5)),
                label=gf_label,
                **gf_scatter_style,
            )
            used_labels[gf_label] = 1

            all_method_samples = []
            for idx, run in enumerate(method_runs):
                style = style_for_run(run["method_params"], method_channel_maps)
                scatter_style = filter_scatter_style_kwargs(
                    style,
                    method,
                    warned_scatter_kwargs,
                )
                model = restore_model_from_run(run, checkpoint_root)
                samples = generate_samples(model, dim, n_samples, seed=1000 + idx)
                all_method_samples.append(samples)
                label = build_method_label_latex(
                    method, run["method_params"], method_varying_keys
                )
                if label in used_labels:
                    used_labels[label] += 1
                    label = label[:-1] + rf"\;\mathrm{{(run\ {used_labels[label]})}}$"
                else:
                    used_labels[label] = 1
                ax.scatter(
                    samples[:, 0],
                    samples[:, 1],
                    s=float(plotting_cfg.get("scatter_size", 20)),
                    alpha=float(plotting_cfg.get("method_alpha", 0.5)),
                    label=label,
                    **scatter_style,
                )

            try:
                dist_cfg = get_dist_cfg(config, distribution)
                plot_pot = build_plot_potential_2d(dist_cfg)
                if plot_pot is not None:
                    joint = np.concatenate(
                        [gf_samples[:, :2]] + [s[:, :2] for s in all_method_samples],
                        axis=0,
                    )
                    low = np.min(joint, axis=0)
                    high = np.max(joint, axis=0)
                    margin = 0.2 * np.maximum(high - low, 1e-3)
                    x_bds = jnp.array([low[0] - margin[0], high[0] + margin[0]])
                    y_bds = jnp.array([low[1] - margin[1], high[1] + margin[1]])
                    plot_pot.plot_function(
                        fig=fig,
                        ax=ax,
                        x_bds=x_bds,
                        y_bds=y_bds,
                        fill=False,
                        levels=20,
                        alpha=0.5,
                    )
            except Exception as exc:
                print(f"[warn] failed contour plot for {distribution}/{method}: {exc}")

            ax.set_title(f"{distribution}: gradient_flow vs {method}")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            ncol, bottom = dynamic_legend_layout(labels, fig.get_size_inches()[0])
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=ncol,
                frameon=True,
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=bottom)
            dist_slug = sanitize_component(distribution)
            method_slug = sanitize_component(method)
            out = output_dir / f"run_all__{dist_slug}__{method_slug}__scatter.pdf"
            fig.savefig(out)
            plt.close(fig)


def create_benchmark_session_dir(output_root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / ts
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def latest_h5(output_root: Path) -> Path | None:
    files = sorted(output_root.glob("**/results.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _get_live_plot_every(common: dict[str, Any]) -> int | None:
    if "live_plot_every" not in common:
        return None
    return max(1, int(common["live_plot_every"]))


def _prepare_planned_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    common = config["common_params"]
    dist_cfgs = distribution_grid(config["distributions"])
    base_seed = int(common.get("seed", 0))

    varying_keys_lookup: dict[tuple[str, str], list[str]] = {}
    for dist in dist_cfgs:
        for method, m_cfg in config["methods"].items():
            candidate_runs: list[dict[str, Any]] = []
            for params in method_grid(m_cfg):
                p = dict(params)
                p.setdefault("stepsize", common["stepsize"])
                p.setdefault("max_iterations", common.get("max_iterations", 300))
                p.setdefault("tolerance", common.get("tolerance", 1e-4))
                candidate_runs.append({"method_params": p})
            varying_keys_lookup[(dist["name"], method)] = get_varying_keys_in_order(
                candidate_runs, list(m_cfg.keys())
            )

    planned_runs: list[dict[str, Any]] = []
    run_counter = 0
    for dist in dist_cfgs:
        for method, m_cfg in config["methods"].items():
            for params in method_grid(m_cfg):
                p = dict(params)
                p.setdefault("stepsize", common["stepsize"])
                p.setdefault("max_iterations", common.get("max_iterations", 300))
                p.setdefault("tolerance", common.get("tolerance", 1e-4))
                dist_slug = sanitize_component(dist["name"])
                method_slug = sanitize_component(method)
                run_id = f"{dist_slug}__{method_slug}__run_{run_counter:04d}"
                run_name = f"{dist['name']} | {method} | {run_id}"
                run_method_label = build_method_label_latex(
                    method,
                    p,
                    varying_keys_lookup[(dist["name"], method)],
                )
                planned_runs.append(
                    {
                        "run_index": run_counter,
                        "run_id": run_id,
                        "run_name": run_name,
                        "run_title_base": f"{dist['name']} ",
                        "run_method_label": run_method_label,
                        "inner_total": int(
                            p.get("max_iterations", common.get("max_iterations", 300))
                        ),
                        "method": method,
                        "params": p,
                        "distribution": dist,
                        "run_seed": base_seed + run_counter,
                    }
                )
                run_counter += 1
    return planned_runs


def _execute_planned_run(
    planned_run: dict[str, Any],
    common: dict[str, Any],
    plotting_cfg: dict[str, Any],
    benchmark_dir: Path,
    live_plot_every: int | None,
    plot_n_samples: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_sink: Callable[[dict[str, Any]], None] | None = None,
    print_warnings: bool = True,
) -> tuple[dict[str, Any], int]:
    run_id = str(planned_run["run_id"])
    method = str(planned_run["method"])
    params = dict(planned_run["params"])
    dist = dict(planned_run["distribution"])

    diagnostic_root = benchmark_dir / "diagnostic_plots"
    live_convergence_path = diagnostic_root / f"run_{run_id}__convergence.pdf"
    live_scatter_path = diagnostic_root / f"run_{run_id}__scatter.pdf"

    live_energy: list[float] = []
    live_grad: list[float] = []
    latest_scatter_samples: np.ndarray | None = None
    warned_1d_scatter = False

    def on_progress(info: dict[str, Any]) -> None:
        nonlocal latest_scatter_samples, warned_1d_scatter

        iteration = int(info.get("iteration", 0)) + 1
        energy = float(info.get("energy", np.nan))
        grad = float(info.get("riemann_grad_norm", np.nan))

        if np.isfinite(energy):
            live_energy.append(energy)
        if np.isfinite(grad):
            live_grad.append(grad)

        scatter_samples = info.get("scatter_samples", None)
        if scatter_samples is not None:
            latest_scatter_samples = np.asarray(scatter_samples)

        if progress_callback is not None:
            progress_callback(info)

        if progress_sink is not None:
            progress_sink(
                {
                    "iteration": iteration,
                    "energy": energy,
                    "grad": grad,
                }
            )

        if live_plot_every is None or iteration % live_plot_every != 0:
            return

        diag_title = (
            f"{planned_run['run_title_base']} | iter={iteration}"
            + "\n"
            + planned_run["run_method_label"]
        )
        save_live_convergence_plot(
            live_energy,
            live_grad,
            live_convergence_path,
            title=diag_title,
        )
        if int(common["dimension"]) >= 2:
            if latest_scatter_samples is not None:
                save_live_scatter_plot(
                    latest_scatter_samples,
                    dist,
                    plotting_cfg,
                    live_scatter_path,
                    title=diag_title,
                )
        elif not warned_1d_scatter:
            if print_warnings:
                tqdm.write(
                    f"[warn] skip diagnostic scatter for {run_id}: dimension is 1"
                )
            warned_1d_scatter = True

    run = run_single(
        method=method,
        method_params=params,
        distribution_cfg=dist,
        common=common,
        run_seed=int(planned_run["run_seed"]),
        checkpoint_root=benchmark_dir,
        run_id=run_id,
        progress_callback=on_progress,
        diagnostic_sample_size=plot_n_samples,
    )

    final_iter = max(1, len(run["energy_history"]))
    if live_plot_every is not None:
        final_title = (
            f"{planned_run['run_title_base']} | iter={final_iter}"
            + "\n"
            + planned_run["run_method_label"]
        )
        save_live_convergence_plot(
            list(np.asarray(run["energy_history"], dtype=np.float64)),
            list(np.asarray(run["riemann_grad_history"], dtype=np.float64)),
            live_convergence_path,
            title=final_title,
        )
        if int(common["dimension"]) >= 2 and latest_scatter_samples is not None:
            save_live_scatter_plot(
                latest_scatter_samples,
                dist,
                plotting_cfg,
                live_scatter_path,
                title=final_title,
            )

    return run, final_iter


def _parallel_worker_loop(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    benchmark_dir: str,
    common: dict[str, Any],
    plotting_cfg: dict[str, Any],
    live_plot_every: int | None,
    plot_n_samples: int,
) -> None:
    def _probe_device() -> tuple[bool, str]:
        try:
            probe = jnp.ones((1,), dtype=jnp.float32)
            device_obj = None
            if hasattr(probe, "device"):
                device_attr = probe.device
                if callable(device_attr):
                    device_obj = device_attr()
                else:
                    device_obj = device_attr
            if device_obj is None and hasattr(probe, "devices"):
                devices = probe.devices()
                if devices:
                    device_obj = next(iter(devices))
            if device_obj is None:
                return False, "unknown"

            platform_name = str(getattr(device_obj, "platform", "")).lower()
            if not platform_name:
                match = re.search(r"(cpu|gpu|cuda)", str(device_obj).lower())
                if match is not None:
                    platform_name = (
                        "gpu" if match.group(1) in {"gpu", "cuda"} else "cpu"
                    )

            is_gpu = platform_name in {"gpu", "cuda"}
            return is_gpu, str(device_obj)
        except Exception:
            return False, "probe-error"

    benchmark_path = Path(benchmark_dir)
    while True:
        task = task_queue.get()
        if task is None:
            break

        planned_run = task["planned_run"]
        run_id = str(planned_run["run_id"])
        probe_is_gpu, probe_device_str = _probe_device()
        result_queue.put(
            {
                "event": "run_started",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "probe_is_gpu": probe_is_gpu,
                "probe_device": probe_device_str,
                "run_id": run_id,
                "run_name": planned_run["run_name"],
                "inner_total": int(planned_run["inner_total"]),
            }
        )

        def sink(update: dict[str, Any]) -> None:
            result_queue.put(
                {
                    "event": "progress",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "run_id": run_id,
                    "iteration": int(update["iteration"]),
                    "energy": float(update["energy"]),
                    "grad": float(update["grad"]),
                }
            )

        try:
            run, _ = _execute_planned_run(
                planned_run=planned_run,
                common=common,
                plotting_cfg=plotting_cfg,
                benchmark_dir=benchmark_path,
                live_plot_every=live_plot_every,
                plot_n_samples=plot_n_samples,
                progress_sink=sink,
                print_warnings=False,
            )
            result_queue.put(
                {
                    "event": "run_complete",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "probe_is_gpu": probe_is_gpu,
                    "probe_device": probe_device_str,
                    "run_id": run_id,
                    "run": run,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "event": "run_error",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "probe_is_gpu": probe_is_gpu,
                    "probe_device": probe_device_str,
                    "run_id": run_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def run_all(config: dict[str, Any], benchmark_dir: Path) -> dict[str, int]:
    common = config["common_params"]
    plotting_cfg = config.get("plotting", {})
    output_h5 = benchmark_dir / "results.h5"
    ckpt_root = benchmark_dir / "model_checkpoints"
    plots_root = benchmark_dir / "plots"
    diagnostic_root = benchmark_dir / "diagnostic_plots"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    live_plot_every = _get_live_plot_every(common)
    plot_n_samples = int(common["plot_n_samples"])
    planned_runs = _prepare_planned_runs(config)
    total_runs = len(planned_runs)
    initialize_h5(output_h5, config)

    success_count = 0
    failed_count = 0
    fail_fast = bool(config.get("fail_fast", False))
    max_workers = int(config.get("parallel", {}).get("max_workers", 1))
    outer_bar = tqdm(
        total=total_runs, desc="Benchmark", position=0, leave=True, unit="run"
    )

    if max_workers <= 1:
        try:
            for planned_run in planned_runs:
                inner_bar = tqdm(
                    total=int(planned_run["inner_total"]),
                    desc=str(planned_run["run_name"]),
                    position=1,
                    leave=False,
                    unit="iter",
                )

                def inner_progress(info: dict[str, Any]) -> None:
                    iteration = int(info.get("iteration", 0)) + 1
                    delta = iteration - inner_bar.n
                    if delta > 0:
                        inner_bar.update(delta)
                    energy = float(info.get("energy", np.nan))
                    grad = float(info.get("riemann_grad_norm", np.nan))
                    if np.isfinite(energy) and np.isfinite(grad):
                        inner_bar.set_postfix_str(f"E={energy:.3e} G={grad:.3e}")
                    elif np.isfinite(energy):
                        inner_bar.set_postfix_str(f"E={energy:.3e}")

                try:
                    run, _ = _execute_planned_run(
                        planned_run=planned_run,
                        common=common,
                        plotting_cfg=plotting_cfg,
                        benchmark_dir=benchmark_dir,
                        live_plot_every=live_plot_every,
                        plot_n_samples=plot_n_samples,
                        progress_callback=inner_progress,
                    )
                    append_run_to_h5(output_h5, str(planned_run["run_id"]), run)
                    success_count += 1
                except Exception as exc:
                    failed_count += 1
                    tqdm.write(f"[error] run {planned_run['run_id']} failed: {exc}")
                    if fail_fast:
                        raise
                finally:
                    inner_bar.close()
                    outer_bar.update(1)
                    outer_bar.set_postfix_str(f"ok={success_count} fail={failed_count}")
                    jax.clear_caches()
                    gc.collect()
            return {"success": success_count, "failed": failed_count}
        finally:
            outer_bar.close()

    parallel_cfg = config.get("parallel", {})
    gpu_ids = list(parallel_cfg.get("gpu_ids", []))
    if not gpu_ids:
        gpu_ids = [0]

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    task_queues: dict[int, mp.Queue] = {}
    workers: dict[int, Any] = {}
    worker_gpu: dict[int, int] = {}
    worker_label: dict[int, str] = {}
    worker_cpu_warned: dict[int, bool] = {}
    worker_bars: dict[int, Any] = {}
    worker_active_run: dict[int, str | None] = {}

    disable_preallocate = bool(parallel_cfg.get("disable_preallocate", True))
    mem_fraction = parallel_cfg.get("mem_fraction", None)

    launch_workers = max_workers
    for worker_id in range(launch_workers):
        gpu_id = int(gpu_ids[worker_id % len(gpu_ids)])
        worker_gpu[worker_id] = gpu_id
        worker_label[worker_id] = f"GPU{gpu_id}"
        worker_cpu_warned[worker_id] = False
        worker_active_run[worker_id] = None

        env_prev = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if disable_preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        elif env_prev["XLA_PYTHON_CLIENT_PREALLOCATE"] is None:
            os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        if mem_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        elif env_prev["XLA_PYTHON_CLIENT_MEM_FRACTION"] is None:
            os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

        task_queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_parallel_worker_loop,
            args=(
                worker_id,
                gpu_id,
                task_queue,
                result_queue,
                str(benchmark_dir),
                common,
                plotting_cfg,
                live_plot_every,
                plot_n_samples,
            ),
            daemon=True,
        )
        proc.start()

        for key, val in env_prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        task_queues[worker_id] = task_queue
        workers[worker_id] = proc
        worker_bars[worker_id] = tqdm(
            total=1,
            desc=f"W{worker_id}|{worker_label[worker_id]}|idle",
            position=worker_id + 1,
            leave=False,
            unit="iter",
        )

    pending_idx = 0
    completed_runs = 0
    abort_due_to_fail_fast = False
    fail_fast_error: str | None = None

    def dispatch_to_worker(wid: int) -> bool:
        nonlocal pending_idx
        if pending_idx >= total_runs:
            worker_active_run[wid] = None
            return False
        planned_run = planned_runs[pending_idx]
        pending_idx += 1
        worker_active_run[wid] = str(planned_run["run_id"])
        task_queues[wid].put({"planned_run": planned_run})
        return True

    try:
        for worker_id in range(launch_workers):
            dispatch_to_worker(worker_id)

        while completed_runs < total_runs:
            try:
                msg = result_queue.get(timeout=0.5)
            except queue.Empty:
                dead_workers = [
                    wid
                    for wid, proc in workers.items()
                    if not proc.is_alive() and worker_active_run[wid] is not None
                ]
                if dead_workers:
                    wid = dead_workers[0]
                    failed_count += 1
                    completed_runs += 1
                    run_id = worker_active_run[wid]
                    worker_active_run[wid] = None
                    tqdm.write(
                        f"[error] worker {wid} on GPU {worker_gpu[wid]} exited unexpectedly"
                        + (f" while running {run_id}" if run_id else "")
                    )
                    worker_bars[wid].set_description_str(
                        f"W{wid}|{worker_label[wid]}|dead"
                    )
                    worker_bars[wid].set_postfix_str("failed")
                    outer_bar.update(1)
                    outer_bar.set_postfix_str(f"ok={success_count} fail={failed_count}")
                    if fail_fast:
                        abort_due_to_fail_fast = True
                        fail_fast_error = f"worker {wid} crashed"
                        break
                if (
                    all(not proc.is_alive() for proc in workers.values())
                    and completed_runs < total_runs
                ):
                    abort_due_to_fail_fast = True
                    fail_fast_error = "all workers exited before completing all runs"
                    break
                continue

            event = str(msg.get("event", ""))
            worker_id = int(msg.get("worker_id", -1))
            if worker_id not in worker_bars:
                continue
            bar = worker_bars[worker_id]

            if event == "run_started":
                probe_is_gpu = bool(msg.get("probe_is_gpu", False))
                probe_device = str(msg.get("probe_device", "unknown"))
                assigned_gpu = worker_gpu[worker_id]
                worker_label[worker_id] = (
                    f"GPU{assigned_gpu}" if probe_is_gpu else "CPU"
                )
                if not probe_is_gpu and not worker_cpu_warned[worker_id]:
                    tqdm.write(
                        f"[warn] worker {worker_id} assigned GPU {assigned_gpu} "
                        f"is running on CPU (probe: {probe_device})"
                    )
                    worker_cpu_warned[worker_id] = True
                inner_total = int(msg.get("inner_total", 1))
                run_id = str(msg.get("run_id", "unknown"))
                run_name = str(msg.get("run_name", run_id))
                bar.reset(total=max(1, inner_total))
                bar.n = 0
                bar.set_description_str(
                    f"W{worker_id}|{worker_label[worker_id]}|{run_name}"
                )
                bar.set_postfix_str("")
                bar.refresh()
            elif event == "progress":
                iteration = int(msg.get("iteration", 0))
                delta = iteration - bar.n
                if delta > 0:
                    bar.update(delta)
                energy = float(msg.get("energy", np.nan))
                grad = float(msg.get("grad", np.nan))
                if np.isfinite(energy) and np.isfinite(grad):
                    bar.set_postfix_str(f"E={energy:.3e} G={grad:.3e}")
                elif np.isfinite(energy):
                    bar.set_postfix_str(f"E={energy:.3e}")
            elif event == "run_complete":
                run_id = str(msg["run_id"])
                run = msg["run"]
                append_run_to_h5(output_h5, run_id, run)
                success_count += 1
                completed_runs += 1
                worker_active_run[worker_id] = None
                outer_bar.update(1)
                outer_bar.set_postfix_str(f"ok={success_count} fail={failed_count}")
                bar.set_postfix_str("done")
                gc.collect()
                if not dispatch_to_worker(worker_id):
                    bar.set_description_str(
                        f"W{worker_id}|{worker_label[worker_id]}|idle"
                    )
            elif event == "run_error":
                run_id = str(msg.get("run_id", "unknown"))
                err = str(msg.get("error", "unknown error"))
                tb = str(msg.get("traceback", ""))
                failed_count += 1
                completed_runs += 1
                worker_active_run[worker_id] = None
                tqdm.write(
                    f"[error] run {run_id} failed on worker {worker_id}"
                    f" ({worker_label[worker_id]}): {err}"
                )
                if tb:
                    tqdm.write(tb)
                outer_bar.update(1)
                outer_bar.set_postfix_str(f"ok={success_count} fail={failed_count}")
                bar.set_postfix_str("failed")
                if fail_fast:
                    abort_due_to_fail_fast = True
                    fail_fast_error = f"run {run_id} failed: {err}"
                    break
                if not dispatch_to_worker(worker_id):
                    bar.set_description_str(
                        f"W{worker_id}|{worker_label[worker_id]}|idle"
                    )

        for worker_id, tq in task_queues.items():
            try:
                tq.put(None)
            except Exception:
                pass

        for worker_id, proc in workers.items():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

        if abort_due_to_fail_fast:
            raise RuntimeError(fail_fast_error or "fail_fast triggered")
        return {"success": success_count, "failed": failed_count}
    finally:
        outer_bar.close()
        for bar in worker_bars.values():
            bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runner for flow methods")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tests/benchmark_config.json"),
        help="Path to JSON config",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip runs and regenerate plots from existing .h5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        h5_path = latest_h5(output_root)
        if h5_path is None or not h5_path.exists():
            raise FileNotFoundError("No .h5 file found for --plot-only mode")
        experiment_config_h5 = load_experiment_config_from_h5(h5_path)
        loaded_runs = load_runs_from_h5(h5_path)
        plot_config = make_plot_config(config, experiment_config_h5)
        missing_ckpts = [
            str((h5_path.parent / run["model_ckpt_relpath"]))
            for run in loaded_runs
            if not (h5_path.parent / run["model_ckpt_relpath"]).exists()
        ]
        if missing_ckpts:
            preview = "\n".join(missing_ckpts[:5])
            raise FileNotFoundError(
                "Some checkpoint directories referenced by this .h5 are missing. "
                f"First missing entries:\n{preview}"
            )
        plots_dir = h5_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_convergence_plots(plot_config, loaded_runs, plots_dir)
        save_scatter_plots(plot_config, loaded_runs, h5_path.parent, plots_dir)
        return

    benchmark_dir = create_benchmark_session_dir(output_root)
    output_h5 = benchmark_dir / "results.h5"
    plots_dir = benchmark_dir / "plots"
    run_all(config, benchmark_dir)
    experiment_config_h5 = load_experiment_config_from_h5(output_h5)
    runs_for_scatter = load_runs_from_h5(output_h5)
    plot_config = make_plot_config(config, experiment_config_h5)
    save_convergence_plots(plot_config, runs_for_scatter, plots_dir)
    save_scatter_plots(plot_config, runs_for_scatter, benchmark_dir, plots_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import gc
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from flax.training import orbax_utils
from jax.scipy.special import logsumexp

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


def build_model(common: dict[str, Any], seed: int) -> tuple[ParametricModel, dict[str, Any]]:
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


def build_problem(distribution_cfg: dict[str, Any], common: dict[str, Any]) -> tuple[Potential, dict[str, Any]]:
    dist_name = distribution_cfg["name"].lower()
    dim = int(common["dimension"])

    if dist_name == "gaussian":
        mean_value = float(distribution_cfg.get("mean_value", 2.0))
        mean = jnp.full((dim,), mean_value)
        if "sigma_diag" in distribution_cfg:
            sigma_diag = jnp.asarray(distribution_cfg["sigma_diag"], dtype=jnp.float32)
            if sigma_diag.shape[0] != dim:
                raise ValueError("gaussian sigma_diag length must match dimension")
        else:
            sigma_diag = jnp.ones((dim,), dtype=jnp.float32)
            if dim >= 1:
                sigma_diag = sigma_diag.at[0].set(float(distribution_cfg.get("sigma_first", 1000.0)))
            if dim >= 2:
                sigma_diag = sigma_diag.at[1].set(float(distribution_cfg.get("sigma_second", 10.0)))
        sigma_inv = jnp.diag(sigma_diag)
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
    potential = Potential(linear=linear_potential, internal=internal_potential, interaction=None)
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


def sample_reference(common: dict[str, Any], seed: int, n_samples: int | None = None) -> jnp.ndarray:
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
    run_namespace: str,
    run_id: str,
) -> dict[str, Any]:
    model, model_cfg = build_model(common, run_seed)
    potential, potential_meta = build_problem(distribution_cfg, common)
    g_mat = G_matrix(model)

    n_samples = int(common["N_samples"])
    max_iterations = int(common.get("max_iterations", 300))
    stepsize = float(common["stepsize"])
    tolerance = float(common.get("tolerance", 1e-4))
    solver = common.get("linear_solver", "cg")
    z_samples = sample_reference(common, run_seed + 13, n_samples=int(common.get("eval_samples", 300)))

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
            progress_every=int(method_params.get("progress_every", common.get("progress_every", 100))),
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
            plot_frequency=int(method_params.get("progress_every", common.get("progress_every", 100))),
            save_param_trajectory=False,
            regularization_factor_gamma=float(method_params.get("regularization", 1e-3)),
            regularization_method_gamma=str(method_params.get("regularization_kind", "l2")),
            ensure_descent=bool(method_params.get("ensure_descent", True)),
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
            solver_regularization=float(method_params.get("solver_regularization", 1e-6)),
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
            plot_frequency=int(method_params.get("progress_every", common.get("progress_every", 100))),
            save_param_trajectory=False,
        )
        final_model = nnx.merge(graphdef, final_params)
        energies = np.asarray(history["energies"], dtype=np.float64)
        riem_grad = np.asarray(history["riemann_grad_history"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported method: {method}")

    runtime_sec = time.perf_counter() - t0

    ckpt_relpath = Path("model_checkpoints") / run_namespace / run_id
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs["config_json"] = json.dumps(config)
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
        grp.attrs["method_params_json"] = json.dumps(run["method_params"], sort_keys=True)
        grp.attrs["model_config_json"] = json.dumps(run["model_config"], sort_keys=True)
        grp.attrs["common_json"] = json.dumps(run["common"], sort_keys=True)
        grp.attrs["runtime_sec"] = float(run["runtime_sec"])
        grp.attrs["model_ckpt_relpath"] = run["model_ckpt_relpath"]
        grp.create_dataset("energy_history", data=run["energy_history"])
        grp.create_dataset("riemann_grad_history", data=run["riemann_grad_history"])
        h5.attrs["run_count"] = int(h5.attrs.get("run_count", 0)) + 1
        h5.flush()


def load_runs_from_h5(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with h5py.File(path, "r") as h5:
        config = json.loads(h5.attrs["config_json"])
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
                    "energy_history": np.asarray(grp["energy_history"][:], dtype=np.float64),
                    "riemann_grad_history": np.asarray(grp["riemann_grad_history"][:], dtype=np.float64),
                }
            )
    return config, out


def format_params(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    chunks = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            if abs(value) >= 1e-2 and abs(value) < 1e3:
                rendered = f"{value:.4g}"
            else:
                rendered = f"{value:.2e}"
        else:
            rendered = str(value)
        chunks.append(f"{key}={rendered}")
    return ", ".join(chunks)


def style_for_run(
    plotting_cfg: dict[str, Any],
    method: str,
    run_idx_within_method: int,
) -> dict[str, Any]:
    colors = plotting_cfg.get("colors", {})
    linestyles = plotting_cfg.get("linestyles", ["-", "--", ":", "-."])
    linewidths = plotting_cfg.get("linewidths", [2.0, 1.5, 1.0])
    markers = plotting_cfg.get("markers", ["o", "^", "s", "D", "x", "P"])
    return {
        "color": colors.get(method, "black"),
        "linestyle": linestyles[run_idx_within_method % len(linestyles)],
        "linewidth": linewidths[run_idx_within_method % len(linewidths)],
        "marker": markers[run_idx_within_method % len(markers)],
    }


def save_convergence_plots(
    runs: list[dict[str, Any]],
    plotting_cfg: dict[str, Any],
    output_dir: Path,
    file_prefix: str,
) -> None:
    distributions = sorted({run["distribution"] for run in runs})
    for distribution in distributions:
        dist_runs = [run for run in runs if run["distribution"] == distribution]
        fig, axes = plt.subplots(1, 2, figsize=tuple(plotting_cfg.get("figsize", [16, 6])))

        per_method_counts: dict[str, int] = {}
        for run in dist_runs:
            m = run["method"]
            idx = per_method_counts.get(m, 0)
            per_method_counts[m] = idx + 1
            style = style_for_run(plotting_cfg, m, idx)
            label = f"{m} | {format_params(run['method_params'])}"
            axes[0].plot(
                run["energy_history"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label,
            )
            axes[1].plot(
                run["riemann_grad_history"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label,
            )

        axes[0].set_title(f"Energy history ({distribution})")
        axes[1].set_title(f"Riemannian gradient history ({distribution})")
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_xlabel("iteration")
        axes[1].set_xlabel("iteration")
        axes[0].set_ylabel("energy")
        axes[1].set_ylabel("riemann grad norm")
        axes[0].grid(True)
        axes[1].grid(True)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{file_prefix}_{distribution}_convergence.pdf"
        fig.savefig(out)
        plt.close(fig)


def restore_model_from_run(run: dict[str, Any], checkpoint_root: Path) -> ParametricModel:
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
    restored_state = checkpointer.restore(str(ckpt_path.absolute()), item=template_state)
    nnx.update(model, restored_state)
    return model


def generate_samples(model: ParametricModel, dim: int, n_samples: int, seed: int) -> np.ndarray:
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
    file_prefix: str,
) -> None:
    plotting_cfg = config.get("plotting", {})
    common = config["common_params"]
    dim = int(common["dimension"])
    n_samples = int(common.get("plot_n_samples", 300))
    distributions = sorted({run["distribution"] for run in runs})

    for distribution in distributions:
        dist_runs = [run for run in runs if run["distribution"] == distribution]
        gf_runs = [run for run in dist_runs if run["method"] == "gradient_flow"]
        if not gf_runs:
            print(f"[warn] skip scatter for {distribution}: no gradient_flow run")
            continue
        gf_baseline = gf_runs[0]
        gf_model = restore_model_from_run(gf_baseline, checkpoint_root)
        gf_samples = generate_samples(gf_model, dim, n_samples, seed=123)

        methods = sorted({run["method"] for run in dist_runs if run["method"] != "gradient_flow"})
        for method in methods:
            method_runs = [run for run in dist_runs if run["method"] == method]
            if not method_runs:
                continue

            fig, ax = plt.subplots(figsize=tuple(plotting_cfg.get("scatter_figsize", [8, 8])))

            ax.scatter(
                gf_samples[:, 0],
                gf_samples[:, 1],
                s=float(plotting_cfg.get("scatter_size", 20)),
                alpha=float(plotting_cfg.get("gf_alpha", 0.5)),
                color=plotting_cfg.get("colors", {}).get("gradient_flow", "blue"),
                label=f"gradient_flow | {format_params(gf_baseline['method_params'])}",
            )

            all_method_samples = []
            for idx, run in enumerate(method_runs):
                style = style_for_run(plotting_cfg, method, idx)
                model = restore_model_from_run(run, checkpoint_root)
                samples = generate_samples(model, dim, n_samples, seed=1000 + idx)
                all_method_samples.append(samples)
                ax.scatter(
                    samples[:, 0],
                    samples[:, 1],
                    s=float(plotting_cfg.get("scatter_size", 20)),
                    alpha=float(plotting_cfg.get("method_alpha", 0.5)),
                    color=style["color"],
                    marker=style["marker"],
                    label=f"{method} | {format_params(run['method_params'])}",
                )

            try:
                dist_cfg = get_dist_cfg(config, distribution)
                plot_pot = build_plot_potential_2d(dist_cfg)
                if plot_pot is not None:
                    joint = np.concatenate([gf_samples[:, :2]] + [s[:, :2] for s in all_method_samples], axis=0)
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
            ax.legend(fontsize=7)
            fig.tight_layout()
            out = output_dir / f"{file_prefix}_{distribution}_{method}_scatter.pdf"
            fig.savefig(out)
            plt.close(fig)


def choose_output_h5(output_dir: Path, experiment_name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{experiment_name}_{ts}.h5"


def latest_h5(output_dir: Path) -> Path | None:
    files = sorted(output_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_all(config: dict[str, Any], output_h5: Path) -> dict[str, int]:
    common = config["common_params"]
    dist_cfgs = distribution_grid(config["distributions"])
    base_seed = int(common.get("seed", 0))
    ckpt_root = output_h5.parent / "model_checkpoints" / output_h5.stem
    ckpt_root.mkdir(parents=True, exist_ok=True)
    initialize_h5(output_h5, config)

    run_counter = 0
    success_count = 0
    failed_count = 0
    fail_fast = bool(config.get("fail_fast", False))
    for dist in dist_cfgs:
        for method, m_cfg in config["methods"].items():
            for params in method_grid(m_cfg):
                params = dict(params)
                params.setdefault("stepsize", common["stepsize"])
                params.setdefault("max_iterations", common.get("max_iterations", 300))
                params.setdefault("tolerance", common.get("tolerance", 1e-4))
                print(
                    f"[run] dist={dist['name']} method={method} "
                    f"params={json.dumps(params, sort_keys=True)}"
                )
                run_id = f"run_{run_counter:04d}"
                try:
                    run = run_single(
                        method=method,
                        method_params=params,
                        distribution_cfg=dist,
                        common=common,
                        run_seed=base_seed + run_counter,
                        checkpoint_root=output_h5.parent,
                        run_namespace=output_h5.stem,
                        run_id=run_id,
                    )
                    append_run_to_h5(output_h5, run_id, run)
                    success_count += 1
                except Exception as exc:
                    failed_count += 1
                    print(f"[error] run {run_id} failed: {exc}")
                    if fail_fast:
                        raise
                finally:
                    run_counter += 1
                    jax.clear_caches()
                    gc.collect()

    return {"success": success_count, "failed": failed_count}


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
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=None,
        help="Path to .h5 file used in --plot-only mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
        help="Directory for .h5 and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = config.get("experiment_name", "benchmark")

    if args.plot_only:
        h5_path = args.input_h5 if args.input_h5 else latest_h5(output_dir)
        if h5_path is None or not h5_path.exists():
            raise FileNotFoundError("No .h5 file found for --plot-only mode")
        print(f"[plot-only] loading {h5_path}")
        loaded_config, loaded_runs = load_runs_from_h5(h5_path)
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
        digest = hashlib.sha1(str(h5_path).encode("utf-8")).hexdigest()[:8]
        prefix = f"{experiment_name}_{digest}"
        save_convergence_plots(loaded_runs, loaded_config["plotting"], output_dir, prefix)
        save_scatter_plots(loaded_config, loaded_runs, h5_path.parent, output_dir, prefix)
        print("[done] plots regenerated")
        return

    output_h5 = choose_output_h5(output_dir, experiment_name)
    stats = run_all(config, output_h5)
    digest = hashlib.sha1(str(output_h5).encode("utf-8")).hexdigest()[:8]
    prefix = f"{experiment_name}_{digest}"
    config_for_scatter, runs_for_scatter = load_runs_from_h5(output_h5)
    save_convergence_plots(runs_for_scatter, config_for_scatter["plotting"], output_dir, prefix)
    save_scatter_plots(config_for_scatter, runs_for_scatter, output_h5.parent, output_dir, prefix)
    print(f"[done] successful runs: {stats['success']}, failed runs: {stats['failed']}")
    print(f"[done] results saved to {output_h5}")


if __name__ == "__main__":
    main()

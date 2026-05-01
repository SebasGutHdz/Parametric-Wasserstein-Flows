#!/usr/bin/env python3
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.optimize import rosen

import argparse
import gc
import hashlib
import itertools
import json
import multiprocessing as mp
import queue
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

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
from flows.nnx_first_order import run_adam, run_sgd
from functionals.CrossEntropy import CrossEntropyEnergy
from functionals.MMD import MMDEnergy, bandwidth_median
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
    "sgd": "sgd",
    "adam": "adam",
}


FUNCTIONAL_KIND_ALIASES = {
    "kl": "KL",
    "mmd": "MMD",
    "crossentropy": "CrossEntropy",
    "cross-entropy": "CrossEntropy",
    "cross_entropy": "CrossEntropy",
}


def normalize_functional_kind(kind_raw: Any) -> str:
    key = str(kind_raw).strip().lower()
    canonical = FUNCTIONAL_KIND_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            "functional.kind must be one of: "
            + ", ".join(sorted(set(FUNCTIONAL_KIND_ALIASES.values())))
        )
    return canonical


def normalize_distribution_entry(distribution_cfg: Any) -> dict[str, Any]:
    if isinstance(distribution_cfg, str):
        return {"name": distribution_cfg}
    if isinstance(distribution_cfg, dict):
        return dict(distribution_cfg)
    raise ValueError(f"Invalid distribution entry: {distribution_cfg}")


def normalize_problem_entry(problem_cfg: Any) -> dict[str, Any]:
    if not isinstance(problem_cfg, dict):
        raise ValueError(f"Invalid problem entry: {problem_cfg}")
    if "functional" not in problem_cfg or "distribution" not in problem_cfg:
        raise ValueError(
            "Each problem must define both 'functional' and 'distribution' keys"
        )

    functional_cfg_raw = problem_cfg["functional"]
    if not isinstance(functional_cfg_raw, dict):
        raise ValueError("problem.functional must be a dict")
    if "name" in functional_cfg_raw:
        raise ValueError("functional.name is not supported; use functional.kind")
    if "kind" not in functional_cfg_raw:
        raise ValueError("problem.functional.kind is required")

    functional_cfg = dict(functional_cfg_raw)
    functional_cfg["kind"] = normalize_functional_kind(functional_cfg["kind"])
    distribution_cfg = normalize_distribution_entry(problem_cfg["distribution"])
    return {
        "functional": functional_cfg,
        "distribution": distribution_cfg,
    }


def problem_grid(problems_cfg: list[Any]) -> list[dict[str, Any]]:
    return [normalize_problem_entry(item) for item in problems_cfg]


def distribution_label(distribution_cfg: dict[str, Any]) -> str:
    if "name" in distribution_cfg:
        return str(distribution_cfg["name"])
    if "file" in distribution_cfg:
        return f"file:{distribution_cfg['file']}"
    return "unknown_distribution"


def problem_key(problem_cfg: dict[str, Any]) -> str:
    return json.dumps(problem_cfg, sort_keys=True)


def problem_label(problem_cfg: dict[str, Any]) -> str:
    kind = problem_cfg["functional"]["kind"]
    return f"{kind} | {distribution_label(problem_cfg['distribution'])}"


def problem_slug(problem_cfg: dict[str, Any]) -> str:
    kind_slug = sanitize_component(problem_cfg["functional"]["kind"])
    dist_cfg = problem_cfg["distribution"]
    if "name" in dist_cfg:
        dist_part = str(dist_cfg["name"])
    elif "file" in dist_cfg:
        dist_part = Path(str(dist_cfg["file"])).stem
    else:
        dist_part = "distribution"
    digest = hashlib.sha1(problem_key(problem_cfg).encode("utf-8")).hexdigest()[:8]
    return f"{kind_slug}__{sanitize_component(dist_part)}__{digest}"


def _attr_to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as infile:
        config = json.load(infile)

    for key in ["common_params", "methods", "plotting"]:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    has_problems = "problems" in config
    has_distributions = "distributions" in config
    if has_problems and has_distributions:
        raise ValueError(
            "Config cannot define both 'problems' and deprecated 'distributions'"
        )
    if not has_problems and not has_distributions:
        raise ValueError("Missing required config key: problems")

    if has_distributions:
        if not isinstance(config["distributions"], list):
            raise ValueError("config['distributions'] must be a list")
        warnings.warn(
            "[warn] Top-level 'distributions' is deprecated and interpreted as KL "
            "problems; please migrate to 'problems'.",
            stacklevel=2,
        )
        config["problems"] = [
            {
                "functional": {"kind": "KL"},
                "distribution": normalize_distribution_entry(dist_cfg),
            }
            for dist_cfg in config["distributions"]
        ]
        config.pop("distributions", None)

    if not isinstance(config["problems"], list):
        raise ValueError("config['problems'] must be a list")
    config["problems"] = problem_grid(config["problems"])

    if not isinstance(config["methods"], dict):
        raise ValueError("config['methods'] must be a dict")

    normalized_methods = {}
    for raw_name, method_cfg in config["methods"].items():
        canonical = METHOD_ALIASES.get(raw_name, raw_name)
        if canonical not in {
            "gradient_flow",
            "anderson",
            "memoryless_qn",
            "sgd",
            "adam",
        }:
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


def build_model(
    common: dict[str, Any],
    seed: int,
    dim_override: int | None = None,
    dataset_context: dict[str, Any] | None = None,
) -> tuple[ParametricModel, dict[str, Any]]:
    dim = int(dim_override if dim_override is not None else common["dimension"])
    n_hidden = int(common["n_hidden"])
    width_hidden = int(common["width_hidden"])
    rhs_model = common.get("rhs_model", "mlp")
    shape_x = common.get("shape_x", None)
    if rhs_model == "concat_conv2d" and shape_x is None:
        if dataset_context is None or dataset_context.get("image_shape") is None:
            raise ValueError(
                "rhs_model='concat_conv2d' requires an image dataset or "
                "common_params.shape_x"
            )
        shape_x = list(dataset_context["image_shape"])
    model_cfg = {
        "parametric_map": common.get("parametric_map", "node"),
        "rhs_model": rhs_model,
        "activation_fn": common.get("activation_fn", "tanh"),
        "time_dependent": bool(common.get("time_dependent", True)),
        "solver": common.get("ode_solver", "euler"),
        "dt0": float(common.get("dt0", 0.01)),
        "ref_density": common.get("ref_density", "gaussian"),
        "scale_factor": float(common.get("scale_factor", 1.0)),
        "architecture": [dim, n_hidden, width_hidden],
        "seed": seed,
    }
    if shape_x is not None:
        model_cfg["shape_x"] = [int(v) for v in shape_x]
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
        shape_x=model_cfg.get("shape_x", None),
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


def _build_kl_problem(
    distribution_cfg: dict[str, Any], common: dict[str, Any]
) -> Potential:
    dist_name = str(distribution_cfg["name"]).lower()
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

    elif dist_name in {"rosen", "Rosenbrock"}:
        potential_fn = lambda _x: rosen(_x.T) / 20.0
    else:
        raise ValueError(f"Unknown distribution: {distribution_cfg['name']}")

    linear_potential = LinearPotential(potential_fn=potential_fn, coeff=1.0)
    internal_potential = InternalPotential(
        functional="entropy", coeff=1.0, method="exact", prob_dim=dim
    )
    potential = Potential(
        linear=linear_potential, internal=internal_potential, interaction=None
    )
    return potential


def checkerboard_generator(n_samples: int, resample_each: int, seed: int = 3):
    key = jax.random.PRNGKey(seed)
    while True:
        key, points_key, shift_x_key, shift_y_key = jax.random.split(key, 4)
        points = jax.random.uniform(points_key, (n_samples, 2))
        shifts_x = jax.random.randint(shift_x_key, (n_samples,), 0, 4) - 2
        shifts_y = (
            jax.random.randint(shift_y_key, (n_samples,), 0, 2) * 2 + shifts_x % 2 - 2
        )
        points = points.at[:, 0].add(shifts_x)
        points = points.at[:, 1].add(shifts_y)
        for _ in range(resample_each):
            yield points


def two_spirals_generator(n_samples: int, resample_each: int, seed: int = 3):
    key = jax.random.PRNGKey(seed)
    while True:
        key, n_key, shift_x_key, shift_y_key, noise_key = jax.random.split(key, 5)
        n = (
            jnp.sqrt(jax.random.uniform(n_key, (n_samples // 2, 1)))
            * 540
            * (2 * jnp.pi)
            / 360
        )
        d1x = (
            -jnp.cos(n) * n + jax.random.uniform(shift_x_key, (n_samples // 2, 1)) * 0.5
        )
        d1y = (
            jnp.sin(n) * n + jax.random.uniform(shift_y_key, (n_samples // 2, 1)) * 0.5
        )
        x = jnp.vstack((jnp.hstack((d1x, d1y)), jnp.hstack((-d1x, -d1y)))) / 3
        x += jax.random.uniform(noise_key, x.shape) * 0.1
        for _ in range(resample_each):
            yield x


def eight_gaussians_generator(n_samples: int, resample_each: int, seed: int = 3):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 8)
    centers = 4.0 * jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)
    key = jax.random.PRNGKey(seed)

    while True:
        key, idx_key, blob_key = jax.random.split(key, 3)
        blob = jax.random.normal(blob_key, (n_samples, 2)) * 0.5
        shift_ids = jax.random.randint(idx_key, n_samples, minval=0, maxval=7)

        x = blob + centers[shift_ids, :]
        x /= 1.414

        for _ in range(resample_each):
            yield x


def file_dataset_generator(file_path: str, n_samples: int, resample_each: int):
    raise NotImplementedError(
        "File-based dataset generator is not implemented yet "
        f"(requested file: {file_path})"
    )


DATASET_DISTRIBUTIONS = {"mnist", "fashion", "miniboone"}
IMAGE_DATASETS = {"mnist", "fashion"}


def distribution_name(distribution_cfg: dict[str, Any]) -> str | None:
    if "name" not in distribution_cfg:
        return None
    return str(distribution_cfg["name"]).lower()


def is_dataset_distribution(distribution_cfg: dict[str, Any]) -> bool:
    name = distribution_name(distribution_cfg)
    return name in DATASET_DISTRIBUTIONS


def is_image_distribution(distribution_cfg: dict[str, Any]) -> bool:
    name = distribution_name(distribution_cfg)
    return name in IMAGE_DATASETS


def flatten_and_normalize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float32).reshape((X_train.shape[0], -1))
    test = np.asarray(X_test, dtype=np.float32).reshape((X_test.shape[0], -1))

    mu = train.mean(axis=0)
    s = train.std(axis=0)
    safe_s = np.where(s > eps, s, 1.0).astype(np.float32)

    train_norm = ((train - mu) / safe_s).astype(np.float32)
    test_norm = ((test - mu) / safe_s).astype(np.float32)
    return train_norm, test_norm, mu.astype(np.float32), s.astype(np.float32), safe_s


def prepare_dataset_context(
    distribution_cfg: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    name = distribution_name(distribution_cfg)
    if name not in DATASET_DISTRIBUTIONS:
        raise ValueError(f"Unsupported dataset distribution: {distribution_cfg}")

    data_root = distribution_cfg.get("data_root", None)
    X_train, X_test, _, _ = load_data(name, data_root=data_root)
    X_train, X_test, mu, s, safe_s = flatten_and_normalize_train_test(
        X_train,
        X_test,
    )
    dataset_dim = int(X_train.shape[1])
    config_dim = int(common.get("dimension", dataset_dim))
    if config_dim != dataset_dim:
        warnings.warn(
            f"common_params.dimension={config_dim} is ignored for dataset "
            f"'{name}'; using dataset dimension {dataset_dim}.",
            stacklevel=2,
        )

    image_shape = (28, 28) if name in IMAGE_DATASETS else None
    return {
        "name": name,
        "X_train": X_train,
        "X_test": X_test,
        "mu": mu,
        "s": s,
        "safe_s": safe_s,
        "dim": dataset_dim,
        "is_image": name in IMAGE_DATASETS,
        "image_shape": image_shape,
    }


def dataset_batch_generator(
    X_train: np.ndarray,
    n_samples: int,
    resample_each: int,
    seed: int = 3,
):
    X_train = jnp.array(X_train)
    key = jax.random.PRNGKey(seed)
    n_train = int(X_train.shape[0])
    n_batches = int(np.ceil(n_train / n_samples))
    while True:
        key, idx_key = jax.random.split(key)
        X_perm = jax.random.permutation(key, X_train)
        for i in range(n_batches):
            batch = jnp.asarray(X_perm[i : i + n_samples], dtype=jnp.float32)
            for _ in range(resample_each):
                yield batch


def sample_dataset_rows(X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    key = jax.random.PRNGKey(seed)
    idx = np.asarray(jax.random.randint(key, (n_samples,), 0, int(X.shape[0])))
    return np.asarray(X[idx], dtype=np.float32)


def build_target_generator(
    distribution_cfg: dict[str, Any],
    common: dict[str, Any],
    dataset_context: dict[str, Any] | None = None,
):
    if "n_samples" not in distribution_cfg:
        raise ValueError("distribution.n_samples is required for generative problems")
    if "resample_each" not in distribution_cfg:
        raise ValueError(
            "distribution.resample_each is required for generative problems"
        )

    n_samples = int(distribution_cfg["n_samples"])
    resample_each = int(distribution_cfg["resample_each"])
    if n_samples <= 0:
        raise ValueError("distribution.n_samples must be > 0")
    if resample_each <= 0:
        raise ValueError("distribution.resample_each must be > 0")

    has_name = "name" in distribution_cfg
    has_file = "file" in distribution_cfg
    if has_name == has_file:
        raise ValueError(
            "Generative distribution must define exactly one of 'name' or 'file'"
        )

    if has_name:
        dist_name = str(distribution_cfg["name"]).lower()
        seed = int(distribution_cfg.get("seed", 3))
        if dist_name in DATASET_DISTRIBUTIONS:
            if dataset_context is None:
                dataset_context = prepare_dataset_context(distribution_cfg, common)
            return dataset_batch_generator(
                X_train=dataset_context["X_train"],
                n_samples=n_samples,
                resample_each=resample_each,
                seed=seed,
            )
        if dist_name == "checkerboard":
            dim = int(common["dimension"])
            if dim != 2:
                raise ValueError(
                    "checkerboard generator currently supports only dimension=2"
                )
            return checkerboard_generator(
                n_samples=n_samples,
                resample_each=resample_each,
                seed=seed,
            )
        if dist_name == "2spirals":
            return two_spirals_generator(
                n_samples=n_samples,
                resample_each=resample_each,
                seed=seed,
            )
        if dist_name == "8gaussians":
            return eight_gaussians_generator(
                n_samples=n_samples,
                resample_each=resample_each,
                seed=seed,
            )
        raise ValueError(
            "Unknown generative toy distribution name: " f"{distribution_cfg['name']}"
        )

    return file_dataset_generator(
        file_path=str(distribution_cfg["file"]),
        n_samples=n_samples,
        resample_each=resample_each,
    )


def build_problem(
    problem_cfg: dict[str, Any],
    common: dict[str, Any],
    dataset_context: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    problem = normalize_problem_entry(problem_cfg)
    functional_cfg = dict(problem["functional"])
    distribution_cfg = dict(problem["distribution"])
    functional_kind = functional_cfg["kind"]

    if functional_kind == "KL":
        potential = _build_kl_problem(distribution_cfg, common)
        return potential, {"problem": problem}

    target_generator = build_target_generator(
        distribution_cfg,
        common,
        dataset_context=dataset_context,
    )
    dataset_meta: dict[str, Any] = {}
    if dataset_context is not None:
        dataset_meta = {
            "dataset_name": dataset_context["name"],
            "dataset_dim": int(dataset_context["dim"]),
            "dataset_is_image": bool(dataset_context["is_image"]),
            "dataset_mu_mean": float(np.mean(dataset_context["mu"])),
            "dataset_mu_std": float(np.std(dataset_context["mu"])),
            "dataset_std_mean": float(np.mean(dataset_context["s"])),
            "dataset_std_std": float(np.std(dataset_context["s"])),
        }

    if functional_kind == "MMD":
        bw_multipliers_raw = functional_cfg.get("bw_multipliers", None)
        if not isinstance(bw_multipliers_raw, list) or not bw_multipliers_raw:
            raise ValueError(
                "functional.bw_multipliers must be a non-empty list for MMD"
            )
        bw_multipliers = [float(v) for v in bw_multipliers_raw]
        bandwidth_samples = int(functional_cfg.get("bandwidth_samples", 2000))
        if bandwidth_samples < 2:
            raise ValueError("functional.bandwidth_samples must be >= 2")

        bw_reference = jnp.asarray(next(target_generator), dtype=jnp.float32)
        bw_reference = bw_reference[: min(bandwidth_samples, bw_reference.shape[0])]
        bw = float(bandwidth_median(bw_reference))
        bandwidths = jnp.asarray(
            [bw * mult for mult in bw_multipliers], dtype=jnp.float32
        )
        potential = MMDEnergy(target_generator, bandwidths)
        return potential, {
            "problem": problem,
            **dataset_meta,
            "mmd_bandwidth_median": bw,
            "mmd_bandwidths": [float(v) for v in np.asarray(bandwidths)],
        }

    if functional_kind == "CrossEntropy":
        trace_method = str(functional_cfg.get("trace_method", "hutchinson"))
        potential = CrossEntropyEnergy(target_generator, trace_method=trace_method)
        return potential, {"problem": problem, **dataset_meta}

    raise ValueError(f"Unsupported functional kind: {functional_kind}")


def build_plot_potential_2d(distribution_cfg: dict[str, Any]) -> LinearPotential | None:
    if "name" not in distribution_cfg:
        return None
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

    elif _name in {"rosen", "Rosenbrock"}:
        potential_fn = lambda _x: rosen(_x.T) / 20.0
        return LinearPotential(potential_fn=potential_fn, coef=1.0)

    return None


def sample_reference(
    common: dict[str, Any],
    seed: int,
    n_samples: int | None = None,
    dim_override: int | None = None,
) -> jnp.ndarray:
    dim = int(dim_override if dim_override is not None else common["dimension"])
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


FIRST_ORDER_CONTROL_KEYS = {
    "stepsize",
    "max_iterations",
    "tolerance",
    "progress_every",
    "verbose",
    "use_tqdm",
}


def extract_first_order_optimizer_kwargs(
    method_params: dict[str, Any],
) -> dict[str, Any]:
    optimizer_kwargs = {
        key: value
        for key, value in method_params.items()
        if key not in FIRST_ORDER_CONTROL_KEYS
    }
    if "learning_rate" in optimizer_kwargs:
        raise ValueError(
            "Use 'stepsize' in benchmark config; it is mapped to optimizer learning_rate"
        )
    return optimizer_kwargs


def run_single(
    method: str,
    method_params: dict[str, Any],
    problem_cfg: dict[str, Any],
    common: dict[str, Any],
    run_seed: int,
    checkpoint_root: Path,
    run_id: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    diagnostic_sample_size: int | None = None,
    dataset_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem = normalize_problem_entry(problem_cfg)
    if dataset_context is None and is_dataset_distribution(problem["distribution"]):
        dataset_context = prepare_dataset_context(problem["distribution"], common)
    problem_dim = (
        int(dataset_context["dim"])
        if dataset_context is not None
        else int(common["dimension"])
    )

    model, model_cfg = build_model(
        common,
        run_seed,
        dim_override=problem_dim,
        dataset_context=dataset_context,
    )
    potential, potential_meta = build_problem(
        problem,
        common,
        dataset_context=dataset_context,
    )
    g_mat = G_matrix(model)

    n_samples = int(common["N_samples"])
    max_iterations = int(common.get("max_iterations", 300))
    tolerance = float(common.get("tolerance", 1e-4))
    solver = common.get("linear_solver", "cg")
    z_samples = sample_reference(
        common,
        run_seed + 13,
        n_samples=int(common.get("eval_samples", 300)),
        dim_override=problem_dim,
    )

    t0 = time.perf_counter()
    euclid_grad = np.asarray([], dtype=np.float64)

    if method == "gradient_flow":
        history = run_gradient_flow(
            model,
            z_samples,
            g_mat,
            potential,
            N_samples=n_samples,
            h=float(method_params.get("stepsize", common.get("stepsize", 1e-3))),
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
            step_size=float(
                method_params.get("stepsize", common.get("stepsize", 1e-3))
            ),
            memory_size=int(method_params.get("memory_size", 8)),
            relaxation=float(method_params.get("relaxation", 1.0)),
            anderson_tol=float(method_params.get("anderson_tol", 1e-6)),
            solver=str(method_params.get("solver", solver)),
            solver_tol=float(method_params.get("solver_tol", tolerance)),
            solver_maxiter=int(method_params.get("solver_maxiter", 50)),
            regularization=float(
                method_params.get("regularization", common.get("regularization", 1e-6))
            ),
            convergence_tol=float(method_params.get("tolerance", tolerance)),
            plot_intermediate=False,
            plot_frequency=int(
                method_params.get("progress_every", common.get("progress_every", 100))
            ),
            save_param_trajectory=False,
            regularization_factor_gamma=float(
                method_params.get("regularization_factor_gamma", 1e-3)
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
            step_size=float(
                method_params.get("stepsize", common.get("stepsize", 1e-3))
            ),
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

    elif method in {"sgd", "adam"}:
        first_order_kwargs = extract_first_order_optimizer_kwargs(method_params)
        first_order_common_kwargs = {
            "parametric_model": model,
            "batch_size": n_samples,
            "test_data_set": z_samples,
            "potential": potential,
            "n_iterations": int(method_params.get("max_iterations", max_iterations)),
            "learning_rate": float(
                method_params.get("stepsize", common.get("stepsize", 1e-3))
            ),
            "convergence_tol": float(method_params.get("tolerance", tolerance)),
            "progress_every": int(
                method_params.get("progress_every", common.get("progress_every", 100))
            ),
            "verbose": bool(method_params.get("verbose", False)),
            "use_tqdm": bool(method_params.get("use_tqdm", False)),
            "progress_callback": progress_callback,
            "diagnostic_sample_size": diagnostic_sample_size,
            **first_order_kwargs,
        }
        if method == "sgd":
            history = run_sgd(**first_order_common_kwargs)
        else:
            history = run_adam(**first_order_common_kwargs)

        final_model = history["final_parametric_model"]
        energies = np.asarray(history["energy_history"], dtype=np.float64)
        riem_grad = np.asarray([], dtype=np.float64)
        euclid_grad = np.asarray(history["euclidean_grad_history"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported method: {method}")

    runtime_sec = time.perf_counter() - t0

    ckpt_relpath = Path("model_checkpoints") / run_id
    save_model_checkpoint(final_model, checkpoint_root / ckpt_relpath)

    return {
        "method": method,
        "method_params": method_params,
        "problem": problem,
        "problem_key": problem_key(problem),
        "problem_label": problem_label(problem),
        "problem_slug": problem_slug(problem),
        "functional_kind": problem["functional"]["kind"],
        "common": common,
        "model_config": model_cfg,
        "problem_meta": potential_meta,
        "energy_history": energies,
        "riemann_grad_history": riem_grad,
        "euclidean_grad_history": euclid_grad,
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
        run_problem = normalize_problem_entry(run["problem"])
        run_problem_key = run.get("problem_key", problem_key(run_problem))
        run_problem_label = run.get("problem_label", problem_label(run_problem))
        run_problem_slug = run.get("problem_slug", problem_slug(run_problem))
        grp.attrs["method"] = run["method"]
        grp.attrs["problem_json"] = json.dumps(run_problem, sort_keys=True)
        grp.attrs["problem_key"] = run_problem_key
        grp.attrs["problem_label"] = run_problem_label
        grp.attrs["problem_slug"] = run_problem_slug
        grp.attrs["functional_kind"] = run_problem["functional"]["kind"]
        grp.attrs["distribution"] = distribution_label(run_problem["distribution"])
        grp.attrs["method_params_json"] = json.dumps(
            run["method_params"], sort_keys=True
        )
        grp.attrs["model_config_json"] = json.dumps(run["model_config"], sort_keys=True)
        grp.attrs["problem_meta_json"] = json.dumps(
            run.get("problem_meta", {}), sort_keys=True
        )
        grp.attrs["common_json"] = json.dumps(run["common"], sort_keys=True)
        grp.attrs["runtime_sec"] = float(run["runtime_sec"])
        grp.attrs["model_ckpt_relpath"] = run["model_ckpt_relpath"]
        grp.create_dataset("energy_history", data=run["energy_history"])
        grp.create_dataset("riemann_grad_history", data=run["riemann_grad_history"])
        grp.create_dataset(
            "euclidean_grad_history",
            data=np.asarray(run.get("euclidean_grad_history", []), dtype=np.float64),
        )
        grp.create_dataset(
            "metrics_iteration_history",
            data=np.asarray(run.get("metrics_iteration_history", []), dtype=np.int64),
        )
        grp.create_dataset(
            "nll_history",
            data=np.asarray(run.get("nll_history", []), dtype=np.float64),
        )
        grp.create_dataset(
            "bits_dim_history",
            data=np.asarray(run.get("bits_dim_history", []), dtype=np.float64),
        )
        grp.create_dataset(
            "sliced_wasserstein_mean_history",
            data=np.asarray(
                run.get("sliced_wasserstein_mean_history", []), dtype=np.float64
            ),
        )
        grp.create_dataset(
            "sliced_wasserstein_std_history",
            data=np.asarray(
                run.get("sliced_wasserstein_std_history", []), dtype=np.float64
            ),
        )
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
    if "problems" not in config_for_plot:
        if "distributions" in config_for_plot and isinstance(
            config_for_plot["distributions"], list
        ):
            config_for_plot["problems"] = [
                {
                    "functional": {"kind": "KL"},
                    "distribution": normalize_distribution_entry(dist_cfg),
                }
                for dist_cfg in config_for_plot["distributions"]
            ]
        else:
            config_for_plot["problems"] = []
    return config_for_plot


def load_runs_from_h5(path: Path) -> list[dict[str, Any]]:
    def _load_problem_from_attrs(attrs: Any) -> dict[str, Any]:
        if "problem_json" in attrs:
            raw_problem_json = _attr_to_str(attrs.get("problem_json", "{}"), "{}")
            loaded_problem = json.loads(raw_problem_json)
            return normalize_problem_entry(loaded_problem)

        # Backward-compatible path for legacy .h5 entries.
        legacy_distribution = _attr_to_str(
            attrs.get("distribution", "unknown"), "unknown"
        )
        legacy_kind = normalize_functional_kind(
            _attr_to_str(attrs.get("functional_kind", "KL"), "KL")
        )
        return {
            "functional": {"kind": legacy_kind},
            "distribution": {"name": legacy_distribution},
        }

    with h5py.File(path, "r") as h5:
        out: list[dict[str, Any]] = []
        for run_id in sorted(h5["runs"].keys()):
            grp = h5["runs"][run_id]
            loaded_problem = _load_problem_from_attrs(grp.attrs)
            loaded_problem_key = _attr_to_str(
                grp.attrs.get("problem_key", problem_key(loaded_problem)),
                problem_key(loaded_problem),
            )
            loaded_problem_label = _attr_to_str(
                grp.attrs.get("problem_label", problem_label(loaded_problem)),
                problem_label(loaded_problem),
            )
            loaded_problem_slug = _attr_to_str(
                grp.attrs.get("problem_slug", problem_slug(loaded_problem)),
                problem_slug(loaded_problem),
            )
            out.append(
                {
                    "run_id": run_id,
                    "method": _attr_to_str(grp.attrs.get("method", ""), ""),
                    "problem": loaded_problem,
                    "problem_key": loaded_problem_key,
                    "problem_label": loaded_problem_label,
                    "problem_slug": loaded_problem_slug,
                    "functional_kind": loaded_problem["functional"]["kind"],
                    "method_params": json.loads(
                        _attr_to_str(grp.attrs.get("method_params_json", "{}"), "{}")
                    ),
                    "model_config": json.loads(
                        _attr_to_str(grp.attrs.get("model_config_json", "{}"), "{}")
                    ),
                    "problem_meta": json.loads(
                        _attr_to_str(grp.attrs.get("problem_meta_json", "{}"), "{}")
                    ),
                    "common": json.loads(
                        _attr_to_str(grp.attrs.get("common_json", "{}"), "{}")
                    ),
                    "runtime_sec": float(grp.attrs["runtime_sec"]),
                    "model_ckpt_relpath": _attr_to_str(
                        grp.attrs.get("model_ckpt_relpath", ""),
                        "",
                    ),
                    "energy_history": np.asarray(
                        grp["energy_history"][:], dtype=np.float64
                    ),
                    "riemann_grad_history": np.asarray(
                        grp["riemann_grad_history"][:], dtype=np.float64
                    ),
                    "euclidean_grad_history": (
                        np.asarray(grp["euclidean_grad_history"][:], dtype=np.float64)
                        if "euclidean_grad_history" in grp
                        else np.asarray([], dtype=np.float64)
                    ),
                    "metrics_iteration_history": (
                        np.asarray(grp["metrics_iteration_history"][:], dtype=np.int64)
                        if "metrics_iteration_history" in grp
                        else np.asarray([], dtype=np.int64)
                    ),
                    "nll_history": (
                        np.asarray(grp["nll_history"][:], dtype=np.float64)
                        if "nll_history" in grp
                        else np.asarray([], dtype=np.float64)
                    ),
                    "bits_dim_history": (
                        np.asarray(grp["bits_dim_history"][:], dtype=np.float64)
                        if "bits_dim_history" in grp
                        else np.asarray([], dtype=np.float64)
                    ),
                    "sliced_wasserstein_mean_history": (
                        np.asarray(
                            grp["sliced_wasserstein_mean_history"][:], dtype=np.float64
                        )
                        if "sliced_wasserstein_mean_history" in grp
                        else np.asarray([], dtype=np.float64)
                    ),
                    "sliced_wasserstein_std_history": (
                        np.asarray(
                            grp["sliced_wasserstein_std_history"][:], dtype=np.float64
                        )
                        if "sliced_wasserstein_std_history" in grp
                        else np.asarray([], dtype=np.float64)
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
    "momentum": r"\mu",
    "nesterov": r"\mathrm{Nes}",
    "b1": r"\beta_1",
    "b2": r"\beta_2",
    "eps": r"\epsilon",
    "eps_root": r"\epsilon_{\mathrm{root}}",
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
    grad_history: list[float],
    out_path: Path,
    title: str,
    grad_label: str = "Riemannian gradient norm",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if energy_history:
        e = np.asarray(energy_history, dtype=np.float64)
        axes[0].plot(e, color="#1f77b4", linewidth=1.8)
        if np.all(e > 0):
            axes[0].set_yscale("log")
    if grad_history:
        g = np.asarray(grad_history, dtype=np.float64)
        axes[1].plot(g, color="#d62728", linewidth=1.8)
        if np.all(g > 0):
            axes[1].set_yscale("log")

    axes[0].set_title("Energy")
    axes[1].set_title(grad_label)
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


def target_scatter_style(plotting_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": float(plotting_cfg.get("target_scatter_size", 6.0)),
        "alpha": float(plotting_cfg.get("target_scatter_alpha", 0.9)),
        "marker": str(plotting_cfg.get("target_scatter_marker", "*")),
        "facecolors": plotting_cfg.get("target_scatter_facecolors", "none"),
        "edgecolors": plotting_cfg.get("target_scatter_edgecolors", "#111111"),
        "linewidths": float(plotting_cfg.get("target_scatter_linewidths", 0.6)),
        "zorder": int(plotting_cfg.get("target_scatter_zorder", 5)),
        "label": str(plotting_cfg.get("target_scatter_label", "target")),
    }


def save_live_scatter_plot(
    scatter_samples: np.ndarray,
    distribution_cfg: dict[str, Any],
    plotting_cfg: dict[str, Any],
    out_path: Path,
    title: str,
    target_samples: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(plotting_cfg.get("scatter_figsize", [8, 8])))

    x = scatter_samples[:, 0]
    y = scatter_samples[:, 1]
    model_label = "model" if target_samples is not None else None
    ax.scatter(
        x,
        y,
        s=float(plotting_cfg.get("scatter_size", 20)),
        alpha=float(plotting_cfg.get("method_alpha", 0.55)),
        color=plotting_cfg.get("diagnostic_color", "#444444"),
        label=model_label,
    )

    target_samples_2d: np.ndarray | None = None
    if target_samples is not None:
        target = np.asarray(target_samples)
        if target.ndim == 2 and target.shape[1] >= 2:
            tstyle = target_scatter_style(plotting_cfg)
            target_label = tstyle.pop("label")
            ax.scatter(
                target[:, 0],
                target[:, 1],
                s=tstyle["size"],
                alpha=tstyle["alpha"],
                marker=tstyle["marker"],
                facecolors=tstyle["facecolors"],
                edgecolors=tstyle["edgecolors"],
                linewidths=tstyle["linewidths"],
                zorder=tstyle["zorder"],
                label=target_label,
            )
            target_samples_2d = target[:, :2]
            ax.legend(loc="upper right", frameon=True)

    try:
        plot_pot = build_plot_potential_2d(distribution_cfg)
        if plot_pot is not None:
            if target_samples_2d is not None:
                bounds_joint = np.concatenate(
                    [scatter_samples[:, :2], target_samples_2d], axis=0
                )
            else:
                bounds_joint = scatter_samples[:, :2]
            low = np.min(bounds_joint, axis=0)
            high = np.max(bounds_joint, axis=0)
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
    plotting_cfg: dict[str, Any] = config.get("plotting", {})
    problem_keys = sorted({run["problem_key"] for run in runs})
    guess_min = plotting_cfg.get("guess_min", False)
    for key in problem_keys:
        problem_runs = [run for run in runs if run["problem_key"] == key]
        if not problem_runs:
            continue
        current_problem_label = problem_runs[0]["problem_label"]
        current_problem_slug = problem_runs[0]["problem_slug"]

        method_runs_map: dict[str, list[dict[str, Any]]] = {}
        for run in problem_runs:
            method_runs_map.setdefault(run["method"], []).append(run)
        style_plan_map: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {
            method: build_method_style_plan(config, plotting_cfg, method, method_runs)
            for method, method_runs in method_runs_map.items()
        }

        fig, axes = plt.subplots(
            1, 2, figsize=tuple(plotting_cfg.get("figsize", [16, 6]))
        )
        euclidean_axis = None
        if any(
            np.asarray(run.get("euclidean_grad_history", []), dtype=np.float64).size > 0
            for run in problem_runs
        ):
            euclidean_axis = axes[1].twinx()

        if guess_min:
            e_min = min(jnp.min(run["energy_history"]) for run in problem_runs) - 1e-6
        else:
            e_min = 0.0

        used_labels: dict[str, int] = {}
        for run in problem_runs:
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
            riemann_history = np.asarray(run["riemann_grad_history"], dtype=np.float64)
            if riemann_history.size > 0:
                axes[1].plot(
                    riemann_history,
                    label=label,
                    **style,
                )
            if euclidean_axis is not None:
                euclidean_history = np.asarray(
                    run.get("euclidean_grad_history", []), dtype=np.float64
                )
                if euclidean_history.size > 0:
                    euclidean_axis.plot(
                        euclidean_history,
                        label=label,
                        **style,
                    )

        if guess_min:
            axes[0].set_yscale("log")
        riemann_histories = [
            np.asarray(run["riemann_grad_history"], dtype=np.float64)
            for run in problem_runs
            if np.asarray(run["riemann_grad_history"], dtype=np.float64).size > 0
        ]
        if riemann_histories and all(np.all(hist > 0) for hist in riemann_histories):
            axes[1].set_yscale("log")
        if euclidean_axis is not None:
            euclidean_histories = [
                np.asarray(run.get("euclidean_grad_history", []), dtype=np.float64)
                for run in problem_runs
                if np.asarray(
                    run.get("euclidean_grad_history", []), dtype=np.float64
                ).size
                > 0
            ]
            if euclidean_histories and all(
                np.all(hist > 0) for hist in euclidean_histories
            ):
                euclidean_axis.set_yscale("log")

        axes[0].set_title(f"Energy history ({current_problem_label})")
        if euclidean_axis is None:
            axes[1].set_title(f"Riemannian gradient history ({current_problem_label})")
        else:
            axes[1].set_title(f"Gradient history ({current_problem_label})")
        axes[0].set_xlabel("iteration")
        axes[1].set_xlabel("iteration")
        axes[0].set_ylabel("energy")
        axes[1].set_ylabel("riemann grad norm")
        if euclidean_axis is not None:
            euclidean_axis.set_ylabel("euclidean grad norm")
            euclidean_axis.grid(False)
        axes[0].grid(True)
        axes[1].grid(True)
        handles, labels = axes[1].get_legend_handles_labels()
        if euclidean_axis is not None:
            euclid_handles, euclid_labels = euclidean_axis.get_legend_handles_labels()
            handles.extend(euclid_handles)
            labels.extend(euclid_labels)

        dedup_handles: list[Any] = []
        dedup_labels: list[str] = []
        seen_labels: set[str] = set()
        for handle, label in zip(handles, labels, strict=False):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)

        ncol, bottom = dynamic_legend_layout(dedup_labels, fig.get_size_inches()[0])
        fig.legend(
            dedup_handles,
            dedup_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=True,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=bottom)
        out = (
            output_dir
            / f"run_all__{sanitize_component(current_problem_slug)}__all_methods__convergence.pdf"
        )
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
        shape_x=run["model_config"].get("shape_x", None),
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


def trace_method_for_metrics(problem: dict[str, Any]) -> str:
    return str(problem.get("functional", {}).get("trace_method", "hutchinson"))


def compute_nll_and_bits_dim(
    model: ParametricModel,
    X_test: np.ndarray,
    trace_method: str,
    chunk_size: int = 1024,
) -> tuple[float, float]:
    dim = int(X_test.shape[1])
    log_probs: list[np.ndarray] = []
    for start in range(0, int(X_test.shape[0]), chunk_size):
        X_chunk = jnp.asarray(X_test[start : start + chunk_size], dtype=jnp.float32)
        z_trajectory, timesteps = model.pull_back(X_chunk, history=True)
        z_trajectory = z_trajectory[:, ::-1, :]
        timesteps = timesteps[::-1]
        z0 = z_trajectory[:, 0, :]
        log_prob_init = -0.5 * (jnp.sum(z0**2, axis=-1) + dim * jnp.log(2.0 * jnp.pi))
        log_pdf_model = model.log_likelihood(
            t=timesteps,
            xt=z_trajectory,
            log_prob_init=log_prob_init,
            method=trace_method,
            log_trajectory=False,
        )
        log_probs.append(np.asarray(log_pdf_model, dtype=np.float64))

    all_log_probs = np.concatenate(log_probs, axis=0)
    nll = float(-np.mean(all_log_probs))
    bits_dim = float(nll / (np.log(2.0) * dim))
    return nll, bits_dim


def compute_sliced_wasserstein_distance(
    model_samples: jnp.ndarray,
    test_samples: jnp.ndarray,
    n_projections: int,
    rng: jax.Array,
) -> float:
    try:
        from ott.tools.sliced import sliced_wasserstein
    except ImportError as exc:
        raise ImportError(
            "Sliced Wasserstein requires ott-jax. Install it only when running "
            "dataset benchmarks that need SW metrics."
        ) from exc

    return float(
        sliced_wasserstein(
            model_samples,
            test_samples,
            n_proj=n_projections,
            rng=rng,
        )[0]
    )


def compute_sliced_wasserstein_stats(
    model: ParametricModel,
    X_test: np.ndarray,
    batch_size_sliced: int,
    sliced_n_draws: int,
    sliced_n_projections: int,
    seed: int,
) -> tuple[float, float]:
    dim = int(X_test.shape[1])
    key = jax.random.PRNGKey(seed)
    values: list[float] = []
    for _ in range(sliced_n_draws):
        key, z_key, test_key, sw_key = jax.random.split(key, 4)
        z = jax.random.normal(z_key, (batch_size_sliced, dim))
        model_samples = model(z)
        idx = jax.random.randint(
            test_key,
            (batch_size_sliced,),
            minval=0,
            maxval=int(X_test.shape[0]),
        )
        test_samples = jnp.asarray(X_test[np.asarray(idx)], dtype=jnp.float32)
        values.append(
            compute_sliced_wasserstein_distance(
                model_samples=model_samples,
                test_samples=test_samples,
                n_projections=sliced_n_projections,
                rng=sw_key,
            )
        )

    values_arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(values_arr)), float(np.std(values_arr))


def denormalize_dataset_samples(
    samples: np.ndarray,
    dataset_context: dict[str, Any],
) -> np.ndarray:
    return (
        np.asarray(samples, dtype=np.float32) * dataset_context["safe_s"]
        + dataset_context["mu"]
    )


def _image_batch_from_flat(
    samples: np.ndarray,
    dataset_context: dict[str, Any],
) -> np.ndarray:
    image_shape = dataset_context.get("image_shape")
    if image_shape is None:
        raise ValueError("image_shape is required for image comparison plots")
    display_samples = denormalize_dataset_samples(samples, dataset_context)
    return display_samples.reshape((-1, int(image_shape[0]), int(image_shape[1])))


def save_live_image_comparison_plot(
    model_samples: np.ndarray,
    test_samples: np.ndarray,
    dataset_context: dict[str, Any],
    out_path: Path,
    title: str,
    n_rows: int = 10,
    n_cols: int = 3,
    cmap: str = "binary",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = n_rows * n_cols
    model_images = _image_batch_from_flat(model_samples[:total], dataset_context)
    test_images = _image_batch_from_flat(test_samples[:total], dataset_context)

    fig, axes = plt.subplots(
        n_rows,
        2 * n_cols,
        figsize=(2.2 * 2 * n_cols, 2.2 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(n_rows, 2 * n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            model_ax = axes[i, j]
            test_ax = axes[i, j + n_cols]
            model_ax.matshow(model_images[i * n_cols + j], cmap=cmap)
            test_ax.matshow(test_images[i * n_cols + j], cmap=cmap)
            model_ax.set_axis_off()
            test_ax.set_axis_off()
            model_ax.set_aspect(1.0)
            test_ax.set_aspect(1.0)

    axes[0, max(0, n_cols // 2)].set_title("Model")
    axes[0, n_cols + max(0, n_cols // 2)].set_title("Test")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
    problem_keys = sorted({run["problem_key"] for run in runs})

    for key in problem_keys:
        problem_runs = [run for run in runs if run["problem_key"] == key]
        if not problem_runs:
            continue
        current_problem_label = problem_runs[0]["problem_label"]
        current_problem_slug = problem_runs[0]["problem_slug"]
        distribution_cfg = dict(problem_runs[0]["problem"]["distribution"])
        if is_image_distribution(distribution_cfg):
            dataset_context = prepare_dataset_context(distribution_cfg, common)
            n_rows = int(distribution_cfg.get("n_rows", 10))
            n_cols = int(distribution_cfg.get("n_cols", 3))
            n_grid = n_rows * n_cols
            for idx, run in enumerate(problem_runs):
                model = restore_model_from_run(run, checkpoint_root)
                run_dim = int(run["model_config"]["architecture"][0])
                model_samples = generate_samples(
                    model,
                    run_dim,
                    n_grid,
                    seed=1000 + idx,
                )
                test_samples = sample_dataset_rows(
                    dataset_context["X_test"],
                    n_grid,
                    seed=2000 + idx,
                )
                title = f"{current_problem_label}: {run['method']}"
                out = (
                    output_dir
                    / f"run_all__{sanitize_component(current_problem_slug)}__{sanitize_component(run['run_id'])}__samples.pdf"
                )
                save_live_image_comparison_plot(
                    model_samples=model_samples,
                    test_samples=test_samples,
                    dataset_context=dataset_context,
                    out_path=out,
                    title=title,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
            continue

        target_generator = None
        if problem_runs[0]["functional_kind"] in {"MMD", "CrossEntropy"}:
            target_generator = build_target_generator(distribution_cfg, common)

        method_runs_map: dict[str, list[dict[str, Any]]] = {}
        for run in problem_runs:
            method_runs_map.setdefault(run["method"], []).append(run)
        style_plan_map: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {
            method: build_method_style_plan(config, plotting_cfg, method, method_runs)
            for method, method_runs in method_runs_map.items()
        }
        warned_scatter_kwargs: set[tuple[str, str]] = set()

        gf_runs = [run for run in problem_runs if run["method"] == "gradient_flow"]
        if not gf_runs:
            print(
                f"[warn] skip scatter for {current_problem_label}: no gradient_flow run"
            )
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
        gf_dim = int(gf_baseline["model_config"].get("architecture", [dim])[0])
        gf_samples = generate_samples(gf_model, gf_dim, n_samples, seed=123)

        methods = sorted(
            {run["method"] for run in problem_runs if run["method"] != "gradient_flow"}
        )
        for method in methods:
            method_runs = [run for run in problem_runs if run["method"] == method]
            if not method_runs:
                continue

            fig, ax = plt.subplots(
                figsize=tuple(plotting_cfg.get("scatter_figsize", [8, 8]))
            )
            method_varying_keys, method_channel_maps = style_plan_map[method]
            used_labels: dict[str, int] = {}
            target_samples_2d: np.ndarray | None = None

            if target_generator is not None:
                target_samples = np.asarray(next(target_generator))
                if target_samples.ndim == 2 and target_samples.shape[1] >= 2:
                    target_samples_2d = target_samples[:, :2]
                    tstyle = target_scatter_style(plotting_cfg)
                    target_label = tstyle.pop("label")
                    ax.scatter(
                        target_samples[:, 0],
                        target_samples[:, 1],
                        s=tstyle["size"],
                        alpha=tstyle["alpha"],
                        marker=tstyle["marker"],
                        facecolors=tstyle["facecolors"],
                        edgecolors=tstyle["edgecolors"],
                        linewidths=tstyle["linewidths"],
                        zorder=tstyle["zorder"],
                        label=target_label,
                    )

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
                run_dim = int(run["model_config"].get("architecture", [dim])[0])
                samples = generate_samples(model, run_dim, n_samples, seed=1000 + idx)
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
                plot_pot = build_plot_potential_2d(distribution_cfg)
                if plot_pot is not None:
                    joint_parts = [gf_samples[:, :2]] + [
                        s[:, :2] for s in all_method_samples
                    ]
                    if target_samples_2d is not None:
                        joint_parts.append(target_samples_2d)
                    joint = np.concatenate(joint_parts, axis=0)
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
                print(
                    "[warn] failed contour plot for "
                    f"{current_problem_label}/{method}: {exc}"
                )

            ax.set_title(f"{current_problem_label}: gradient_flow vs {method}")
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
            method_slug = sanitize_component(method)
            out = (
                output_dir
                / f"run_all__{sanitize_component(current_problem_slug)}__{method_slug}__scatter.pdf"
            )
            fig.savefig(out)
            plt.close(fig)


def create_benchmark_session_dir(output_root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / ts
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def latest_h5(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None

    files = [
        p
        for p in output_root.glob("**/results*.h5")
        if p.name == "results.h5"
        or re.fullmatch(r"results__run_\d{4}\.h5", p.name) is not None
    ]
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def plots_dir_for_h5_path(h5_path: Path) -> Path:
    if h5_path.name == "results.h5":
        return h5_path.parent / "plots"
    match = re.fullmatch(r"results__(run_\d{4})\.h5", h5_path.name)
    if match is None:
        return h5_path.parent / "plots"
    return h5_path.parent / f"plots__{match.group(1)}"


def _get_live_plot_every(common: dict[str, Any]) -> int | None:
    if "live_plot_every" not in common:
        return None
    return max(1, int(common["live_plot_every"]))


def _prepare_planned_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    common = config["common_params"]
    configured_problems = problem_grid(config["problems"])
    base_seed = int(common.get("seed", 0))

    varying_keys_lookup: dict[tuple[str, str], list[str]] = {}
    for problem in configured_problems:
        key = problem_key(problem)
        for method, m_cfg in config["methods"].items():
            candidate_runs: list[dict[str, Any]] = []
            for params in method_grid(m_cfg):
                p = dict(params)
                p.setdefault("stepsize", common.get("stepsize", 1e-4))
                p.setdefault("max_iterations", common.get("max_iterations", 300))
                p.setdefault("tolerance", common.get("tolerance", 1e-4))
                candidate_runs.append({"method_params": p})
            varying_keys_lookup[(key, method)] = get_varying_keys_in_order(
                candidate_runs, list(m_cfg.keys())
            )

    planned_runs: list[dict[str, Any]] = []
    run_counter = 0
    for problem in configured_problems:
        key = problem_key(problem)
        run_problem_label = problem_label(problem)
        run_problem_slug = problem_slug(problem)
        for method, m_cfg in config["methods"].items():
            for params in method_grid(m_cfg):
                p = dict(params)
                p.setdefault("stepsize", common.get("stepsize", 1e-4))
                p.setdefault("max_iterations", common.get("max_iterations", 300))
                p.setdefault("tolerance", common.get("tolerance", 1e-4))
                method_slug = sanitize_component(method)
                run_id = f"{run_problem_slug}__{method_slug}__run_{run_counter:04d}"
                run_name = f"{run_problem_label} | {method} | {run_id}"
                run_method_label = build_method_label_latex(
                    method,
                    p,
                    varying_keys_lookup[(key, method)],
                )
                planned_runs.append(
                    {
                        "run_index": run_counter,
                        "run_id": run_id,
                        "run_name": run_name,
                        "run_title_base": run_problem_label,
                        "run_method_label": run_method_label,
                        "inner_total": int(
                            p.get("max_iterations", common.get("max_iterations", 300))
                        ),
                        "method": method,
                        "params": p,
                        "problem": problem,
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
    problem = normalize_problem_entry(planned_run["problem"])
    dist = dict(problem["distribution"])
    dataset_context = None
    if is_dataset_distribution(dist):
        dataset_context = prepare_dataset_context(dist, common)
    target_generator = None
    if problem["functional"]["kind"] in {"MMD", "CrossEntropy"}:
        target_generator = build_target_generator(
            problem["distribution"],
            common,
            dataset_context=dataset_context,
        )

    diagnostic_root = benchmark_dir / "diagnostic_plots"
    live_convergence_path = diagnostic_root / f"run_{run_id}__convergence.pdf"
    live_scatter_path = diagnostic_root / f"run_{run_id}__scatter.pdf"
    live_samples_path = diagnostic_root / f"run_{run_id}__samples.pdf"

    live_energy: list[float] = []
    live_grad: list[float] = []
    live_grad_label = "Riemannian gradient norm"
    latest_scatter_samples: np.ndarray | None = None
    latest_model: ParametricModel | None = None
    latest_params: Any | None = None
    warned_1d_scatter = False
    warned_missing_metric_model = False
    warned_nll = False
    warned_sw = False
    metrics_iteration_history: list[int] = []
    nll_history: list[float] = []
    bits_dim_history: list[float] = []
    sliced_wasserstein_mean_history: list[float] = []
    sliced_wasserstein_std_history: list[float] = []
    image_grid_size = 0
    if dataset_context is not None and bool(dataset_context["is_image"]):
        image_grid_size = int(dist.get("n_rows", 10)) * int(dist.get("n_cols", 3))

    def materialize_progress_model(
        model: ParametricModel | None,
        params: Any | None,
    ) -> ParametricModel | None:
        if model is None:
            return None
        if params is None:
            return model
        graphdef, _ = nnx.split(model)
        return nnx.merge(graphdef, params)

    def record_metrics(iteration: int) -> None:
        nonlocal warned_missing_metric_model, warned_nll, warned_sw
        if dataset_context is None:
            return

        eval_model = materialize_progress_model(latest_model, latest_params)
        if eval_model is None:
            if not warned_missing_metric_model:
                warnings.warn(
                    f"Skipping metrics for {run_id}: current model is unavailable.",
                    stacklevel=2,
                )
                warned_missing_metric_model = True
            return

        metrics_iteration_history.append(int(iteration))
        try:
            nll, bits_dim = compute_nll_and_bits_dim(
                eval_model,
                dataset_context["X_test"],
                trace_method=trace_method_for_metrics(problem),
            )
        except Exception as exc:
            if not warned_nll:
                warnings.warn(
                    f"Skipping NLL/bits-dim for {run_id}: {exc}",
                    stacklevel=2,
                )
                warned_nll = True
            nll = np.nan
            bits_dim = np.nan
        nll_history.append(float(nll))
        bits_dim_history.append(float(bits_dim))

        try:
            sw_mean, sw_std = compute_sliced_wasserstein_stats(
                eval_model,
                dataset_context["X_test"],
                batch_size_sliced=max(1, int(dist.get("batch_size_sliced", 256))),
                sliced_n_draws=max(1, int(dist.get("sliced_n_draws", 8))),
                sliced_n_projections=max(1, int(dist.get("sliced_n_projections", 128))),
                seed=int(planned_run["run_seed"]) + 100_000 + int(iteration),
            )
        except (ImportError, NotImplementedError) as exc:
            if not warned_sw:
                warnings.warn(
                    f"Skipping sliced Wasserstein for {run_id}: {exc}",
                    stacklevel=2,
                )
                warned_sw = True
            sw_mean = np.nan
            sw_std = np.nan
        sliced_wasserstein_mean_history.append(float(sw_mean))
        sliced_wasserstein_std_history.append(float(sw_std))

    def on_progress(info: dict[str, Any]) -> None:
        nonlocal latest_scatter_samples, latest_model, latest_params
        nonlocal warned_1d_scatter, live_grad_label

        iteration = int(info.get("iteration", 0)) + 1
        energy = float(info.get("energy", np.nan))
        if "euclidean_grad_norm" in info:
            grad = float(info.get("euclidean_grad_norm", np.nan))
            live_grad_label = "Euclidean gradient norm"
        else:
            grad = float(info.get("riemann_grad_norm", np.nan))
            live_grad_label = "Riemannian gradient norm"

        if np.isfinite(energy):
            live_energy.append(energy)
        if np.isfinite(grad):
            live_grad.append(grad)

        scatter_samples = info.get("scatter_samples", None)
        if scatter_samples is not None:
            latest_scatter_samples = np.asarray(scatter_samples)
        if "model" in info:
            latest_model = info["model"]
        if "params" in info:
            latest_params = info["params"]

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
        record_metrics(iteration)
        save_live_convergence_plot(
            live_energy,
            live_grad,
            live_convergence_path,
            title=diag_title,
            grad_label=live_grad_label,
        )
        effective_dim = (
            int(dataset_context["dim"])
            if dataset_context is not None
            else int(common["dimension"])
        )
        if dataset_context is not None and bool(dataset_context["is_image"]):
            if latest_scatter_samples is not None:
                test_samples = sample_dataset_rows(
                    dataset_context["X_test"],
                    image_grid_size,
                    seed=int(planned_run["run_seed"]) + int(iteration),
                )
                save_live_image_comparison_plot(
                    latest_scatter_samples,
                    test_samples,
                    dataset_context,
                    live_samples_path,
                    title=diag_title,
                    n_rows=int(dist.get("n_rows", 10)),
                    n_cols=int(dist.get("n_cols", 3)),
                )
        elif effective_dim >= 2:
            if latest_scatter_samples is not None:
                target_samples = None
                if target_generator is not None:
                    target_samples = np.asarray(next(target_generator))
                save_live_scatter_plot(
                    latest_scatter_samples,
                    dist,
                    plotting_cfg,
                    live_scatter_path,
                    title=diag_title,
                    target_samples=target_samples,
                )
        elif not warned_1d_scatter:
            if print_warnings:
                tqdm.write(
                    f"[warn] skip diagnostic scatter for {run_id}: dimension is 1"
                )
            warned_1d_scatter = True

    diagnostic_sample_size = max(int(plot_n_samples), image_grid_size)
    run = run_single(
        method=method,
        method_params=params,
        problem_cfg=problem,
        common=common,
        run_seed=int(planned_run["run_seed"]),
        checkpoint_root=benchmark_dir,
        run_id=run_id,
        progress_callback=on_progress,
        diagnostic_sample_size=diagnostic_sample_size,
        dataset_context=dataset_context,
    )

    final_iter = max(1, len(run["energy_history"]))
    if live_plot_every is not None:
        final_title = (
            f"{planned_run['run_title_base']} | iter={final_iter}"
            + "\n"
            + planned_run["run_method_label"]
        )
        final_euclidean_grad = np.asarray(
            run.get("euclidean_grad_history", []), dtype=np.float64
        )
        if final_euclidean_grad.size > 0:
            final_grad = list(final_euclidean_grad)
            final_grad_label = "Euclidean gradient norm"
        else:
            final_grad = list(np.asarray(run["riemann_grad_history"], dtype=np.float64))
            final_grad_label = "Riemannian gradient norm"
        record_metrics(final_iter)
        save_live_convergence_plot(
            list(np.asarray(run["energy_history"], dtype=np.float64)),
            final_grad,
            live_convergence_path,
            title=final_title,
            grad_label=final_grad_label,
        )
        final_effective_dim = (
            int(dataset_context["dim"])
            if dataset_context is not None
            else int(common["dimension"])
        )
        if (
            dataset_context is not None
            and bool(dataset_context["is_image"])
            and latest_scatter_samples is not None
        ):
            test_samples = sample_dataset_rows(
                dataset_context["X_test"],
                image_grid_size,
                seed=int(planned_run["run_seed"]) + int(final_iter),
            )
            save_live_image_comparison_plot(
                latest_scatter_samples,
                test_samples,
                dataset_context,
                live_samples_path,
                title=final_title,
                n_rows=int(dist.get("n_rows", 10)),
                n_cols=int(dist.get("n_cols", 3)),
            )
        elif final_effective_dim >= 2 and latest_scatter_samples is not None:
            target_samples = None
            if target_generator is not None:
                target_samples = np.asarray(next(target_generator))
            save_live_scatter_plot(
                latest_scatter_samples,
                dist,
                plotting_cfg,
                live_scatter_path,
                title=final_title,
                target_samples=target_samples,
            )

    run["metrics_iteration_history"] = np.asarray(
        metrics_iteration_history, dtype=np.int64
    )
    run["nll_history"] = np.asarray(nll_history, dtype=np.float64)
    run["bits_dim_history"] = np.asarray(bits_dim_history, dtype=np.float64)
    run["sliced_wasserstein_mean_history"] = np.asarray(
        sliced_wasserstein_mean_history, dtype=np.float64
    )
    run["sliced_wasserstein_std_history"] = np.asarray(
        sliced_wasserstein_std_history, dtype=np.float64
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


def run_all(
    config: dict[str, Any],
    benchmark_dir: Path,
    output_h5: Path,
    run_id: int | None = None,
) -> dict[str, int]:
    common = config["common_params"]
    plotting_cfg = config.get("plotting", {})
    ckpt_root = benchmark_dir / "model_checkpoints"
    diagnostic_root = benchmark_dir / "diagnostic_plots"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    live_plot_every = _get_live_plot_every(common)
    plot_n_samples = int(common["plot_n_samples"])
    planned_runs = _prepare_planned_runs(config)
    total_runs = len(planned_runs)
    initialize_h5(output_h5, config)

    success_count = 0
    failed_count = 0
    fail_fast = bool(config.get("fail_fast", False))

    if run_id is not None:
        selected_run = next(
            (run for run in planned_runs if int(run["run_index"]) == int(run_id)),
            None,
        )
        if selected_run is None:
            if not planned_runs:
                raise ValueError("No planned runs found in benchmark configuration")
            min_run_id = min(int(run["run_index"]) for run in planned_runs)
            max_run_id = max(int(run["run_index"]) for run in planned_runs)
            raise ValueError(
                f"Unknown run_id={run_id}. Valid run_index range: "
                f"[{min_run_id}, {max_run_id}]"
            )

        try:
            run, _ = _execute_planned_run(
                planned_run=selected_run,
                common=common,
                plotting_cfg=plotting_cfg,
                benchmark_dir=benchmark_dir,
                live_plot_every=live_plot_every,
                plot_n_samples=plot_n_samples,
            )
            append_run_to_h5(output_h5, str(selected_run["run_id"]), run)
            success_count = 1
        except Exception:
            failed_count = 1
            if fail_fast:
                raise
        finally:
            jax.clear_caches()
            gc.collect()
        return {"success": success_count, "failed": failed_count}

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
                    if "euclidean_grad_norm" in info:
                        grad = float(info.get("euclidean_grad_norm", np.nan))
                    else:
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
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run only planned run with this numeric run_index",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output subdirectory name under output_root (used instead of timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only and args.run_id is not None:
        raise ValueError("--plot-only and --run-id cannot be used together")

    config = load_config(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    run_name_sanitized = None
    if args.run_name is not None:
        run_name_sanitized = sanitize_component(args.run_name)
        if not run_name_sanitized:
            raise ValueError("--run-name must contain at least one valid character")

    selected_output_root = output_root
    if run_name_sanitized is not None:
        selected_output_root = output_root / run_name_sanitized
        selected_output_root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        h5_path = latest_h5(selected_output_root)
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
        plots_dir = plots_dir_for_h5_path(h5_path)
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_convergence_plots(plot_config, loaded_runs, plots_dir)
        save_scatter_plots(plot_config, loaded_runs, h5_path.parent, plots_dir)
        return

    if run_name_sanitized is not None:
        benchmark_dir = selected_output_root
        benchmark_dir.mkdir(parents=True, exist_ok=True)
    else:
        benchmark_dir = create_benchmark_session_dir(output_root)

    run_suffix = None
    if args.run_id is not None:
        run_suffix = f"run_{int(args.run_id):04d}"

    if run_suffix is None:
        output_h5 = benchmark_dir / "results.h5"
        plots_dir = benchmark_dir / "plots"
    else:
        output_h5 = benchmark_dir / f"results__{run_suffix}.h5"
        plots_dir = benchmark_dir / f"plots__{run_suffix}"

    run_all(config, benchmark_dir, output_h5=output_h5, run_id=args.run_id)
    experiment_config_h5 = load_experiment_config_from_h5(output_h5)
    runs_for_scatter = load_runs_from_h5(output_h5)
    plot_config = make_plot_config(config, experiment_config_h5)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_convergence_plots(plot_config, runs_for_scatter, plots_dir)
    save_scatter_plots(plot_config, runs_for_scatter, benchmark_dir, plots_dir)


def load_miniboone(data_root: str | Path | None = None):
    raise NotImplementedError(
        "MiniBoone loading is not implemented yet. Return "
        "(X_train, X_test, y_train, y_test), where X arrays are numpy-compatible "
        "and the first axis is the sample axis."
    )


def load_data(
    target: Literal["mnist", "fashion", "miniboone"],
    data_root: str | Path | None = None,
):
    match str(target).lower():
        case "fashion":
            import mnist_reader

            root = str(data_root) if data_root is not None else "data/fashion"
            X_train, y_train = mnist_reader.load_mnist(root, kind="train")
            X_test, y_test = mnist_reader.load_mnist(root, kind="t10k")
        case "mnist":
            # Load data from https://www.openml.org/d/554
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split

            fetch_kwargs = {}
            if data_root is not None:
                fetch_kwargs["data_home"] = str(data_root)
            X, y = fetch_openml(
                "mnist_784",
                version=1,
                return_X_y=True,
                as_frame=False,
                **fetch_kwargs,
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)
        case "miniboone":
            X_train, X_test, y_train, y_test = load_miniboone(data_root=data_root)

        case _:
            raise ValueError(f"dataset {target} not supported")

    return X_train, X_test, y_train, y_test


def plot_batch(
    X: np.array,
    nrow: int,
    ncol: int,
    cmap="binary",
):
    X = np.asarray(X)
    if X.ndim == 2:
        if X.shape[1] != 28 * 28:
            raise ValueError("flat image batches must have dimension 28*28")
        X_img = X[: nrow * ncol].reshape((nrow, ncol, 28, 28))
    elif X.ndim == 3:
        X_img = X[: nrow * ncol, :, :].reshape((nrow, ncol, 28, 28))
    else:
        raise ValueError("plot_batch expects flat or image-shaped batches")
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True)
    axs = np.asarray(axs).reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            ax = axs[i, j]
            ax.matshow(X_img[i, j, :, :], cmap=cmap)
            ax.set_aspect(1.0)
            ax.set_axis_off()

    return fig


if __name__ == "__main__":
    main()

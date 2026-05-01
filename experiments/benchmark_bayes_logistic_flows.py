#  python3 experiments/benchmark_bayes_logistic_flows.py   --datasets splice waveform twonorm ringnorm german image diabetis banana   --seeds 0 1 2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np
from flax import nnx

from datasets.bayesian_dataset import get_train_test_datasets
from experiments.bayesian_logistic_benchmark import (
    build_bayesian_logistic_setup,
    build_potential_for_bayes_logistic,
    classification_accuracy_from_particles,
    predictive_log_likelihood_from_particles,
)
from flows.anderson_acceleration import anderson_method
from flows.gradient_flow import run_gradient_flow
from geometry.G_matrix import G_matrix
from parametric_model.parametric_model import ParametricModel


DEFAULT_DATASETS = [
    "covtype",
    "splice",
    "waveform",
    "twonorm",
    "ringnorm",
    "german",
    "image",
    "diabetis",
    "banana",
]


@dataclass
class BenchmarkConfig:
    datasets: list[str]
    seeds: list[int]
    data_path: str
    results_root: str
    test_ratio: float
    standardize: bool
    alpha_prior_shape: float
    alpha_prior_rate: float
    linear_coeff: float
    ref_density: str
    parametric_map: str
    activation_fn: str
    width_layers: int
    num_layers: int
    scale_factor: float
    rhs_model: str
    time_dependent: bool
    ode_solver: str
    dt0: float
    gf_iterations: int
    gf_step_size: float
    gf_batch_size: int
    gf_solver: str
    gf_solver_tol: float
    gf_convergence_tol: float
    gf_regularization: float
    gf_plot_frequency: int
    gf_solver_maxiter: int
    aa_iterations: int
    aa_step_size: float
    aa_batch_size: int
    aa_memory_size: int
    aa_relaxation: float
    aa_anderson_tol: float
    aa_solver: str
    aa_solver_tol: float
    aa_solver_maxiter: int
    aa_regularization: float
    aa_l2_reg_gamma: float
    aa_reg_method: str
    aa_convergence_tol: float
    eval_num_particles: int


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if len(rows) == 0:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_last(values: list[Any], default: float = float("nan")) -> float:
    if values is None or len(values) == 0:
        return default
    return float(values[-1])


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


WORK_ITER_KEYS = [
    "grad_calls_cum",
    "solve_calls_cum",
    "direct_mvp_calls_cum",
    "grad_mvp_equiv_cum",
    "solve_mvp_equiv_cum",
    "direct_mvp_equiv_cum",
    "opt_mvp_equiv_cum",
    "grad_sample_work_cum",
    "solve_sample_work_cum",
    "direct_mvp_sample_work_cum",
    "opt_mvp_equiv_sample_work_cum",
]


def _method_run_id(dataset: str, seed: int, method: str) -> str:
    return f"{dataset}-s{seed}-{method}"


def _anderson_method_name(regularization_method: str) -> str:
    if regularization_method == "l2":
        return "AA-L2"
    if regularization_method == "adaptive":
        return "AA-adaptive"
    raise ValueError(f"Unsupported Anderson gamma regularization: {regularization_method}")


def _build_model(config: BenchmarkConfig, problem_dim: int, seed: int) -> ParametricModel:
    return ParametricModel(
        parametric_map=config.parametric_map,
        architecture=[problem_dim, config.num_layers, config.width_layers],
        activation_fn=config.activation_fn,
        key=jax.random.PRNGKey(seed),
        ref_density=config.ref_density,
        scale_factor=config.scale_factor,
        rhs_model=config.rhs_model,
        time_dependent=config.time_dependent,
        solver=config.ode_solver,
        dt0=config.dt0,
    )


def run_benchmark(config: BenchmarkConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.results_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []
    iter_records: list[dict] = []
    big_solve_records: list[dict] = []
    aa_method_name = _anderson_method_name(config.aa_reg_method)
    methods = [
        ("GF", None),
        (aa_method_name, config.aa_reg_method),
    ]
    challenger_methods = [aa_method_name]

    for dataset_name in config.datasets:
        for seed in config.seeds:
            print(f"\n=== Dataset={dataset_name} Seed={seed} ===")
            dataset = get_train_test_datasets(
                name=dataset_name,
                data_path=config.data_path,
                test_ratio=config.test_ratio,
                seed=seed,
                standardize=config.standardize,
            )
            setup = build_bayesian_logistic_setup(
                dataset=dataset,
                alpha_prior_shape=config.alpha_prior_shape,
                alpha_prior_rate=config.alpha_prior_rate,
            )
            potential = build_potential_for_bayes_logistic(
                setup, linear_coeff=config.linear_coeff
            )
            problem_dim = setup.X_train_jax.shape[1] + 1

            # Fair initialization across methods: one canonical init_params per dataset/seed.
            init_model = _build_model(config, problem_dim=problem_dim, seed=seed)
            _, init_params = nnx.split(init_model)

            # Shared evaluation reference samples for methods expecting test_data_set.
            key_eval = jax.random.PRNGKey(seed + 10_000)
            z_eval = init_model.sampler(key_eval, config.eval_num_particles)

            for method_name, aa_gamma_method in methods:
                print(f"Running {method_name} ...")
                model = _build_model(
                    config,
                    problem_dim=problem_dim,
                    seed=seed + 1234,
                )
                graphdef, _ = nnx.split(model)
                model = nnx.merge(graphdef, init_params)
                g_mat = G_matrix(model)
                run_id = _method_run_id(dataset_name, seed, method_name)
                method_iter_rows: list[dict] = []

                def progress_callback(payload: dict[str, Any]) -> None:
                    row = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method_name,
                        "outer_iter": payload.get("iteration"),
                        "energy": _as_float(payload.get("energy", np.nan)),
                        "residual_norm": _as_float(
                            payload.get("riemann_grad_norm", np.nan)
                        ),
                    }
                    for key in WORK_ITER_KEYS:
                        if key in payload:
                            row[key] = _as_float(payload[key])
                    method_iter_rows.append(row)

                if method_name == "GF":
                    results = run_gradient_flow(
                        parametric_model=model,
                        z_samples=z_eval,
                        G_mat=g_mat,
                        potential=potential,
                        N_samples=config.gf_batch_size,
                        h=config.gf_step_size,
                        solver=config.gf_solver,
                        max_iterations=config.gf_iterations,
                        solver_maxiter=config.gf_solver_maxiter,
                        tolerance=config.gf_solver_tol,
                        regularization=config.gf_regularization,
                        progress_every=config.gf_plot_frequency,
                        plot_intermediate=False,
                        verbose=False,
                        use_tqdm=False,
                        progress_callback=progress_callback,
                    )
                    final_model = results["final_parametric_model"]
                    inst = results.get("instrumentation", {})
                    work = results.get("work_summary", {})
                    final_energy = _safe_last(results.get("energy_history", []))
                    final_residual = _safe_last(
                        results.get("riemann_grad_norm_history", [])
                    )
                    final_iter = int(results["convergence_info"]["iterations"])
                else:
                    graphdef, _ = nnx.split(model)
                    final_params, history = anderson_method(
                        parametric_model=model,
                        batch_size=config.aa_batch_size,
                        test_data_set=z_eval,
                        G_mat=g_mat,
                        potential=potential,
                        initial_params=init_params,
                        n_iterations=config.aa_iterations,
                        step_size=config.aa_step_size,
                        memory_size=config.aa_memory_size,
                        relaxation=config.aa_relaxation,
                        anderson_tol=config.aa_anderson_tol,
                        solver=config.aa_solver,
                        solver_maxiter=config.aa_solver_maxiter,
                        solver_tol=config.aa_solver_tol,
                        regularization=config.aa_regularization,
                        regularization_factor_gamma=config.aa_l2_reg_gamma,
                        regularization_method_gamma=aa_gamma_method,
                        convergence_tol=config.aa_convergence_tol,
                        ensure_descent=True,
                        verbose=False,
                        progress_callback=progress_callback,
                    )
                    final_model = nnx.merge(graphdef, final_params)
                    inst = {}
                    work = history.get("work_summary", {})
                    final_energy = _safe_last(history.get("energies", []))
                    final_residual = _safe_last(history.get("riemann_grad_history", []))
                    final_iter = int(history.get("final_iteration", -1))

                key_particles = jax.random.PRNGKey(seed + 20_000)
                z_particles = model.sampler(key_particles, config.eval_num_particles)
                theta_particles = final_model(z_particles)
                test_acc = _as_float(
                    classification_accuracy_from_particles(
                        theta_particles, setup.X_test_jax, setup.y_test_jax
                    )
                )
                test_loglik = _as_float(
                    predictive_log_likelihood_from_particles(
                        theta_particles, setup.X_test_jax, setup.y_test_jax
                    )
                )

                inst_big_rows = inst.get("big_solve_records", [])
                iter_records.extend(method_iter_rows)
                for row in inst_big_rows:
                    big_solve_records.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            **row,
                        }
                    )

                rec = {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": method_name,
                    "final_iter": final_iter,
                    "final_energy": _as_float(final_energy),
                    "final_residual_norm": _as_float(final_residual),
                    "final_test_accuracy": test_acc,
                    "final_test_log_likelihood": test_loglik,
                    "lsc_big": int(inst.get("lsc_big", 0)),
                    "ksi_big": int(inst.get("ksi_big", 0)),
                    "ksi_big_mean": _as_float(inst.get("ksi_big_mean", np.nan)),
                    "time_big_solve_sec": _as_float(inst.get("time_big_solve_sec", np.nan)),
                    "elapsed_total_sec": _as_float(work.get("elapsed_total_sec", np.nan)),
                    "final_opt_mvp_equiv": _as_float(
                        work.get("opt_mvp_equiv_cum", np.nan)
                    ),
                    "final_opt_mvp_equiv_sample_work": _as_float(
                        work.get("opt_mvp_equiv_sample_work_cum", np.nan)
                    ),
                    "final_grad_calls": _as_float(work.get("grad_calls_cum", np.nan)),
                    "final_solve_calls": _as_float(work.get("solve_calls_cum", np.nan)),
                    "final_direct_mvp_calls": _as_float(
                        work.get("direct_mvp_calls_cum", np.nan)
                    ),
                }
                run_records.append(rec)

    # Summary by dataset-method with median and IQR.
    summary_rows: list[dict] = []
    by_key: dict[tuple[str, str], list[dict]] = {}
    for rec in run_records:
        by_key.setdefault((rec["dataset"], rec["method"]), []).append(rec)
    metrics = [
        "final_energy",
        "final_residual_norm",
        "final_test_accuracy",
        "final_test_log_likelihood",
        "lsc_big",
        "ksi_big",
        "final_opt_mvp_equiv",
        "final_opt_mvp_equiv_sample_work",
        "final_grad_calls",
        "final_solve_calls",
        "final_direct_mvp_calls",
        "time_big_solve_sec",
        "elapsed_total_sec",
    ]
    for (dataset, method), items in sorted(by_key.items()):
        row = {"dataset": dataset, "method": method, "n_runs": len(items)}
        for m in metrics:
            vals = np.asarray([_as_float(it[m]) for it in items], dtype=np.float64)
            row[f"{m}_median"] = float(np.nanmedian(vals))
            q1 = float(np.nanquantile(vals, 0.25))
            q3 = float(np.nanquantile(vals, 0.75))
            row[f"{m}_iqr"] = q3 - q1
        summary_rows.append(row)

    # Win-rate vs GF (by seed,dataset pair).
    win_rows: list[dict] = []
    pair_index: dict[tuple[str, int], dict[str, dict]] = {}
    for rec in run_records:
        pair_index.setdefault((rec["dataset"], rec["seed"]), {})[rec["method"]] = rec
    for (dataset, seed), methods_map in sorted(pair_index.items()):
        if "GF" not in methods_map:
            continue
        gf = methods_map["GF"]
        for challenger in challenger_methods:
            if challenger not in methods_map:
                continue
            ch = methods_map[challenger]
            win_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "challenger": challenger,
                    "wins_energy_vs_gf": int(
                        _as_float(ch["final_energy"]) < _as_float(gf["final_energy"])
                    ),
                    "wins_accuracy_vs_gf": int(
                        _as_float(ch["final_test_accuracy"])
                        > _as_float(gf["final_test_accuracy"])
                    ),
                    "wins_loglik_vs_gf": int(
                        _as_float(ch["final_test_log_likelihood"])
                        > _as_float(gf["final_test_log_likelihood"])
                    ),
                    "wins_ksi_vs_gf": int(_as_float(ch["ksi_big"]) < _as_float(gf["ksi_big"])),
                    "wins_work_vs_gf": int(
                        _as_float(ch["final_opt_mvp_equiv_sample_work"])
                        < _as_float(gf["final_opt_mvp_equiv_sample_work"])
                    ),
                }
            )

    win_summary_rows: list[dict] = []
    for challenger in challenger_methods:
        rows = [r for r in win_rows if r["challenger"] == challenger]
        if len(rows) == 0:
            continue
        win_summary_rows.append(
            {
                "challenger": challenger,
                "n_pairs": len(rows),
                "energy_win_rate": float(np.mean([r["wins_energy_vs_gf"] for r in rows])),
                "accuracy_win_rate": float(
                    np.mean([r["wins_accuracy_vs_gf"] for r in rows])
                ),
                "loglik_win_rate": float(
                    np.mean([r["wins_loglik_vs_gf"] for r in rows])
                ),
                "ksi_win_rate": float(np.mean([r["wins_ksi_vs_gf"] for r in rows])),
                "work_win_rate": float(np.mean([r["wins_work_vs_gf"] for r in rows])),
            }
        )

    _write_jsonl(out_dir / "runs.jsonl", run_records)
    _write_csv(out_dir / "runs.csv", run_records)
    _write_csv(out_dir / "iter_metrics.csv", iter_records)
    _write_jsonl(out_dir / "big_solves.jsonl", big_solve_records)
    _write_csv(out_dir / "summary_by_dataset.csv", summary_rows)
    _write_csv(out_dir / "win_pairs.csv", win_rows)
    _write_csv(out_dir / "win_summary.csv", win_summary_rows)
    with (out_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\nSaved benchmark artifacts to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GF vs AA-l2 vs AA-adaptive for Bayesian logistic regression."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results/bayes_logistic_benchmark",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--alpha-prior-shape", type=float, default=1.0)
    parser.add_argument("--alpha-prior-rate", type=float, default=0.01)
    parser.add_argument("--linear-coeff", type=float, default=1.0)

    parser.add_argument("--parametric-map", type=str, default="node")
    parser.add_argument("--activation-fn", type=str, default="tanh")
    parser.add_argument("--width-layers", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--scale-factor", type=float, default=0.1)
    parser.add_argument("--ref-density", type=str, default="gaussian")
    parser.add_argument("--rhs-model", type=str, default="mlp")
    parser.add_argument("--time-dependent", action="store_true", default=True)
    parser.add_argument("--ode-solver", type=str, default="euler")
    parser.add_argument("--dt0", type=float, default=0.1)

    parser.add_argument("--gf-iterations", type=int, default=50)
    parser.add_argument("--gf-step-size", type=float, default=5e-3)
    parser.add_argument("--gf-batch-size", type=int, default=5000)
    parser.add_argument("--gf-solver", type=str, default="cg")
    parser.add_argument("--gf-solver-tol", type=float, default=1e-4)
    parser.add_argument("--gf-convergence-tol", type=float, default=1e-4)
    parser.add_argument("--gf-regularization", type=float, default=1e-2)
    parser.add_argument("--gf-plot-frequency", type=int, default=100)
    parser.add_argument("--gf-solver_maxiter", type=int, default=10)

    parser.add_argument("--aa-iterations", type=int, default=50)
    parser.add_argument("--aa-step-size", type=float, default=5e-3)
    parser.add_argument("--aa-batch-size", type=int, default=5000)
    parser.add_argument("--aa-memory-size", type=int, default=8)
    parser.add_argument("--aa-relaxation", type=float, default=1.5)
    parser.add_argument("--aa-anderson-tol", type=float, default=1e-4)
    parser.add_argument("--aa-solver", type=str, default="cg")
    parser.add_argument("--aa-solver-tol", type=float, default=1e-4)
    parser.add_argument("--aa-regularization", type=float, default=1e-2)
    parser.add_argument("--aa-l2-reg-gamma", type=float, default=1e-1)
    parser.add_argument("--aa-reg-method",type=str,default = 'l2',choices=['l2','adaptive'])
    parser.add_argument("--aa-convergence-tol", type=float, default=1e-4)
    parser.add_argument("--aa-solver_maxiter", type=int, default=10)

    parser.add_argument("--eval-num-particles", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        datasets=args.datasets,
        seeds=args.seeds,
        data_path=args.data_path,
        results_root=args.results_root,
        test_ratio=args.test_ratio,
        standardize=args.standardize,
        alpha_prior_shape=args.alpha_prior_shape,
        alpha_prior_rate=args.alpha_prior_rate,
        linear_coeff=args.linear_coeff,
        ref_density=args.ref_density,
        parametric_map=args.parametric_map,
        activation_fn=args.activation_fn,
        width_layers=args.width_layers,
        num_layers=args.num_layers,
        scale_factor=args.scale_factor,
        rhs_model=args.rhs_model,
        time_dependent=args.time_dependent,
        ode_solver=args.ode_solver,
        dt0=args.dt0,
        gf_iterations=args.gf_iterations,
        gf_step_size=args.gf_step_size,
        gf_batch_size=args.gf_batch_size,
        gf_solver=args.gf_solver,
        gf_solver_tol=args.gf_solver_tol,
        gf_convergence_tol=args.gf_convergence_tol,
        gf_regularization=args.gf_regularization,
        gf_plot_frequency=args.gf_plot_frequency,
        gf_solver_maxiter=args.gf_solver_maxiter,
        aa_iterations=args.aa_iterations,
        aa_step_size=args.aa_step_size,
        aa_batch_size=args.aa_batch_size,
        aa_memory_size=args.aa_memory_size,
        aa_relaxation=args.aa_relaxation,
        aa_anderson_tol=args.aa_anderson_tol,
        aa_solver=args.aa_solver,
        aa_solver_tol=args.aa_solver_tol,
        aa_solver_maxiter=args.aa_solver_maxiter,
        aa_regularization=args.aa_regularization,
        aa_l2_reg_gamma=args.aa_l2_reg_gamma,
        aa_reg_method=args.aa_reg_method,
        aa_convergence_tol=args.aa_convergence_tol,
        eval_num_particles=args.eval_num_particles,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()

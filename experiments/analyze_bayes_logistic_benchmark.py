#python3 experiments/benchmark_bayes_logistic_flows.py   --datasets splice waveform twonorm ringnorm german image diabetis banana   --seeds 0 1 2

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X_LABEL_CHARGED_WORK = "Charged MVP-Equivalent Sample Work (cumulative)"


REQ_RUNS_COLS = {
    "run_id",
    "dataset",
    "seed",
    "method",
    "final_energy",
    "final_test_accuracy",
}

OPTIONAL_RUN_COLS = {
    "final_residual_norm",
    "final_test_log_likelihood",
    "elapsed_total_sec",
    "final_opt_mvp_equiv",
    "final_opt_mvp_equiv_sample_work",
    "final_grad_calls",
    "final_solve_calls",
    "final_direct_mvp_calls",
}

NUMERIC_RUN_COLS = [
    "final_energy",
    "final_residual_norm",
    "final_test_accuracy",
    "final_test_log_likelihood",
    "elapsed_total_sec",
    "final_opt_mvp_equiv",
    "final_opt_mvp_equiv_sample_work",
    "final_grad_calls",
    "final_solve_calls",
    "final_direct_mvp_calls",
]


def _iqr(series: pd.Series) -> float:
    q = series.quantile([0.25, 0.75])
    return float(q.iloc[1] - q.iloc[0])


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_runs(results_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    runs_csv = results_dir / "runs.csv"
    runs_jsonl = results_dir / "runs.jsonl"
    if runs_csv.exists():
        df = pd.read_csv(runs_csv)
    elif runs_jsonl.exists():
        rows = [json.loads(line) for line in runs_jsonl.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError(
            f"Could not find runs.csv or runs.jsonl in {results_dir}"
        )

    missing = REQ_RUNS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in runs data: {sorted(missing)}")

    warnings = []
    for c in OPTIONAL_RUN_COLS:
        if c not in df.columns:
            df[c] = np.nan
            warnings.append(f"Optional column missing in runs data; filled NaN: {c}")

    for c in NUMERIC_RUN_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, warnings


def _load_iter(results_dir: Path) -> pd.DataFrame | None:
    p = results_dir / "iter_metrics.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    if len(df) == 0:
        return None
    return df


def _has_positive_finite(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return bool(np.isfinite(values).any() and np.nanmax(values) > 0.0)


def _prepare_work_axis(
    runs_df: pd.DataFrame, iter_df: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None, str | None, str | None, list[str]]:
    warnings = []
    if _has_positive_finite(runs_df, "final_opt_mvp_equiv_sample_work"):
        run_col = "final_opt_mvp_equiv_sample_work"
        iter_col = "opt_mvp_equiv_sample_work_cum"
        label = X_LABEL_CHARGED_WORK
        suffix = "charged_mvp_work"
    else:
        run_col = None
        iter_col = None
        label = None
        suffix = None
        warnings.append(
            "No positive finite charged MVP-equivalent work column found; "
            "skipping work-based analysis."
        )

    runs_df = runs_df.copy()
    runs_df["solver_work"] = (
        pd.to_numeric(runs_df[run_col], errors="coerce") if run_col else np.nan
    )
    if iter_df is not None:
        iter_df = iter_df.copy()
        if iter_col and iter_col in iter_df.columns:
            iter_df["solver_work_cum"] = pd.to_numeric(
                iter_df[iter_col], errors="coerce"
            )
        else:
            iter_df["solver_work_cum"] = np.nan
            if iter_col:
                warnings.append(
                    f"Iteration work column missing; skipped work curves: {iter_col}"
                )
    return runs_df, iter_df, label, suffix, warnings


def _ensure_zero_work_anchor(iter_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if iter_df is None or len(iter_df) == 0:
        return iter_df
    if "run_id" not in iter_df.columns or "solver_work_cum" not in iter_df.columns:
        return iter_df

    df = iter_df.copy()
    df["solver_work_cum"] = pd.to_numeric(df["solver_work_cum"], errors="coerce")
    if "outer_iter" in df.columns:
        df["outer_iter"] = pd.to_numeric(df["outer_iter"], errors="coerce")
    if "energy" in df.columns:
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")

    anchor_rows = []
    for _, run_df in df.groupby("run_id", sort=False):
        rd = run_df[np.isfinite(run_df["solver_work_cum"])].copy()
        if len(rd) == 0:
            continue
        sort_cols = ["solver_work_cum"]
        if "outer_iter" in rd.columns:
            sort_cols.append("outer_iter")
        rd = rd.sort_values(sort_cols)
        first = rd.iloc[0]
        if float(first["solver_work_cum"]) <= 0.0:
            continue

        anchor = first.to_dict()
        first_outer = first.get("outer_iter", np.nan)
        anchor["outer_iter"] = int(first_outer) - 1 if np.isfinite(first_outer) else -1
        anchor["solver_work_cum"] = 0
        if "elapsed_total_sec" in anchor:
            anchor["elapsed_total_sec"] = 0.0
        if "residual_norm" in anchor:
            anchor["residual_norm"] = np.nan
        anchor_rows.append(anchor)

    if len(anchor_rows) == 0:
        return df
    return pd.concat([df, pd.DataFrame(anchor_rows)], ignore_index=True, sort=False)


def _summary_tables(runs_df: pd.DataFrame, baseline_method: str) -> dict[str, pd.DataFrame]:
    overall = (
        runs_df.groupby("method")
        .agg(
            n_runs=("run_id", "count"),
            accuracy_median=("final_test_accuracy", "median"),
            accuracy_iqr=("final_test_accuracy", _iqr),
            loglik_median=("final_test_log_likelihood", "median"),
            loglik_iqr=("final_test_log_likelihood", _iqr),
            energy_median=("final_energy", "median"),
            energy_iqr=("final_energy", _iqr),
            solver_work_median=("solver_work", "median"),
            solver_work_iqr=("solver_work", _iqr),
            wall_median=("elapsed_total_sec", "median"),
            wall_iqr=("elapsed_total_sec", _iqr),
        )
        .reset_index()
        .sort_values("accuracy_median", ascending=False)
    )

    by_dataset = (
        runs_df.groupby(["dataset", "method"])
        .agg(
            n_runs=("run_id", "count"),
            accuracy_median=("final_test_accuracy", "median"),
            accuracy_iqr=("final_test_accuracy", _iqr),
            loglik_median=("final_test_log_likelihood", "median"),
            loglik_iqr=("final_test_log_likelihood", _iqr),
            energy_median=("final_energy", "median"),
            energy_iqr=("final_energy", _iqr),
            solver_work_median=("solver_work", "median"),
            solver_work_iqr=("solver_work", _iqr),
            wall_median=("elapsed_total_sec", "median"),
            wall_iqr=("elapsed_total_sec", _iqr),
        )
        .reset_index()
        .sort_values(["dataset", "accuracy_median"], ascending=[True, False])
    )

    # Pairwise wins vs baseline on matched (dataset, seed).
    pair_index: dict[tuple[str, int], dict[str, dict]] = {}
    for rec in runs_df.to_dict("records"):
        pair_index.setdefault((rec["dataset"], int(rec["seed"])), {})[rec["method"]] = rec

    win_rows = []
    methods = sorted([m for m in runs_df["method"].unique() if m != baseline_method])
    for (dataset, seed), m in sorted(pair_index.items()):
        if baseline_method not in m:
            continue
        base = m[baseline_method]
        for challenger in methods:
            if challenger not in m:
                continue
            ch = m[challenger]
            win_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "challenger": challenger,
                    "win_accuracy": int(ch["final_test_accuracy"] > base["final_test_accuracy"]),
                    "win_loglik": int(
                        ch["final_test_log_likelihood"] > base["final_test_log_likelihood"]
                    ),
                    "win_energy": int(ch["final_energy"] < base["final_energy"]),
                    "win_solver_work": int(ch["solver_work"] < base["solver_work"]),
                }
            )
    win_pairs = pd.DataFrame(win_rows)
    if len(win_pairs) == 0:
        win_summary = pd.DataFrame(
            columns=[
                "challenger",
                "n_pairs",
                "accuracy_win_rate",
                "energy_win_rate",
                "solver_work_win_rate",
            ]
        )
    else:
        win_summary = (
            win_pairs.groupby("challenger")
            .agg(
                n_pairs=("seed", "count"),
                accuracy_win_rate=("win_accuracy", "mean"),
                loglik_win_rate=("win_loglik", "mean"),
                energy_win_rate=("win_energy", "mean"),
                solver_work_win_rate=("win_solver_work", "mean"),
            )
            .reset_index()
            .sort_values("accuracy_win_rate", ascending=False)
        )

    # Efficiency table
    eff = runs_df.copy()
    denom = np.maximum(eff["solver_work"], 1.0)
    eff["accuracy_per_1k_work"] = 1000.0 * eff["final_test_accuracy"] / denom
    eff["loglik_per_1k_work"] = 1000.0 * eff["final_test_log_likelihood"] / denom
    eff["energy_per_1k_work"] = 1000.0 * eff["final_energy"] / denom
    efficiency = (
        eff.groupby("method")
        .agg(
            n_runs=("run_id", "count"),
            acc_per_1k_work_median=("accuracy_per_1k_work", "median"),
            acc_per_1k_work_iqr=("accuracy_per_1k_work", _iqr),
            loglik_per_1k_work_median=("loglik_per_1k_work", "median"),
            loglik_per_1k_work_iqr=("loglik_per_1k_work", _iqr),
            energy_per_1k_work_median=("energy_per_1k_work", "median"),
            energy_per_1k_work_iqr=("energy_per_1k_work", _iqr),
        )
        .reset_index()
        .sort_values("acc_per_1k_work_median", ascending=False)
    )

    return {
        "overall": overall,
        "by_dataset": by_dataset,
        "win_pairs": win_pairs,
        "win_summary": win_summary,
        "efficiency": efficiency,
    }


def _make_plots(
    runs_df: pd.DataFrame,
    iter_df: pd.DataFrame | None,
    out_dir: Path,
    baseline_method: str,
    top_k_datasets_for_curves: int,
    work_label: str | None,
    work_suffix: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy distribution by method
    methods = sorted(runs_df["method"].unique())
    data = [runs_df.loc[runs_df["method"] == m, "final_test_accuracy"].dropna().values for m in methods]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel("Final Test Accuracy")
    plt.title("Accuracy Distribution by Method")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_accuracy_box_by_method.png", dpi=150)
    plt.close()

    has_work = work_label is not None and _has_positive_finite(runs_df, "solver_work")

    # 2) Solver work distribution by method
    if has_work:
        data = [runs_df.loc[runs_df["method"] == m, "solver_work"].dropna().values for m in methods]
        plt.figure(figsize=(7, 4))
        plt.boxplot(data, labels=methods, showfliers=False)
        plt.ylabel(work_label)
        plt.title("Solver Work Distribution by Method")
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_{work_suffix}_box_by_method.png", dpi=150)
        plt.close()

    # 3) Predictive log-likelihood distribution by method
    data = [runs_df.loc[runs_df["method"] == m, "final_test_log_likelihood"].dropna().values for m in methods]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel("Final Test Log-Likelihood")
    plt.title("Predictive Log-Likelihood by Method")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_loglik_box_by_method.png", dpi=150)
    plt.close()

    # 4) Pareto scatter accuracy vs solver work
    if has_work:
        plt.figure(figsize=(7, 5))
        for m in methods:
            d = runs_df[runs_df["method"] == m]
            plt.scatter(d["solver_work"], d["final_test_accuracy"], s=30, alpha=0.6, label=m)
        plt.xlabel(work_label)
        plt.ylabel("Final Test Accuracy")
        plt.title("Accuracy vs Solver Work")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_pareto_accuracy_vs_{work_suffix}.png", dpi=150)
        plt.close()

    # 5) Accuracy by dataset/method (median with IQR errorbar)
    agg = (
        runs_df.groupby(["dataset", "method"])["final_test_accuracy"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    datasets = sorted(agg["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    plt.figure(figsize=(max(10, len(datasets) * 1.2), 4.8))
    for i, m in enumerate(methods):
        d = agg[agg["method"] == m].set_index("dataset").reindex(datasets)
        med = d["median"].values
        low = med - d["q25"].values
        high = d["q75"].values - med
        xpos = x - 0.4 + width / 2 + i * width
        plt.bar(xpos, med, width=width, label=m, alpha=0.85)
        plt.errorbar(xpos, med, yerr=[low, high], fmt="none", ecolor="black", capsize=2, lw=0.8)
    plt.xticks(x, datasets, rotation=30, ha="right")
    plt.ylabel("Final Test Accuracy (median ± IQR)")
    plt.title("Accuracy by Dataset and Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_accuracy_by_dataset_method.png", dpi=150)
    plt.close()

    # 6) Log-likelihood by dataset/method (median with IQR errorbar)
    agg_ll = (
        runs_df.groupby(["dataset", "method"])["final_test_log_likelihood"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    datasets = sorted(agg_ll["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    plt.figure(figsize=(max(10, len(datasets) * 1.2), 4.8))
    for i, m in enumerate(methods):
        d = agg_ll[agg_ll["method"] == m].set_index("dataset").reindex(datasets)
        med = d["median"].values
        low = med - d["q25"].values
        high = d["q75"].values - med
        xpos = x - 0.4 + width / 2 + i * width
        plt.bar(xpos, med, width=width, label=m, alpha=0.85)
        plt.errorbar(xpos, med, yerr=[low, high], fmt="none", ecolor="black", capsize=2, lw=0.8)
    plt.xticks(x, datasets, rotation=30, ha="right")
    plt.ylabel("Final Test Log-Likelihood (median ± IQR)")
    plt.title("Predictive Log-Likelihood by Dataset and Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_loglik_by_dataset_method.png", dpi=150)
    plt.close()

    # 7) Iteration curves vs solver work (if iter_metrics exists)
    if iter_df is not None and len(iter_df) > 0 and _has_positive_finite(iter_df, "solver_work_cum"):
        numeric_cols = ["energy", "residual_norm", "solver_work_cum", "outer_iter"]
        for c in numeric_cols:
            if c in iter_df.columns:
                iter_df[c] = pd.to_numeric(iter_df[c], errors="coerce")

        # choose top-k datasets by count
        ds_counts = iter_df["dataset"].value_counts()
        chosen_ds = ds_counts.head(top_k_datasets_for_curves).index.tolist()
        grid = np.linspace(0.0, float(iter_df["solver_work_cum"].max()), 120)

        for ds in chosen_ds:
            ds_df = iter_df[iter_df["dataset"] == ds].copy()
            if len(ds_df) == 0:
                continue

            for metric, fname, title in [
                ("energy", f"plot_energy_vs_{work_suffix}_curves_{ds}.png", f"Energy vs Solver Work ({ds})"),
                ("residual_norm", f"plot_residual_vs_{work_suffix}_curves_{ds}.png", f"Residual vs Solver Work ({ds})"),
            ]:
                plt.figure(figsize=(7.2, 4.6))
                for m in sorted(ds_df["method"].unique()):
                    method_df = ds_df[ds_df["method"] == m]
                    traces = []
                    for run_id, run_df in method_df.groupby("run_id"):
                        run_df = run_df.sort_values("solver_work_cum")
                        x = run_df["solver_work_cum"].values
                        y = run_df[metric].values
                        good = np.isfinite(x) & np.isfinite(y)
                        x = x[good]
                        y = y[good]
                        if len(x) < 2:
                            continue
                        # enforce monotonic x for interp
                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                        x_unique, idx = np.unique(x, return_index=True)
                        y = y[idx]
                        if len(x_unique) < 2:
                            continue
                        yi = np.interp(grid, x_unique, y, left=np.nan, right=np.nan)
                        yi[grid < x_unique.min()] = np.nan
                        yi[grid > x_unique.max()] = np.nan
                        traces.append(yi)
                    if len(traces) == 0:
                        continue
                    arr = np.asarray(traces)
                    med = np.nanmedian(arr, axis=0)
                    q25 = np.nanquantile(arr, 0.25, axis=0)
                    q75 = np.nanquantile(arr, 0.75, axis=0)
                    plt.plot(grid, med, label=m)
                    plt.fill_between(grid, q25, q75, alpha=0.2)
                plt.xlabel(work_label)
                plt.ylabel(metric)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / fname, dpi=150)
                plt.close()

        # 8) Normalized energy progress vs solver work, aggregated across all runs
        norm_df = _ensure_zero_work_anchor(iter_df)
        norm_df["energy"] = pd.to_numeric(norm_df["energy"], errors="coerce")
        norm_df["solver_work_cum"] = pd.to_numeric(norm_df["solver_work_cum"], errors="coerce")
        run_meta = (
            norm_df.sort_values("outer_iter")
            .groupby("run_id")
            .agg(
                method=("method", "first"),
                dataset=("dataset", "first"),
                e0=("energy", "first"),
                efinal=("energy", "last"),
                work_max=("solver_work_cum", "max"),
            )
            .reset_index()
        )
        norm_df = norm_df.merge(run_meta[["run_id", "e0", "efinal", "work_max"]], on="run_id", how="left")
        denom = norm_df["e0"] - norm_df["efinal"]
        norm_df["energy_norm"] = (norm_df["energy"] - norm_df["efinal"]) / np.where(np.abs(denom) > 1e-12, denom, np.nan)

        global_work_max = float(np.nanmax(norm_df["solver_work_cum"].values))
        if np.isfinite(global_work_max) and global_work_max > 0:
            grid = np.linspace(0.0, global_work_max, 160)
            plt.figure(figsize=(8, 5))
            for m in sorted(norm_df["method"].dropna().unique()):
                md = norm_df[norm_df["method"] == m]
                traces = []
                for run_id, rd in md.groupby("run_id"):
                    rd = rd.sort_values("solver_work_cum")
                    x = rd["solver_work_cum"].values
                    y = rd["energy_norm"].values
                    good = np.isfinite(x) & np.isfinite(y)
                    x = x[good]
                    y = y[good]
                    if len(x) < 2:
                        continue
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    xu, idx = np.unique(x, return_index=True)
                    y = y[idx]
                    if len(xu) < 2:
                        continue
                    yi = np.interp(grid, xu, y, left=np.nan, right=np.nan)
                    yi[grid < xu.min()] = np.nan
                    yi[grid > xu.max()] = np.nan
                    traces.append(yi)
                if len(traces) == 0:
                    continue
                arr = np.asarray(traces)
                med = np.nanmedian(arr, axis=0)
                q25 = np.nanquantile(arr, 0.25, axis=0)
                q75 = np.nanquantile(arr, 0.75, axis=0)
                plt.plot(grid, med, label=m)
                plt.fill_between(grid, q25, q75, alpha=0.2)
            plt.xlabel(work_label)
            plt.ylabel("Normalized Energy (1=start, 0=end)")
            plt.title("Normalized Energy Progress vs Solver Work (All Datasets)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"plot_energy_norm_vs_{work_suffix}_all.png", dpi=150)
            plt.close()

            # 9) Faceted small multiples by dataset (normalized energy)
            datasets = sorted(norm_df["dataset"].dropna().unique().tolist())
            n = len(datasets)
            if n > 0:
                ncols = 3
                nrows = int(np.ceil(n / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
                for ax in axes.flat:
                    ax.axis("off")
                for i, ds in enumerate(datasets):
                    ax = axes[i // ncols, i % ncols]
                    ax.axis("on")
                    dsd = norm_df[norm_df["dataset"] == ds]
                    for m in sorted(dsd["method"].dropna().unique()):
                        md = dsd[dsd["method"] == m]
                        traces = []
                        for run_id, rd in md.groupby("run_id"):
                            rd = rd.sort_values("solver_work_cum")
                            x = rd["solver_work_cum"].values
                            y = rd["energy_norm"].values
                            good = np.isfinite(x) & np.isfinite(y)
                            x = x[good]
                            y = y[good]
                            if len(x) < 2:
                                continue
                            order = np.argsort(x)
                            x = x[order]
                            y = y[order]
                            xu, idx = np.unique(x, return_index=True)
                            y = y[idx]
                            if len(xu) < 2:
                                continue
                            yi = np.interp(grid, xu, y, left=np.nan, right=np.nan)
                            yi[grid < xu.min()] = np.nan
                            yi[grid > xu.max()] = np.nan
                            traces.append(yi)
                        if len(traces) == 0:
                            continue
                        arr = np.asarray(traces)
                        med = np.nanmedian(arr, axis=0)
                        q25 = np.nanquantile(arr, 0.25, axis=0)
                        q75 = np.nanquantile(arr, 0.75, axis=0)
                        ax.plot(grid, med, label=m, lw=1.4)
                        ax.fill_between(grid, q25, q75, alpha=0.2)
                    ax.set_title(ds)
                    ax.set_xlabel("Solver Work")
                    ax.set_ylabel("Norm Energy")
                handles, labels = axes[0, 0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
                fig.suptitle("Normalized Energy vs Solver Work by Dataset", y=1.02)
                fig.tight_layout()
                fig.savefig(out_dir / f"plot_energy_norm_vs_{work_suffix}_facets.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


def _work_to_target_tables(iter_df: pd.DataFrame | None, baseline_method: str) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    if iter_df is None or len(iter_df) == 0 or "solver_work_cum" not in iter_df.columns:
        return {
            "work_to_target_pairs": empty,
            "work_to_target_summary": empty,
            "work_speedup_vs_baseline": empty,
            "work_winrate_vs_baseline": empty,
        }

    df = _ensure_zero_work_anchor(iter_df)
    for c in ["energy", "solver_work_cum", "outer_iter"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["run_id", "method", "dataset", "seed", "energy", "solver_work_cum"])
    if len(df) == 0:
        return {
            "work_to_target_pairs": empty,
            "work_to_target_summary": empty,
            "work_speedup_vs_baseline": empty,
            "work_winrate_vs_baseline": empty,
        }

    # Normalize each run energy to [~1 -> 0] based on observed start/end.
    run_meta = (
        df.sort_values("outer_iter")
        .groupby("run_id")
        .agg(
            method=("method", "first"),
            dataset=("dataset", "first"),
            seed=("seed", "first"),
            e0=("energy", "first"),
            efinal=("energy", "last"),
        )
        .reset_index()
    )
    df = df.merge(run_meta[["run_id", "e0", "efinal"]], on="run_id", how="left")
    denom = df["e0"] - df["efinal"]
    df["energy_norm"] = (df["energy"] - df["efinal"]) / np.where(np.abs(denom) > 1e-12, denom, np.nan)

    targets = [0.5, 0.2, 0.1]
    rows = []
    for run_id, rd in df.groupby("run_id"):
        rd = rd.sort_values("solver_work_cum")
        method = rd["method"].iloc[0]
        dataset = rd["dataset"].iloc[0]
        seed = rd["seed"].iloc[0]
        for t in targets:
            reached = rd[rd["energy_norm"] <= t]
            work_hit = np.nan if len(reached) == 0 else float(reached["solver_work_cum"].iloc[0])
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "target_energy_norm": t,
                    "work_to_target": work_hit,
                    "hit_target": int(np.isfinite(work_hit)),
                }
            )
    pairs = pd.DataFrame(rows)
    if len(pairs) == 0:
        return {
            "work_to_target_pairs": empty,
            "work_to_target_summary": empty,
            "work_speedup_vs_baseline": empty,
            "work_winrate_vs_baseline": empty,
        }

    summary = (
        pairs.groupby(["target_energy_norm", "dataset", "method"])
        .agg(
            n_runs=("run_id", "count"),
            hit_rate=("hit_target", "mean"),
            work_to_target_median=("work_to_target", "median"),
            work_to_target_iqr=("work_to_target", _iqr),
        )
        .reset_index()
    )

    # Speedup/win-rate vs baseline per matched dataset-seed-target
    speed_rows = []
    win_rows = []
    methods = sorted([m for m in pairs["method"].dropna().unique() if m != baseline_method])
    idx = {}
    for r in pairs.to_dict("records"):
        key = (r["dataset"], int(r["seed"]), float(r["target_energy_norm"]))
        idx.setdefault(key, {})[r["method"]] = r

    for key, m in idx.items():
        if baseline_method not in m:
            continue
        b = m[baseline_method]
        base_work = b["work_to_target"]
        for ch in methods:
            if ch not in m:
                continue
            c = m[ch]
            challenger_work = c["work_to_target"]
            speedup = np.nan
            if np.isfinite(base_work) and np.isfinite(challenger_work) and challenger_work > 0:
                speedup = float(base_work / challenger_work)
            speed_rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "target_energy_norm": key[2],
                    "challenger": ch,
                    "baseline_work_to_target": base_work,
                    "challenger_work_to_target": challenger_work,
                    "speedup_vs_baseline": speedup,
                }
            )
            win_rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "target_energy_norm": key[2],
                    "challenger": ch,
                    "win_work_to_target": int(
                        np.isfinite(challenger_work)
                        and (not np.isfinite(base_work) or challenger_work < base_work)
                    ),
                }
            )

    speed_df = pd.DataFrame(speed_rows)
    win_df = pd.DataFrame(win_rows)
    if len(win_df) > 0:
        win_summary = (
            win_df.groupby(["target_energy_norm", "challenger"])
            .agg(
                n_pairs=("seed", "count"),
                win_rate=("win_work_to_target", "mean"),
            )
            .reset_index()
            .sort_values(["target_energy_norm", "challenger"])
        )
    else:
        win_summary = pd.DataFrame(columns=["target_energy_norm", "challenger", "n_pairs", "win_rate"])

    return {
        "work_to_target_pairs": pairs,
        "work_to_target_summary": summary,
        "work_speedup_vs_baseline": speed_df,
        "work_winrate_vs_baseline": win_summary,
    }


def _plot_work_to_target_winrate(work_winrate_df: pd.DataFrame, out_dir: Path, baseline_method: str) -> None:
    if len(work_winrate_df) == 0:
        return
    targets = sorted(work_winrate_df["target_energy_norm"].unique().tolist(), reverse=True)
    challengers = sorted(work_winrate_df["challenger"].unique().tolist())
    x = np.arange(len(targets))
    width = 0.8 / max(1, len(challengers))
    plt.figure(figsize=(8, 4.8))
    for i, ch in enumerate(challengers):
        d = work_winrate_df[work_winrate_df["challenger"] == ch].set_index("target_energy_norm").reindex(targets)
        vals = d["win_rate"].values
        xpos = x - 0.4 + width / 2 + i * width
        plt.bar(xpos, vals, width=width, label=f"{ch} vs {baseline_method}", alpha=0.85)
    plt.xticks(x, [str(t) for t in targets])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Win Rate")
    plt.xlabel("Target Normalized Energy")
    plt.title("Win Rate on Work-to-Target Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_work_to_target_winrate.png", dpi=150)
    plt.close()


def _highlights(runs_df: pd.DataFrame, tables: dict[str, pd.DataFrame], baseline_method: str) -> dict:
    overall = tables["overall"]
    by_dataset = tables["by_dataset"]

    best_acc_method = overall.sort_values("accuracy_median", ascending=False).iloc[0]
    best_loglik_method = overall.sort_values("loglik_median", ascending=False).iloc[0]
    best_work_method = overall.sort_values("solver_work_median", ascending=True).iloc[0]

    dataset_winners = []
    for ds, grp in by_dataset.groupby("dataset"):
        top = grp.sort_values("accuracy_median", ascending=False).iloc[0]
        dataset_winners.append(
            {
                "dataset": ds,
                "winner_method": top["method"],
                "winner_accuracy_median": float(top["accuracy_median"]),
            }
        )

    out = {
        "best_overall_accuracy_method": {
            "method": best_acc_method["method"],
            "accuracy_median": float(best_acc_method["accuracy_median"]),
        },
        "best_overall_loglik_method": {
            "method": best_loglik_method["method"],
            "loglik_median": float(best_loglik_method["loglik_median"]),
        },
        "lowest_solver_work_method": {
            "method": best_work_method["method"],
            "solver_work_median": float(best_work_method["solver_work_median"]),
        },
        "dataset_accuracy_winners": dataset_winners,
        "baseline_method": baseline_method,
    }
    return out


def _write_markdown_report(out_dir: Path, tables: dict[str, pd.DataFrame], highlights: dict) -> None:
    def _to_md(df: pd.DataFrame, max_rows: int = 200) -> str:
        if len(df) == 0:
            return "_No rows_\\n"
        return df.head(max_rows).to_markdown(index=False) + "\\n"

    lines = []
    lines.append("# Bayesian Logistic Benchmark Analysis\\n")
    lines.append(f"- Generated: {datetime.now().isoformat()}\\n")
    lines.append("")
    lines.append("## Highlights\\n")
    lines.append(f"- Best overall accuracy method: `{highlights['best_overall_accuracy_method']['method']}` "
                 f"(median={highlights['best_overall_accuracy_method']['accuracy_median']:.4f})")
    lines.append(f"- Best overall log-likelihood method: `{highlights['best_overall_loglik_method']['method']}` "
                 f"(median={highlights['best_overall_loglik_method']['loglik_median']:.4f})")
    lines.append(f"- Lowest solver work method: `{highlights['lowest_solver_work_method']['method']}` "
                 f"(median work={highlights['lowest_solver_work_method']['solver_work_median']:.2f})\\n")

    lines.append("## Overall Table\\n")
    lines.append(_to_md(tables["overall"]))

    lines.append("## By Dataset Table\\n")
    lines.append(_to_md(tables["by_dataset"]))

    lines.append("## Win Summary vs Baseline\\n")
    lines.append(_to_md(tables["win_summary"]))

    if "work_winrate_vs_baseline" in tables:
        lines.append("## Work-to-Target Win Summary vs Baseline\\n")
        lines.append(_to_md(tables["work_winrate_vs_baseline"]))

    lines.append("## Efficiency Table\\n")
    lines.append(_to_md(tables["efficiency"]))

    (out_dir / "report_tables.md").write_text("\\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Bayesian logistic benchmark outputs into plots and tables."
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--baseline-method", type=str, default="GF")
    parser.add_argument("--top-k-datasets-for-curves", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    _ensure_exists(results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df, load_warnings = _load_runs(results_dir)
    iter_df = _load_iter(results_dir)

    warnings = list(load_warnings)

    if args.datasets is not None:
        before = len(runs_df)
        runs_df = runs_df[runs_df["dataset"].isin(args.datasets)]
        if iter_df is not None:
            iter_df = iter_df[iter_df["dataset"].isin(args.datasets)]
        warnings.append(f"Dataset filter applied: {before} -> {len(runs_df)} rows.")

    if args.methods is not None:
        before = len(runs_df)
        runs_df = runs_df[runs_df["method"].isin(args.methods)]
        if iter_df is not None:
            iter_df = iter_df[iter_df["method"].isin(args.methods)]
        warnings.append(f"Method filter applied: {before} -> {len(runs_df)} rows.")

    runs_df = runs_df.dropna(subset=["dataset", "seed", "method"])
    if len(runs_df) == 0:
        raise ValueError("No runs left after filtering.")

    # Incompleteness diagnostics by pair.
    methods_present = sorted(runs_df["method"].dropna().unique().tolist())
    by_pair = runs_df.groupby(["dataset", "seed"])["method"].nunique().reset_index(name="n_methods")
    incomplete = by_pair[by_pair["n_methods"] < len(methods_present)]
    if len(incomplete) > 0:
        warnings.append(
            f"Incomplete dataset-seed pairs detected: {len(incomplete)} pairs have fewer than {len(methods_present)} methods."
        )
        incomplete.to_csv(out_dir / "incomplete_pairs.csv", index=False)

    runs_df, iter_df, work_label, work_suffix, work_warnings = _prepare_work_axis(
        runs_df, iter_df
    )
    warnings.extend(work_warnings)

    tables = _summary_tables(runs_df, baseline_method=args.baseline_method)
    work_tables = _work_to_target_tables(iter_df, baseline_method=args.baseline_method)
    tables.update(work_tables)

    tables["overall"].to_csv(out_dir / "table_overall.csv", index=False)
    tables["by_dataset"].to_csv(out_dir / "table_by_dataset.csv", index=False)
    tables["win_pairs"].to_csv(out_dir / "table_win_pairs_vs_baseline.csv", index=False)
    tables["win_summary"].to_csv(out_dir / "table_winrate_vs_baseline.csv", index=False)
    tables["efficiency"].to_csv(out_dir / "table_efficiency.csv", index=False)
    tables["work_to_target_pairs"].to_csv(out_dir / "table_work_to_target_pairs.csv", index=False)
    tables["work_to_target_summary"].to_csv(out_dir / "table_work_to_target_summary.csv", index=False)
    tables["work_speedup_vs_baseline"].to_csv(out_dir / "table_work_speedup_vs_baseline.csv", index=False)
    tables["work_winrate_vs_baseline"].to_csv(out_dir / "table_work_winrate_vs_baseline.csv", index=False)

    _make_plots(
        runs_df=runs_df,
        iter_df=iter_df,
        out_dir=out_dir,
        baseline_method=args.baseline_method,
        top_k_datasets_for_curves=args.top_k_datasets_for_curves,
        work_label=work_label,
        work_suffix=work_suffix,
    )
    _plot_work_to_target_winrate(
        tables["work_winrate_vs_baseline"], out_dir=out_dir, baseline_method=args.baseline_method
    )

    highlights = _highlights(runs_df, tables, baseline_method=args.baseline_method)
    (out_dir / "highlights.json").write_text(json.dumps(highlights, indent=2), encoding="utf-8")

    _write_markdown_report(out_dir, tables, highlights)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "results_dir": str(results_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "baseline_method": args.baseline_method,
        "datasets_filter": args.datasets,
        "methods_filter": args.methods,
        "n_runs_rows": int(len(runs_df)),
        "iter_metrics_present": bool(iter_df is not None and len(iter_df) > 0),
        "work_axis": work_label,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if warnings:
        (out_dir / "analysis_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
        print("\n".join(warnings))

    print(f"Saved analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

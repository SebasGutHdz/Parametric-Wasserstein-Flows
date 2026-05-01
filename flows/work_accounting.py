from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _sample_size(z_samples: Any) -> int:
    try:
        return int(z_samples.shape[0])
    except (AttributeError, IndexError, TypeError):
        return 1


@dataclass
class WorkCounter:
    """Charged MVP-equivalent optimization work accounting."""

    grad_mvp_weight: float = 1.0
    grad_calls: int = 0
    solve_calls: int = 0
    direct_mvp_calls: int = 0
    grad_mvp_equiv: float = 0.0
    solve_mvp_equiv: float = 0.0
    direct_mvp_equiv: float = 0.0
    grad_sample_work: float = 0.0
    solve_sample_work: float = 0.0
    direct_mvp_sample_work: float = 0.0

    def record_grad(self, z_samples: Any) -> None:
        sample_size = _sample_size(z_samples)
        self.grad_calls += 1
        self.grad_mvp_equiv += self.grad_mvp_weight
        self.grad_sample_work += self.grad_mvp_weight * sample_size

    def record_solve(self, z_samples: Any, solver_maxiter: int) -> None:
        sample_size = _sample_size(z_samples)
        charged_work = float(solver_maxiter)
        self.solve_calls += 1
        self.solve_mvp_equiv += charged_work
        self.solve_sample_work += charged_work * sample_size

    def record_direct_mvp(self, z_samples: Any, count: int = 1) -> None:
        if count <= 0:
            return
        sample_size = _sample_size(z_samples)
        charged_work = float(count)
        self.direct_mvp_calls += int(count)
        self.direct_mvp_equiv += charged_work
        self.direct_mvp_sample_work += charged_work * sample_size

    def snapshot(self) -> dict[str, float | int]:
        opt_mvp_equiv = (
            self.grad_mvp_equiv + self.solve_mvp_equiv + self.direct_mvp_equiv
        )
        opt_sample_work = (
            self.grad_sample_work
            + self.solve_sample_work
            + self.direct_mvp_sample_work
        )
        return {
            "grad_calls_cum": self.grad_calls,
            "solve_calls_cum": self.solve_calls,
            "direct_mvp_calls_cum": self.direct_mvp_calls,
            "grad_mvp_equiv_cum": self.grad_mvp_equiv,
            "solve_mvp_equiv_cum": self.solve_mvp_equiv,
            "direct_mvp_equiv_cum": self.direct_mvp_equiv,
            "opt_mvp_equiv_cum": opt_mvp_equiv,
            "grad_sample_work_cum": self.grad_sample_work,
            "solve_sample_work_cum": self.solve_sample_work,
            "direct_mvp_sample_work_cum": self.direct_mvp_sample_work,
            "opt_mvp_equiv_sample_work_cum": opt_sample_work,
        }

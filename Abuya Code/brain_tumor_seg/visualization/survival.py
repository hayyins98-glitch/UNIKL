"""
Survival label views: where one case sits among the labeled cohort.

Only a minority of cases carry an overall-survival value, so these helpers
always report the coverage alongside the numbers — a distribution drawn from a
sparse subset is easy to over-read otherwise.

Survival is plotted on a log x-axis because the labels span 10 to 5274 days;
on a linear axis the handful of multi-year survivors flatten everything else
into the first bin.
"""
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from brain_tumor_seg.data.survival import summarize_survival_days

DAYS_PER_YEAR = 365.25


def format_survival(days: Optional[float]) -> str:
    """
    Render a survival value with a years figure next to it.

    Args:
        days: Survival in days, or None for an unlabeled case.

    Returns:
        Human-readable string, e.g. '397 days (1.1 years)' or 'no label'.
    """
    if days is None:
        return "no label"
    return f"{days:,.0f} days ({days / DAYS_PER_YEAR:.1f} years)"


def describe_case_survival(case_id: str, survival: Dict[str, float]) -> str:
    """
    One-line summary of a single case's survival label and its percentile.

    Args:
        case_id:  Case folder name, e.g. 'BraTS-PED-00001-000'.
        survival: {case_id: survival_days} for the labeled cases.

    Returns:
        A line naming the case, its survival and where it falls in the cohort.
    """
    days = survival.get(case_id)
    if days is None:
        return f"{case_id}: no survival label in the metadata"

    values = np.asarray(list(survival.values()), dtype=np.float64)
    percentile = float((values <= days).mean() * 100.0)
    return (
        f"{case_id}: {format_survival(days)} "
        f"— {percentile:.0f}th percentile of {values.size} labeled cases"
    )


def print_survival_report(
    survival: Dict[str, float],
    case_ids: Optional[Iterable[str]] = None,
    highlight_case: Optional[str] = None,
) -> None:
    """
    Print label coverage, the cohort's spread, and optionally one case.

    Args:
        survival:       {case_id: survival_days} for the labeled cases.
        case_ids:       Cases the model actually trains on, used to report how
                        many of them are labeled. None skips the coverage line.
        highlight_case: Case to describe individually.
    """
    stats = summarize_survival_days(survival.values())
    print(f"Labeled cases in metadata : {stats['count']}")

    if case_ids is not None:
        ids = list(case_ids)
        labeled = sum(1 for case_id in ids if case_id in survival)
        share = 100.0 * labeled / len(ids) if ids else 0.0
        print(f"Labeled in this folder    : {labeled} of {len(ids)} ({share:.0f}%)")

    print(f"Survival range            : {format_survival(stats['min'])}"
          f"  to  {format_survival(stats['max'])}")
    print(f"Median / mean             : {stats['median']:,.0f} / {stats['mean']:,.0f} days")

    if highlight_case is not None:
        print(f"\n{describe_case_survival(highlight_case, survival)}")


def plot_survival_distribution(
    survival: Dict[str, float],
    highlight_case: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 4),
    show: bool = True,
) -> plt.Figure:
    """
    Histogram of overall survival across the labeled cases.

    Args:
        survival:       {case_id: survival_days} for the labeled cases.
        highlight_case: Case to mark with a vertical line, if it has a label.
        bins:           Histogram bin count (log-spaced).
        figsize:        Figure size in inches.
        show:           Display inline; False just returns the figure.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If there are no labeled cases to plot.
    """
    values = np.asarray(list(survival.values()), dtype=np.float64)
    if values.size == 0:
        raise ValueError("No survival labels to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    edges = np.logspace(np.log10(max(values.min(), 1.0)), np.log10(values.max()), bins)
    ax.hist(values, bins=edges, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xscale("log")

    median = float(np.median(values))
    ax.axvline(median, color="#555555", linestyle="--", linewidth=1.5,
               label=f"median {median:,.0f} d")

    highlight_days = survival.get(highlight_case) if highlight_case else None
    if highlight_days is not None:
        ax.axvline(highlight_days, color="#C44E52", linewidth=2.0,
                   label=f"{highlight_case}  {highlight_days:,.0f} d")

    ax.set_xlabel("Overall survival (days, log scale)")
    ax.set_ylabel("Cases")
    ax.set_title(f"Overall survival across {values.size} labeled cases")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

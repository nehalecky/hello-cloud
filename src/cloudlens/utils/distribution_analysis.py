"""
Distribution analysis utilities for time series data.

Provides reusable functions for comparing distributions, computing statistical tests,
and visualizing results using matplotlib/seaborn.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from typing import Optional, Dict, Tuple


def plot_pdf_cdf_comparison(
    distribution1: np.ndarray,
    distribution2: Optional[np.ndarray] = None,
    label1: str = "Distribution 1",
    label2: str = "Distribution 2",
    color1: str = '#1f77b4',
    color2: str = '#ff7f0e',
    figsize: Tuple[int, int] = (14, 5),
    alpha: float = 0.3
) -> plt.Figure:
    """
    Plot PDF and CDF side-by-side for one or two distributions.

    Args:
        distribution1: First distribution (numpy array)
        distribution2: Optional second distribution for comparison
        label1: Label for first distribution
        label2: Label for second distribution
        color1: Color for first distribution (hex or named)
        color2: Color for second distribution (hex or named)
        figsize: Figure size (width, height)
        alpha: Transparency for filled regions

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- PDF Plot ---
    # Compute KDE for first distribution
    x_min = distribution1.min()
    x_max = distribution1.max()

    if distribution2 is not None:
        x_min = min(x_min, distribution2.min())
        x_max = max(x_max, distribution2.max())

    x_range = np.linspace(x_min, x_max, 200)
    kde1 = gaussian_kde(distribution1)
    pdf1 = kde1(x_range)

    # Plot first distribution
    axes[0].plot(x_range, pdf1, linewidth=2.5, label=label1, color=color1)
    axes[0].fill_between(x_range, pdf1, alpha=alpha, color=color1)

    # Plot second distribution if provided
    if distribution2 is not None:
        kde2 = gaussian_kde(distribution2)
        pdf2 = kde2(x_range)
        axes[0].plot(x_range, pdf2, linewidth=2.5, label=label2, color=color2)
        axes[0].fill_between(x_range, pdf2, alpha=alpha, color=color2)

    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Probability Density Function (PDF)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)

    # --- CDF Plot ---
    # Compute empirical CDF for first distribution
    sorted1 = np.sort(distribution1)
    cdf1 = np.arange(1, len(sorted1) + 1) / len(sorted1)
    axes[1].plot(sorted1, cdf1, linewidth=2.5, label=label1, color=color1)

    # Plot second distribution if provided
    if distribution2 is not None:
        sorted2 = np.sort(distribution2)
        cdf2 = np.arange(1, len(sorted2) + 1) / len(sorted2)
        axes[1].plot(sorted2, cdf2, linewidth=2.5, label=label2, color=color2)

    axes[1].set_xlabel('Value', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distribution_comparison(
    distribution1: np.ndarray,
    distribution2: np.ndarray,
    label1: str,
    label2: str,
    palette: str = 'colorblind',
    figsize: Tuple[int, int] = (14, 5),
    bins: int = 50
) -> plt.Figure:
    """
    Plot KDE and histogram side-by-side for two distributions using seaborn.

    Args:
        distribution1: First distribution (numpy array)
        distribution2: Second distribution (numpy array)
        label1: Label for first distribution
        label2: Label for second distribution
        palette: Seaborn color palette name
        figsize: Figure size (width, height)
        bins: Number of histogram bins

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Prepare data for seaborn
    data = pd.concat([
        pd.DataFrame({'value': distribution1, 'type': label1}),
        pd.DataFrame({'value': distribution2, 'type': label2})
    ])

    # KDE plot
    sns.kdeplot(data=data, x='value', hue='type', fill=True, alpha=0.5,
                linewidth=2.5, ax=axes[0], palette=palette)
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'KDE: {label1} vs {label2}', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Histogram with KDE overlay
    sns.histplot(data=data, x='value', hue='type', bins=bins,
                 stat='density', alpha=0.6, kde=True, ax=axes[1], palette=palette)
    axes[1].set_xlabel('Value', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'Histogram: {label1} vs {label2}', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    return fig


def compute_ks_tests(
    comparisons: Dict[str, Tuple],
    data_segments: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Compute Kolmogorov-Smirnov tests for multiple distribution pairs.

    The KS test tests the null hypothesis that two samples come from the same distribution.
    A p-value < alpha indicates significantly different distributions.

    Args:
        comparisons: Dictionary mapping comparison names to either:
            - (dist1, dist2) tuples of numpy arrays, OR
            - (key1, key2) tuples of string keys to look up in data_segments
        data_segments: Optional dictionary mapping keys to numpy arrays
        alpha: Significance level for hypothesis test

    Returns:
        Dictionary of results (can be wrapped in DataFrame)
    """
    results = {}

    for name, (item1, item2) in comparisons.items():
        # Resolve data: either use directly or look up in data_segments
        if isinstance(item1, str) and data_segments is not None:
            dist1 = data_segments[item1]
            dist2 = data_segments[item2]
        else:
            dist1 = item1
            dist2 = item2

        ks_stat, ks_pval = stats.ks_2samp(dist1, dist2)
        results[name] = {
            'KS Statistic': float(ks_stat),
            'p-value': float(ks_pval),
            f'Significant (α={alpha})': bool(ks_pval < alpha),
            'Interpretation': 'Different distributions' if ks_pval < alpha else 'Similar distributions'
        }

    return results


def compute_kl_divergence(
    reference: np.ndarray,
    comparison: np.ndarray,
    bins: int = 100
) -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.

    KL(P || Q) measures how distribution Q diverges from reference distribution P.
    Higher values indicate greater distributional difference. KL divergence is asymmetric.

    Args:
        reference: Reference distribution (P)
        comparison: Comparison distribution (Q)
        bins: Number of bins for histogram approximation

    Returns:
        KL divergence value (non-negative float)
    """
    # Create common bin edges
    all_data = np.concatenate([reference, comparison])
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)

    # Compute normalized histograms (probability distributions)
    p_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(comparison, bins=bin_edges, density=True)

    # Normalize to probabilities (ensure they sum to 1)
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    # Compute KL divergence: sum(p * log(p/q))
    kl_div = np.sum(rel_entr(p_hist, q_hist))

    return float(kl_div)


def compute_kl_divergences(
    comparisons: Dict[str, Tuple],
    data_segments: Optional[Dict[str, np.ndarray]] = None,
    bins: int = 100,
    symmetric: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute KL divergences for multiple distribution pairs.

    Args:
        comparisons: Dictionary mapping comparison names to either:
            - (dist1, dist2) tuples of numpy arrays, OR
            - (key1, key2) tuples of string keys to look up in data_segments
        data_segments: Optional dictionary mapping keys to numpy arrays
        bins: Number of bins for histogram approximation
        symmetric: If True, compute both KL(dist1||dist2) and KL(dist2||dist1)

    Returns:
        Dictionary of results (can be wrapped in DataFrame)
    """
    results = {}

    for name, (item1, item2) in comparisons.items():
        # Resolve data: either use directly or look up in data_segments
        if isinstance(item1, str) and data_segments is not None:
            dist1 = data_segments[item1]
            dist2 = data_segments[item2]
        else:
            dist1 = item1
            dist2 = item2

        kl_forward = compute_kl_divergence(dist1, dist2, bins=bins)

        # Interpret KL divergence magnitude
        interpretation = (
            'Very different' if kl_forward > 1.0 else
            'Moderately different' if kl_forward > 0.1 else
            'Similar'
        )

        results[name] = {
            'KL Divergence': float(kl_forward),
            'Interpretation': interpretation
        }

        # Add reverse direction if symmetric
        if symmetric:
            kl_reverse = compute_kl_divergence(dist2, dist1, bins=bins)
            reverse_name = ' ↔ '.join(reversed(name.split(' vs '))) if ' vs ' in name else f'{name} (reversed)'

            interpretation_reverse = (
                'Very different' if kl_reverse > 1.0 else
                'Moderately different' if kl_reverse > 0.1 else
                'Similar'
            )

            results[reverse_name] = {
                'KL Divergence': float(kl_reverse),
                'Interpretation': interpretation_reverse
            }

    return results


def plot_statistical_tests(
    ks_results: Dict[str, Dict],
    kl_results: Dict[str, Dict],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Visualize KS test and KL divergence results as horizontal bar charts.

    Args:
        ks_results: Dictionary from compute_ks_tests()
        kl_results: Dictionary from compute_kl_divergences()
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- KS Test Results ---
    ks_comparisons = list(ks_results.keys())
    ks_statistics = [ks_results[k]['KS Statistic'] for k in ks_comparisons]
    ks_significant = [ks_results[k].get('Significant (α=0.05)', False) for k in ks_comparisons]

    # Color by significance
    colors_ks = ['#d62728' if sig else '#2ca02c' for sig in ks_significant]

    axes[0].barh(ks_comparisons, ks_statistics, color=colors_ks, alpha=0.7)
    axes[0].set_xlabel('KS Statistic', fontsize=12)
    axes[0].set_title('Kolmogorov-Smirnov Test Results', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0.05, color='orange', linestyle='--', linewidth=2,
                    label='Typical Threshold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='x', alpha=0.3)

    # --- KL Divergence Results ---
    kl_comparisons = list(kl_results.keys())
    kl_divergences = [kl_results[k]['KL Divergence'] for k in kl_comparisons]
    kl_interpretations = [kl_results[k]['Interpretation'] for k in kl_comparisons]

    # Color by interpretation
    colors_kl = [
        '#d62728' if 'Very' in interp else
        '#ff7f0e' if 'Moderately' in interp else
        '#2ca02c'
        for interp in kl_interpretations
    ]

    axes[1].barh(kl_comparisons, kl_divergences, color=colors_kl, alpha=0.7)
    axes[1].set_xlabel('KL Divergence', fontsize=12)
    axes[1].set_title('Kullback-Leibler Divergence', fontsize=13, fontweight='bold')
    axes[1].axvline(x=0.1, color='orange', linestyle='--', linewidth=2,
                    label='Moderate')
    axes[1].axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                    label='High')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def print_distribution_summary(
    ks_results: Dict[str, Dict],
    kl_results: Dict[str, Dict],
    key_comparison: str = 'Train vs Test'
) -> None:
    """
    Print a formatted summary of distribution analysis results.

    Args:
        ks_results: Dictionary from compute_ks_tests()
        kl_results: Dictionary from compute_kl_divergences()
        key_comparison: Name of the most important comparison to highlight
    """
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS SUMMARY")
    print("=" * 70)

    # Print all KS test results
    for comparison, results in ks_results.items():
        p_value = results['p-value']
        significant = p_value < 0.05

        status = 'SIGNIFICANTLY DIFFERENT ⚠️' if significant else 'SIMILAR ✓'
        print(f"{comparison}: {status} (p={p_value:.4f})")

    # Special handling for key comparison
    if key_comparison in ks_results:
        key_results = ks_results[key_comparison]
        key_significant = key_results['p-value'] < 0.05

        print(f"\n{'=' * 70}")
        print(f"KEY FINDING: {key_comparison}")
        print(f"{'=' * 70}")

        if key_significant:
            print("  ⚠️  Distribution shift detected - model may not generalize well")
            print("  → Consider temporal validation or distribution-aware training")
        else:
            print("  ✓  Distributions are similar - standard train/test split is valid")
            print("  → Safe to proceed with standard modeling approaches")

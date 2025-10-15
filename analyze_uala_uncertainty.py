import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np


def load_entries(path: str) -> List[dict]:
    entries: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except Exception:
                # skip malformed lines
                continue
    return entries


def extract_final_uncertainty(entry: dict) -> Tuple[float, str]:
    """
    Returns (final_uncertainty, source), where source is either "react" or "cot".
    Prefers ReAct uncertainty when available; falls back to CoT uncertainty.
    """
    react_unc = entry.get("react_uncertainty")
    if isinstance(react_unc, (int, float)) and not math.isnan(float(react_unc)):
        return float(react_unc), "react"
    cot_unc = entry.get("cot_uncertainty")
    if isinstance(cot_unc, (int, float)) and not math.isnan(float(cot_unc)):
        return float(cot_unc), "cot"
    return float("nan"), "none"


def point_biserial_corr(binary: List[int], values: List[float]) -> float:
    """
    Pearson correlation between a binary variable (0/1) and a continuous variable.
    """
    x = np.asarray(binary, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to handle ties (like scipy.stats.rankdata with method='average')."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1)
    # handle ties: compute average rank per unique value
    unique_vals, first_idx, counts = np.unique(values[order], return_index=True, return_counts=True)
    for start, cnt in zip(first_idx, counts):
        if cnt > 1:
            avg_rank = (start + 1 + start + cnt) / 2.0
            ranks[order[start:start + cnt]] = avg_rank
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xr = _rankdata(np.asarray(x, dtype=float))
    yr = _rankdata(np.asarray(y, dtype=float))
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def roc_auc(labels: List[int], scores: List[float]) -> float:
    """Compute ROC AUC via Mann–Whitney U (no external deps). labels: 1=positive."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve(labels: List[int], scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PR curve points by sorting scores (desc) and accumulating TP/FP.
    labels: 1=positive. scores: higher means more likely positive.
    Returns precision, recall arrays.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0
    fp = 0
    total_pos = int(np.sum(y == 1))
    if total_pos == 0:
        return np.array([np.nan]), np.array([np.nan])
    precisions = []
    recalls = []
    for i in range(y_sorted.size):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / total_pos
        precisions.append(prec)
        recalls.append(rec)
    return np.asarray(precisions), np.asarray(recalls)


def pr_auc(precisions: np.ndarray, recalls: np.ndarray) -> float:
    if len(precisions) < 2 or len(recalls) < 2 or np.any(np.isnan(precisions)):
        return float("nan")
    # Sort by recall ascending for integration
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]
    return float(np.trapz(p, r))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_sd = math.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled_sd == 0:
        return float("nan")
    return mean_diff / pooled_sd


def ks_statistic(group1: List[float], group2: List[float]) -> float:
    a = np.sort(np.asarray(group1, dtype=float))
    b = np.sort(np.asarray(group2, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Evaluate CDFs on the union of unique points
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def risk_coverage_aurrc(labels: List[int], uncertainties: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build Risk-Coverage curve using confidence = -uncertainty.
    coverage k/N vs risk (1 - accuracy among top-k by confidence).
    Returns coverage, risk, and area under the RC curve (AURC) via trapezoidal rule.
    """
    y = np.asarray(labels, dtype=int)
    u = np.asarray(uncertainties, dtype=float)
    if y.size == 0:
        return np.array([np.nan]), np.array([np.nan]), float("nan")
    confidence = -u
    order = np.argsort(-confidence)
    y_sorted = y[order]
    risks = []
    coverages = []
    wrong = 0
    for k in range(1, y_sorted.size + 1):
        if y_sorted[k - 1] == 0:
            wrong += 1
        risk_k = wrong / k
        cov_k = k / y_sorted.size
        risks.append(risk_k)
        coverages.append(cov_k)
    risks = np.asarray(risks, dtype=float)
    coverages = np.asarray(coverages, dtype=float)
    aurc = float(np.trapz(risks, coverages))
    return coverages, risks, aurc


def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between EM and uncertainty for UALA outputs.")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/llama2-hotpotqa-dev-uala-nooracle.jsonl",
        help="Path to JSONL outputs produced by run_hotpotqa_llama2.py in uala mode.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    entries = load_entries(args.input)

    # Collect overall final uncertainties and EMs
    final_uncertainties: List[float] = []
    ems: List[int] = []
    sources: List[str] = []

    # Also collect by source for separate stats
    react_uncs: List[float] = []
    react_ems: List[int] = []
    cot_uncs: List[float] = []
    cot_ems: List[int] = []

    for e in entries:
        em_val = e.get("em")
        if isinstance(em_val, bool):
            em_int = 1 if em_val else 0
        elif isinstance(em_val, (int, float)):
            em_int = 1 if int(em_val) != 0 else 0
        else:
            # skip entries without EM
            continue

        unc, src = extract_final_uncertainty(e)
        if not math.isfinite(unc):
            # skip if we couldn't extract uncertainty
            continue

        final_uncertainties.append(unc)
        ems.append(em_int)
        sources.append(src)

        if src == "react":
            react_uncs.append(unc)
            react_ems.append(em_int)
        elif src == "cot":
            cot_uncs.append(unc)
            cot_ems.append(em_int)

    # Overall correlation (expect negative if higher uncertainty => more errors)
    overall_corr = point_biserial_corr(ems, final_uncertainties)
    overall_spearman = spearman_rho(ems, final_uncertainties)

    # Group statistics
    correct_uncs = [u for u, em in zip(final_uncertainties, ems) if em == 1]
    wrong_uncs = [u for u, em in zip(final_uncertainties, ems) if em == 0]
    d_effect = cohens_d(wrong_uncs, correct_uncs)
    ks = ks_statistic(wrong_uncs, correct_uncs)

    # AUC metrics
    # Treat EM==1 as positive; uncertainty is inversely related to correctness, so use score = -uncertainty
    auc_correct = roc_auc(ems, [-u for u in final_uncertainties])
    prec_c, rec_c = precision_recall_curve(ems, [-u for u in final_uncertainties])
    auprc_correct = pr_auc(prec_c, rec_c)

    # Also report for detecting errors (EM==0 as positive)
    inv_labels = [1 - em for em in ems]
    auc_error = roc_auc(inv_labels, final_uncertainties)
    prec_e, rec_e = precision_recall_curve(inv_labels, final_uncertainties)
    auprc_error = pr_auc(prec_e, rec_e)

    # Risk-Coverage
    cov, risk, aurc = risk_coverage_aurrc(ems, final_uncertainties)

    print("=== UALA: EM と最終不確実性の相関解析 ===")
    print(f"入力: {args.input}")
    print(f"サンプル数: {len(final_uncertainties)} (ReAct由来: {sources.count('react')}, CoTのみ: {sources.count('cot')})")
    print("")
    print(f"相関係数（ポイント・バイセリアル; EM(1/0) vs 不確実性）: {overall_corr:.4f}")
    print(f"スピアマンρ（EM(1/0) vs 不確実性）: {overall_spearman:.4f}")
    print("")
    print("[全体 不確実性要約]")
    print(json.dumps(summarize(final_uncertainties), ensure_ascii=False, indent=2))
    print("")
    print("[正解(EM=1)側 不確実性要約]")
    print(json.dumps(summarize(correct_uncs), ensure_ascii=False, indent=2))
    print("")
    print("[不正解(EM=0)側 不確実性要約]")
    print(json.dumps(summarize(wrong_uncs), ensure_ascii=False, indent=2))
    print("")
    print("[分布差の指標]")
    print(f"Cohen's d (EM=0 vs EM=1 の不確実性): {d_effect:.4f}")
    print(f"KS統計量 (EM=0 vs EM=1 の不確実性CDF差の最大): {ks:.4f}")
    print("")
    print("[AUC系の指標]")
    print(f"AUROC (正解検出; スコア=-不確実性): {auc_correct:.4f}")
    print(f"AUPRC (正解検出; スコア=-不確実性): {auprc_correct:.4f}")
    print(f"AUROC (誤り検出; スコア=不確実性): {auc_error:.4f}")
    print(f"AUPRC (誤り検出; スコア=不確実性): {auprc_error:.4f}")
    print("")
    print("[Risk-Coverage]")
    print(f"AURC: {aurc:.4f}  (小さいほど良い; 低リスクで広くカバー)")

    # Separate stats per source for additional insight
    if react_uncs:
        corr_react = point_biserial_corr(react_ems, react_uncs)
        print("[ReActのみ 最終不確実性 vs EM]")
        print(f"相関係数: {corr_react:.4f}")
        print(json.dumps(summarize(react_uncs), ensure_ascii=False, indent=2))
        print("")

    if cot_uncs:
        corr_cot = point_biserial_corr(cot_ems, cot_uncs)
        print("[CoTのみ 最終不確実性 vs EM]")
        print(f"相関係数: {corr_cot:.4f}")
        print(json.dumps(summarize(cot_uncs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



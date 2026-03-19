from __future__ import annotations

import ast
from collections import Counter
import os
from pathlib import Path
import re
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import ConstantInputWarning, mannwhitneyu, pearsonr, spearmanr
from tqdm import tqdm

from ahmm_plotting import show_pv_representation_ahmm_dataset
from ahmm_utils import (
    SequenceDataset,
    decode_posteriors_filtered,
    load_ahmm,
    make_session_lookup,
    records_to_dataset,
    train_test_split_by_sequence,
)


def resolve_data_path(path: str | os.PathLike, *, data_roots: tuple[str | os.PathLike, ...] = ("data", "demo_data", ".", "models")) -> Path:
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        for root in data_roots:
            candidates.append(Path(root) / raw)
            candidates.append((Path(root) / raw.name))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve data path: {path}")


def resolve_existing_path(path: str | os.PathLike, *, search_roots: tuple[str | os.PathLike, ...] = ("models", ".", "data", "demo_data")) -> str:
    raw = Path(str(path)).expanduser()
    candidate_list = []
    if raw.is_absolute():
        candidate_list.append(raw)
    else:
        candidate_list.append(Path.cwd() / raw)
        for root in search_roots:
            root_path = Path(root)
            candidate_list.append(root_path / raw)
            candidate_list.append(root_path / raw.name)

    raw_str = str(raw).replace("\\", "/")
    anchors = [
        "ahmm_models_per_session_all/",
        "random_train_val_split/",
        "25/",
        "30/",
        "35/",
        "40/",
        "50/",
    ]
    for anchor in anchors:
        if anchor in raw_str:
            suffix = raw_str.split(anchor, 1)[1]
            for root in search_roots:
                root_path = Path(root)
                candidate_list.append(root_path / anchor.rstrip("/") / suffix)
                candidate_list.append(root_path / suffix)
                if anchor == "ahmm_models_per_session_all/":
                    candidate_list.append(root_path / "models" / suffix)
                if "random_train_val_split/" in raw_str:
                    candidate_list.append(root_path / "models" / suffix)

    seen = set()
    for candidate in candidate_list:
        norm = str(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if candidate.exists():
            return str(candidate.resolve())

    basename = raw.name
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        hits = list(root_path.rglob(basename))
        if hits:
            return str(hits[0].resolve())

    return normalize_path(str(path))


def repair_model_paths(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ("save_path", "model_path"),
    search_roots: tuple[str | os.PathLike, ...] = ("models", ".", "data", "demo_data"),
) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda path: resolve_existing_path(path, search_roots=search_roots))
    return out


def _compute_nll(model, obs, acts, *, eps: float = 1e-12, use_action_as_evidence: bool = True) -> float:
    obs = np.asarray(obs, dtype=int)
    acts = np.asarray(acts, dtype=int)
    if len(obs) == 0:
        return 0.0

    logpi = np.log(np.asarray(model.pi, float) + eps)
    logT = np.log(np.asarray(model.trans, float) + eps)
    logE = np.log(np.asarray(model.emit, float) + eps)
    logPi = np.log(np.asarray(model.policy, float) + eps)

    n_states = logpi.shape[0]
    log_alpha = np.full((len(obs), n_states), -np.inf)
    log_alpha[0] = logpi + logE[:, obs[0]]
    if use_action_as_evidence:
        log_alpha[0] += logPi[:, acts[0]]

    for t in range(1, len(obs)):
        trans_term = np.logaddexp.reduce(log_alpha[t - 1][:, None] + logT[int(acts[t - 1])], axis=0)
        log_alpha[t] = trans_term + logE[:, int(obs[t])]
        if use_action_as_evidence:
            log_alpha[t] += logPi[:, int(acts[t])]

    return -float(np.logaddexp.reduce(log_alpha[-1]))


def compute_nll_any(
    model,
    *,
    dataset: SequenceDataset,
    mean: bool = False,
    use_action_as_evidence: bool = True,
    eps: float = 1e-12,
) -> float:
    total_nll = 0.0
    total_steps = 0
    start = 0

    for length in np.asarray(dataset.lengths, dtype=int):
        end = start + int(length)
        total_nll += _compute_nll(
            model,
            dataset.obs[start:end],
            dataset.act[start:end],
            eps=eps,
            use_action_as_evidence=use_action_as_evidence,
        )
        total_steps += int(length)
        start = end

    return total_nll / max(total_steps, 1) if mean else total_nll


def nll_null_model(states, actions) -> float:
    states = np.asarray(states)
    actions = np.asarray(actions)
    if len(states) == 0:
        return 0.0

    pairs = list(zip(states.tolist(), actions.tolist()))
    state_set = sorted(set(states.tolist()))
    action_set = sorted(set(actions.tolist()))
    counts = Counter(pairs)
    probs = {(s, a): counts[(s, a)] / len(pairs) for s in state_set for a in action_set}
    return -float(sum(np.log(probs[(s, a)]) for s, a in pairs))


def get_pde(nll_model_test: float, nll_null_test: float) -> float:
    return 1.0 - (nll_model_test / nll_null_test)


def compare_dist_matrices(D1, D2, method: str = "spearman", on_constant: str = "nan", jitter: float = 1e-9) -> float:
    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    if D1.shape != D2.shape:
        raise ValueError("Distance matrices must have same shape")

    i, j = np.triu_indices_from(D1, k=1)
    a, b = D1[i, j], D2[i, j]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return np.nan

    const_a = np.ptp(a) == 0.0
    const_b = np.ptp(b) == 0.0
    if const_a or const_b:
        if on_constant == "nan":
            return np.nan
        if on_constant == "zero":
            return 0.0
        if on_constant == "skip":
            return np.nan
        if on_constant == "jitter":
            scale_a = max(np.abs(a).max(), 1.0)
            scale_b = max(np.abs(b).max(), 1.0)
            rng = np.random.default_rng(0)
            a = a + rng.normal(0, jitter * scale_a, size=a.shape)
            b = b + rng.normal(0, jitter * scale_b, size=b.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        if method == "spearman":
            r, _ = spearmanr(a, b)
        elif method == "pearson":
            r, _ = pearsonr(a, b)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
    return float(r)


def build_optimal_height_distance_matrix(
    h_min: int = 200,
    h_max: int = 650,
    step: int = 50,
    as_dataframe: bool = True,
):
    heights = np.arange(h_min, h_max + step, step)
    matrix = np.abs(heights[:, None] - heights[None, :])
    if as_dataframe:
        labels = [str(h) for h in heights]
        return pd.DataFrame(matrix, index=labels, columns=labels)
    return matrix, heights


def cov_to_1_minus_corr_norm(Sigma, *, eps: float = 1e-12):
    Sigma = np.asarray(Sigma, dtype=float)
    var = np.diag(Sigma)
    std = np.sqrt(np.maximum(var, 0.0))
    denom = np.outer(std, std)

    corr = np.full_like(Sigma, np.nan, dtype=float)
    valid = denom > eps
    corr[valid] = Sigma[valid] / denom[valid]
    corr = np.clip(corr, -1.0, 1.0)

    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    max_val = np.nanmax(dist)
    if np.isfinite(max_val) and max_val > eps:
        dist = dist / max_val
    return dist


def load_neural_covariances(mat_path: str = "neural_covariances.mat") -> dict[tuple[str, str], np.ndarray]:
    mat = sio.loadmat(resolve_data_path(mat_path), simplify_cells=True)
    session_meta = mat["new_sessions"]
    cells = mat["new_ses_dist_mat"].ravel()
    neural_cov = np.stack(cells, axis=2)

    all_neural = {}
    for i in range(neural_cov.shape[2]):
        animal = str(session_meta[i][0])
        date = pd.to_datetime(str(int(session_meta[i][1])), format="%Y%m%d").strftime("%Y-%m-%d")
        all_neural[(animal, date)] = cov_to_1_minus_corr_norm(neural_cov[:, :, i].astype(float, copy=False))
    return all_neural


def build_model_x_neural_similarity_df(
    df: pd.DataFrame,
    all_neural: dict[tuple[str, str], np.ndarray],
    *,
    col: str = "rep_matrix",
    compare_metric: str = "pearson",
    on_constant: str = "skip",
) -> pd.DataFrame:
    rows = []
    neural_items = list(all_neural.items())

    for _, row in tqdm(list(df.iterrows()), total=len(df)):
        rep = row.get(col, None)
        if rep is None:
            continue
        rep = np.asarray(rep)
        train_date = pd.to_datetime(row["train_date"]).strftime("%Y-%m-%d")

        for (test_animal, test_date), neural in neural_items:
            if rep.shape != np.asarray(neural).shape:
                similarity = np.nan
            else:
                similarity = compare_dist_matrices(
                    neural,
                    rep,
                    method=compare_metric,
                    on_constant=on_constant,
                )
                similarity = float(similarity) if np.isfinite(similarity) else np.nan

            rows.append(
                {
                    "train_animal": row["train_animal"],
                    "train_date": train_date,
                    "seed": row.get("seed", np.nan),
                    "pde": row.get("pde", np.nan),
                    "pde_rank": row.get("pde_rank", np.nan),
                    "test_animal": test_animal,
                    "test_date": test_date,
                    "similarity": similarity,
                    "rep_matrix": rep,
                }
            )

    return pd.DataFrame(rows)


def pick_nth_by_similarity_within_topk_pde(sim_df: pd.DataFrame, k: int, n: int = 1) -> pd.DataFrame:
    picked = sim_df.loc[sim_df["pde_rank"] <= k].copy()
    picked["sim_rank_within_k"] = (
        picked.groupby(["train_animal", "train_date", "test_animal", "test_date"])["similarity"]
        .rank(method="first", ascending=False)
    )
    return picked.loc[picked["sim_rank_within_k"] == n, [
        "train_animal", "train_date", "test_animal", "test_date",
        "similarity", "pde_rank", "sim_rank", "sim_rank_within_k",
    ]].rename(columns={"similarity": "similarity_pick"})


def pick_nth_by_similarity_within_k_range_pde(sim_df: pd.DataFrame, k1: int, k2: int, n: int = 1) -> pd.DataFrame:
    picked = sim_df.loc[(sim_df["pde_rank"] >= k1) & (sim_df["pde_rank"] <= k2)].copy()
    picked["sim_rank_within_k"] = (
        picked.groupby(["train_animal", "train_date", "test_animal", "test_date"])["similarity"]
        .rank(method="average", ascending=False)
    )
    return picked.loc[picked["sim_rank_within_k"] == n, [
        "train_animal", "train_date", "test_animal", "test_date", "seed",
        "similarity", "pde_rank", "sim_rank", "sim_rank_within_k",
    ]].rename(columns={"similarity": "similarity_pick"})


def within_between_rank_sum_on_ranks(sim_df: pd.DataFrame, k: int, n: int = 1, *, alternative: str = "two-sided") -> dict:
    picked = pick_nth_by_similarity_within_topk_pde(sim_df, k=k, n=n).copy()
    picked["train_key"] = list(zip(picked["train_animal"], picked["train_date"]))
    picked["test_key"] = list(zip(picked["test_animal"], picked["test_date"]))
    picked = picked[np.isfinite(picked["similarity_pick"])].copy()
    picked["rank_in_test"] = picked.groupby("test_key")["similarity_pick"].rank(method="average", ascending=False)

    within_ranks = picked.loc[picked["train_key"] == picked["test_key"], "rank_in_test"].to_numpy(float)
    between_ranks = picked.loc[picked["train_key"] != picked["test_key"], "rank_in_test"].to_numpy(float)
    if within_ranks.size < 5 or between_ranks.size < 5:
        return dict(k=k, n=n, n_within=int(within_ranks.size), n_between=int(between_ranks.size), stat=np.nan, p=np.nan)

    stat, p = mannwhitneyu(within_ranks, between_ranks, alternative=alternative)
    return dict(k=k, n=n, n_within=int(within_ranks.size), n_between=int(between_ranks.size), stat=float(stat), p=float(p))


def within_between_similarity_on_ranks(sim_df: pd.DataFrame, k: int, n: int = 1, *, alternative: str = "two-sided") -> dict:
    picked = pick_nth_by_similarity_within_topk_pde(sim_df, k=k, n=n).copy()
    picked["train_key"] = list(zip(picked["train_animal"], picked["train_date"]))
    picked["test_key"] = list(zip(picked["test_animal"], picked["test_date"]))
    picked = picked[np.isfinite(picked["similarity_pick"])].copy()

    within_scores = picked.loc[picked["train_key"] == picked["test_key"], "similarity_pick"].to_numpy(float)
    between_scores = picked.loc[picked["train_key"] != picked["test_key"], "similarity_pick"].to_numpy(float)
    if within_scores.size < 5 or between_scores.size < 5:
        return dict(k=k, n=n, n_within=int(within_scores.size), n_between=int(between_scores.size), stat=np.nan, p=np.nan)

    stat, p = mannwhitneyu(within_scores, between_scores, alternative=alternative)
    return dict(k=k, n=n, n_within=int(within_scores.size), n_between=int(between_scores.size), stat=float(stat), p=float(p))


def within_between_rank_sum_on_ranks_in_range(
    sim_df: pd.DataFrame,
    k1: int,
    k2: int,
    n: int = 1,
    *,
    alternative: str = "two-sided",
) -> dict:
    picked = pick_nth_by_similarity_within_k_range_pde(sim_df, k1=k1, k2=k2, n=n).copy()
    picked["train_key"] = list(zip(picked["train_animal"], picked["train_date"]))
    picked["test_key"] = list(zip(picked["test_animal"], picked["test_date"]))
    picked = picked[np.isfinite(picked["similarity_pick"])].copy()
    picked["rank_in_test"] = picked.groupby("test_key")["similarity_pick"].rank(method="average", ascending=False)

    within_ranks = picked.loc[picked["train_key"] == picked["test_key"], "rank_in_test"].to_numpy(float)
    between_ranks = picked.loc[picked["train_key"] != picked["test_key"], "rank_in_test"].to_numpy(float)
    if within_ranks.size < 5 or between_ranks.size < 5:
        return dict(k1=k1, k2=k2, n=n, n_within=int(within_ranks.size), n_between=int(between_ranks.size), stat=np.nan, p=np.nan)

    stat, p = mannwhitneyu(within_ranks, between_ranks, alternative=alternative)
    return dict(k1=k1, k2=k2, n=n, n_within=int(within_ranks.size), n_between=int(between_ranks.size), stat=float(stat), p=float(p))


def sweep_k1_k2(sim_df: pd.DataFrame, k1s, k2s, alternative: str = "less") -> pd.DataFrame:
    records = []
    for k1 in k1s:
        for k2 in k2s:
            records.append(within_between_rank_sum_on_ranks_in_range(sim_df, int(k1), int(k2), n=1, alternative=alternative))
    return pd.DataFrame(records)


def gather_ranked_violin_df_with_optimal(
    sim_df: pd.DataFrame,
    k1: int,
    k2: int,
    n: int = 1,
    *,
    optimal_mat,
    all_neural: dict[tuple[str, str], np.ndarray],
    compare_metric: str = "pearson",
    on_constant: str = "skip",
    ascending: bool = True,
) -> pd.DataFrame:
    picked = pick_nth_by_similarity_within_k_range_pde(sim_df, k1=k1, k2=k2, n=n).copy()
    picked = picked[np.isfinite(picked["similarity_pick"])].copy()
    picked["train_session_key"] = list(zip(picked["train_animal"], picked["train_date"]))
    picked["test_key"] = list(zip(picked["test_animal"], picked["test_date"]))
    picked["group"] = np.where(picked["train_session_key"] == picked["test_key"], "AA", "BA")
    picked["model_key"] = list(zip(picked["train_animal"], picked["train_date"], picked["seed"]))
    picked = picked.rename(columns={"similarity_pick": "similarity"})
    picked = picked[["test_key", "train_session_key", "model_key", "group", "similarity"]]

    opt_rows = []
    optimal_mat = np.asarray(optimal_mat)
    for animal, date in picked["test_key"].drop_duplicates().tolist():
        key = (animal, pd.to_datetime(date).strftime("%Y-%m-%d"))
        if key not in all_neural:
            continue
        neural = np.asarray(all_neural[key])
        if optimal_mat.shape != neural.shape:
            continue
        similarity = compare_dist_matrices(
            optimal_mat,
            neural,
            method=compare_metric,
            on_constant=on_constant,
        )
        if not np.isfinite(similarity):
            continue
        opt_rows.append(
            {
                "test_key": (animal, date),
                "model_key": ("OPT", "OPT"),
                "group": "OPT",
                "similarity": float(similarity),
            }
        )

    combined = pd.concat([picked, pd.DataFrame(opt_rows)], ignore_index=True)
    combined = combined[np.isfinite(combined["similarity"])].copy()
    combined["rank_in_test"] = combined.groupby("test_key")["similarity"].rank(method="average", ascending=ascending)
    return combined


def build_df_all_for_violin_both(
    sim_df: pd.DataFrame,
    k1: int,
    k2: int,
    *,
    optimal_mat,
    all_neural: dict[tuple[str, str], np.ndarray],
    n: int = 1,
    compare_metric: str = "pearson",
    on_constant: str = "skip",
    ascending: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfv = gather_ranked_violin_df_with_optimal(
        sim_df,
        k1,
        k2,
        n=n,
        optimal_mat=optimal_mat,
        all_neural=all_neural,
        compare_metric=compare_metric,
        on_constant=on_constant,
        ascending=ascending,
    )
    keep = dfv[dfv["group"].isin(["AA", "BA", "OPT"])].copy()
    df_all = pd.DataFrame(
        {
            "group": keep["group"].map(
                {"AA": "Within-session", "BA": "Across-session", "OPT": "Optimal-model"}
            ).to_numpy(),
            "similarity": keep["similarity"].to_numpy(float),
            "rank": keep["rank_in_test"].to_numpy(float),
        }
    )
    return df_all.dropna(subset=["similarity", "rank"]).reset_index(drop=True), dfv


def plot_violin_with_pbars(df_all: pd.DataFrame, metric: str, *, ascending: bool = True):
    order = ["Within-session", "Across-session", "Optimal-model"]
    palette = {
        "Within-session": "#93eadb",
        "Across-session": "#88c6f2",
        "Optimal-model": "#ebcfff",
    }

    fig, ax = plt.subplots(figsize=(3.4, 5.4))
    sns.violinplot(
        data=df_all,
        x="group",
        y=metric,
        order=order,
        hue="group",
        palette=palette,
        dodge=False,
        inner="box",
        density_norm="width",
        bw_adjust=1.2,
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xlabel("")
    alt = "greater" if ascending else "less"
    ax.set_ylabel("Rank within test session (100 = best)" if ascending else "Rank within test session (1 = best)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    sns.despine(ax=ax)
    ax.grid(False)

    def add_pbar(x1, x2, y, text, color="black"):
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color=color, linewidth=1.0)
        ax.text((x1 + x2) / 2, y + 0.025, text, ha="center", va="bottom", fontsize=8, color=color)

    y_min = np.nanmin(df_all[metric].to_numpy())
    y_max = np.nanmax(df_all[metric].to_numpy())
    y_range = y_max - y_min if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0
    y0 = y_max + 0.08 * y_range
    dy = 0.10 * y_range

    g_within = df_all.loc[df_all["group"] == "Within-session", metric].to_numpy(float)
    g_across = df_all.loc[df_all["group"] == "Across-session", metric].to_numpy(float)
    g_opt = df_all.loc[df_all["group"] == "Optimal-model", metric].to_numpy(float)

    if g_within.size and g_across.size:
        _, p = mannwhitneyu(g_within, g_across, alternative=alt)
        add_pbar(0, 1, y0, f"P={p:.1e}", color="red" if p < 5e-2 else "black")
    if g_within.size and g_opt.size:
        _, p = mannwhitneyu(g_within, g_opt, alternative=alt)
        add_pbar(0, 2, y0 + dy, f"P={p:.1e}", color="red" if p < 5e-2 else "black")
    if g_across.size and g_opt.size:
        _, p = mannwhitneyu(g_across, g_opt, alternative=alt)
        add_pbar(1, 2, y0 + 2 * dy, f"P={p:.1e}", color="red" if p < 5e-2 else "black")

    plt.tight_layout()
    plt.show()
    return fig


def add_within_between_groups(
    cg_df: pd.DataFrame,
    *,
    group_col: str = "group",
    within_label: str = "Within-session",
    between_label: str = "Across-session",
) -> pd.DataFrame:
    df = cg_df.copy()
    df["train_date"] = pd.to_datetime(df["train_date"]).dt.strftime("%Y-%m-%d")
    df["test_date"] = pd.to_datetime(df["test_date"]).dt.strftime("%Y-%m-%d")
    is_within = (df["train_animal"] == df["test_animal"]) & (df["train_date"] == df["test_date"])
    df[group_col] = np.where(is_within, within_label, between_label)
    return df


def plot_violin_within_between(
    cg_df: pd.DataFrame,
    metric: str,
    *,
    ascending: bool = True,
    group_col: str = "group",
    within_label: str = "Within-session",
    between_label: str = "Across-session",
    palette: dict | None = None,
    y_legend: str = "Behavior Correlation",
) -> pd.DataFrame:
    df = add_within_between_groups(
        cg_df,
        group_col=group_col,
        within_label=within_label,
        between_label=between_label,
    )

    order = [within_label, between_label]
    if palette is None:
        palette = {within_label: "#93eadb", between_label: "#88c6f2"}

    fig, ax = plt.subplots(figsize=(3.1, 5.0))
    sns.violinplot(
        data=df,
        x=group_col,
        y=metric,
        order=order,
        hue=group_col,
        palette=palette,
        dodge=False,
        inner="box",
        density_norm="width",
        bw_adjust=1.2,
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(y_legend)
    alt = "greater" if ascending else "less"
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    sns.despine(ax=ax)
    ax.grid(False)

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    def add_pbar(x1, x2, y, text, color="black"):
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color=color, linewidth=1.0)
        ax.text((x1 + x2) / 2, y + 0.025, text, ha="center", va="bottom", fontsize=8, color=color)

    values = df[metric].to_numpy(float)
    y_min = np.nanmin(values)
    y_max = np.nanmax(values)
    y_range = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0
    y0 = y_max + 0.08 * y_range

    within_vals = df.loc[df[group_col] == within_label, metric].to_numpy(float)
    between_vals = df.loc[df[group_col] == between_label, metric].to_numpy(float)
    if within_vals.size and between_vals.size:
        _, p = mannwhitneyu(within_vals, between_vals, alternative=alt)
        add_pbar(0, 1, y0, f"P={p:.1e}", color="red" if p < 5e-2 else "black")

    plt.tight_layout()
    plt.show()
    return df


def _stack_mats(mats, *, name: str = "mats"):
    if len(mats) == 0:
        return np.zeros((0, 0, 0), dtype=float)
    mats = [np.asarray(m, float) for m in mats]
    height, width = mats[0].shape
    for i, mat in enumerate(mats):
        if mat.shape != (height, width):
            raise ValueError(f"{name}[{i}] has shape {mat.shape}, expected {(height, width)}")
    return np.stack(mats, axis=2)


def save_violin_reps_to_mat_stacked(
    sim_df: pd.DataFrame,
    *,
    k1: int,
    k2: int,
    n: int,
    optimal_mat,
    all_neural: dict[tuple[str, str], np.ndarray],
    out_path: str,
    compare_metric: str = "pearson",
    on_constant: str = "skip",
    ascending: bool = True,
    eps_const: float = 1e-12,
    drop_constant: bool = True,
) -> None:
    def is_constant(mat) -> bool:
        mat = np.asarray(mat, float)
        if mat.size == 0:
            return True
        if not np.isfinite(mat).all():
            return True
        return np.nanstd(mat) < eps_const

    dfv = gather_ranked_violin_df_with_optimal(
        sim_df,
        k1,
        k2,
        n=n,
        optimal_mat=optimal_mat,
        all_neural=all_neural,
        compare_metric=compare_metric,
        on_constant=on_constant,
        ascending=ascending,
    )

    if "model_key" in dfv.columns:
        dfv_key_col = "model_key"
    elif "train_key" in dfv.columns:
        dfv_key_col = "train_key"
    else:
        raise KeyError("dfv must contain 'model_key' or 'train_key' to identify selected models.")

    data = sim_df.copy().dropna(subset=["rep_matrix"]).copy()
    has_seed = "seed" in data.columns
    if has_seed:
        data["model_key"] = list(zip(data["train_animal"], data["train_date"], data["seed"]))
    data["train_key"] = list(zip(data["train_animal"], data["train_date"]))
    data["test_key"] = list(zip(data["test_animal"], data["test_date"]))

    if drop_constant:
        data = data[~data["rep_matrix"].apply(is_constant)].copy()

    example_key = dfv[dfv_key_col].dropna().iloc[0] if len(dfv) else None
    dfv_keys_are_seeded = isinstance(example_key, tuple) and len(example_key) == 3

    if dfv_keys_are_seeded:
        if not has_seed:
            raise KeyError("dfv uses seeded keys (len==3) but sim_df has no 'seed' column.")
        rep_lookup = data.set_index(["model_key", "test_key"])["rep_matrix"].to_dict()
        use_seeded_lookup = True
    else:
        rep_lookup = data.set_index(["train_key", "test_key"])["rep_matrix"].to_dict()
        use_seeded_lookup = False

    aa_reps, ba_reps = [], []
    aa_meta, ba_meta = [], []
    for _, row in dfv.iterrows():
        if row["group"] == "OPT":
            continue
        test_key = row["test_key"]

        if use_seeded_lookup:
            model_key = row[dfv_key_col]
            rep = rep_lookup.get((model_key, test_key), None)
            if rep is None:
                continue
            meta = {
                "train_animal": str(model_key[0]),
                "train_date": str(model_key[1]),
                "seed": str(model_key[2]),
                "test_animal": str(test_key[0]),
                "test_date": str(test_key[1]),
            }
        else:
            train_key = row[dfv_key_col]
            rep = rep_lookup.get((train_key, test_key), None)
            if rep is None:
                continue
            meta = {
                "train_animal": str(train_key[0]),
                "train_date": str(train_key[1]),
                "seed": "",
                "test_animal": str(test_key[0]),
                "test_date": str(test_key[1]),
            }

        if row["group"] == "AA":
            aa_reps.append(rep)
            aa_meta.append(meta)
        else:
            ba_reps.append(rep)
            ba_meta.append(meta)

    def meta_struct(meta_list):
        if len(meta_list) == 0:
            return {
                "train_animal": np.array([], dtype=object),
                "train_date": np.array([], dtype=object),
                "seed": np.array([], dtype=object),
                "test_animal": np.array([], dtype=object),
                "test_date": np.array([], dtype=object),
            }
        return {
            "train_animal": np.array([m["train_animal"] for m in meta_list], dtype=object),
            "train_date": np.array([m["train_date"] for m in meta_list], dtype=object),
            "seed": np.array([m["seed"] for m in meta_list], dtype=object),
            "test_animal": np.array([m["test_animal"] for m in meta_list], dtype=object),
            "test_date": np.array([m["test_date"] for m in meta_list], dtype=object),
        }

    sio.savemat(
        out_path,
        {
            "AA_rep": _stack_mats(aa_reps, name="AA_reps"),
            "BA_rep": _stack_mats(ba_reps, name="BA_reps"),
            "OPT_rep": np.asarray(optimal_mat, float),
            "AA_meta": meta_struct(aa_meta),
            "BA_meta": meta_struct(ba_meta),
            "params": {
                "k1": int(k1),
                "k2": int(k2),
                "n": int(n),
                "ascending": bool(ascending),
                "compare_metric": str(compare_metric),
                "on_constant": str(on_constant),
                "use_seeded_lookup": bool(use_seeded_lookup),
                "drop_constant": bool(drop_constant),
                "eps_const": float(eps_const),
            },
        },
        do_compression=True,
    )


def date_iso(x) -> str:
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def select_rank1_pde_models(df: pd.DataFrame) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    ranked = df.copy()
    ranked["train_date"] = ranked["train_date"].map(date_iso)
    best = ranked[ranked["pde_rank"] == 1].copy()
    best = best.sort_values(["train_animal", "train_date", "seed"]).drop_duplicates(["train_animal", "train_date"], keep="first")
    best["model_path"] = best["save_path"].astype(str).str.replace("\\", "/", regex=False)
    best_model_by_session = {
        (row.train_animal, row.train_date): row.model_path
        for row in best.itertuples(index=False)
    }
    return best_model_by_session, best


def _date_iso(x) -> str:
    return date_iso(x)


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def parse_model_key(model_key) -> tuple[str, str, int]:
    if isinstance(model_key, str):
        try:
            model_key = ast.literal_eval(model_key)
        except Exception:
            pass

    if isinstance(model_key, (tuple, list)) and len(model_key) == 3:
        animal, date, seed = model_key
    elif isinstance(model_key, (tuple, list)) and len(model_key) == 2:
        (animal, date), seed = model_key
    else:
        raise ValueError(f"Unexpected model_key format: {model_key!r}")

    return str(animal), date_iso(date), int(seed)


def build_model_path_lookup_from_dfv(
    dfv_rand: pd.DataFrame,
    df_models: pd.DataFrame,
    *,
    group_filter=("AA", "BA"),
    model_key_col: str = "model_key",
    group_col: str = "group",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    seed_col: str = "seed",
    path_col: str = "save_path",
) -> tuple[dict[tuple[str, str], list[str]], list[tuple[str, str, int]]]:
    dfv = dfv_rand.copy()
    if group_filter is not None:
        dfv = dfv[dfv[group_col].isin(group_filter)].copy()

    used_keys = {parse_model_key(model_key) for model_key in dfv[model_key_col]}

    dm = df_models.copy()
    dm[train_date_col] = dm[train_date_col].map(date_iso)
    dm[path_col] = dm[path_col].map(normalize_path)
    path_lookup = {
        (str(row[train_animal_col]), str(row[train_date_col]), int(row[seed_col])): row[path_col]
        for _, row in dm.iterrows()
    }

    lookup: dict[tuple[str, str], list[str]] = {}
    missing: list[tuple[str, str, int]] = []
    for animal, date, seed in used_keys:
        path = path_lookup.get((animal, date, seed))
        if path is None:
            missing.append((animal, date, seed))
            continue
        lookup.setdefault((animal, date), []).append(path)

    for key in lookup:
        lookup[key] = sorted(lookup[key])
    return lookup, missing


def sample_ahmm(model, trial_len: int, n_trials: int, *, seed=None, s0=None, gamma0=None):
    return sample_ahmm_nonstream(model, trial_len=trial_len, n_trials=n_trials, seed=seed, s0=s0, gamma0=gamma0)


def get_best_model_row_for_session(
    df_models: pd.DataFrame,
    animal,
    date,
    *,
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    pde_col: str = "pde",
    path_col: str = "save_path",
) -> pd.Series:
    date_str = date_iso(date)

    dm = df_models.copy()
    dm[train_animal_col] = dm[train_animal_col].astype(str)
    dm[train_date_col] = dm[train_date_col].map(date_iso)
    dm[pde_col] = pd.to_numeric(dm[pde_col], errors="coerce")
    dm[path_col] = dm[path_col].map(normalize_path)

    sub = dm[(dm[train_animal_col] == str(animal)) & (dm[train_date_col] == date_str)]
    if sub.empty:
        raise KeyError(f"No rows in df_models for session {animal} {date_str}")
    return dm.loc[sub[pde_col].idxmax()]


def load_best_pde_model_for_session(
    df_models: pd.DataFrame,
    animal,
    date,
    *,
    load_fn=load_ahmm,
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    pde_col: str = "pde",
    path_col: str = "save_path",
):
    best_row = get_best_model_row_for_session(
        df_models,
        animal,
        date,
        train_animal_col=train_animal_col,
        train_date_col=train_date_col,
        pde_col=pde_col,
        path_col=path_col,
    )

    model_path = str(best_row[path_col])
    model_path = resolve_existing_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model path not found:\n{model_path}")
    model, meta = load_fn(model_path)
    return model, meta, best_row


def load_within_session_model_for_session(
    dfv: pd.DataFrame,
    df_models: pd.DataFrame,
    animal,
    date,
    *,
    load_fn=load_ahmm,
    group_col: str = "group",
    model_key_col: str = "model_key",
    test_key_col: str = "test_key",
    train_session_key_col: str = "train_session_key",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    seed_col: str = "seed",
    path_col: str = "save_path",
    allow_groups=("Within-session", "AA"),
    prefer_group_order=("Within-session", "AA", "BA"),
):
    animal = str(animal)
    date_str = date_iso(date)
    dv = dfv.copy()

    if test_key_col in dv.columns:
        dv[test_key_col] = dv[test_key_col].apply(lambda x: tuple(x) if isinstance(x, (list, tuple)) else x)
    if train_session_key_col in dv.columns:
        dv[train_session_key_col] = dv[train_session_key_col].apply(lambda x: tuple(x) if isinstance(x, (list, tuple)) else x)

    sub = pd.DataFrame()
    if test_key_col in dv.columns:
        sub = dv[dv[test_key_col] == (animal, date_str)]
    if sub.empty and train_session_key_col in dv.columns:
        sub = dv[dv[train_session_key_col] == (animal, date_str)]
    if sub.empty:
        raise KeyError(f"No rows in dfv for session {animal} {date_str}")

    if group_col in sub.columns and allow_groups is not None:
        sub = sub[sub[group_col].isin(allow_groups)]
    if sub.empty:
        raise KeyError(f"No within-session rows in dfv for session {animal} {date_str} (allowed groups={allow_groups})")

    if group_col in sub.columns and prefer_group_order is not None:
        group_rank = {group: idx for idx, group in enumerate(prefer_group_order)}
        sub = sub.assign(_g_rank=sub[group_col].map(lambda group: group_rank.get(group, 9999))).sort_values("_g_rank")

    row_v = sub.iloc[0]
    train_animal, train_date, seed = parse_model_key(row_v[model_key_col])

    dm = df_models.copy()
    dm[train_animal_col] = dm[train_animal_col].astype(str)
    dm[train_date_col] = dm[train_date_col].map(date_iso)
    dm[seed_col] = pd.to_numeric(dm[seed_col], errors="coerce").astype("Int64")
    dm[path_col] = dm[path_col].map(normalize_path)

    hit = dm[
        (dm[train_animal_col] == train_animal)
        & (dm[train_date_col] == train_date)
        & (dm[seed_col] == seed)
    ]
    if hit.empty:
        raise KeyError(f"Model from dfv not found in df_models: ({train_animal}, {train_date}, seed={seed})")

    row_model = hit.iloc[0]
    model_path = str(row_model[path_col])
    model_path = resolve_existing_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Within-session model path not found:\n{model_path}")

    model, meta = load_fn(model_path)
    return model, meta, row_v, row_model


def _height_numbers(vocab, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{prefix}_(\d+)$", re.IGNORECASE)
    return sorted({int(match.group(1)) for token in vocab.obs_to_id for match in [pattern.match(token)] if match})


def _parse_height(token: str, side: str) -> int | None:
    match = re.match(rf"(?i)^{side}_(\d+)$", str(token).strip())
    return int(match.group(1)) if match else None


def compute_state_tuning_all(model, train, vocab, stream: bool = False):
    del stream
    n_states, n_obs = model.emit.shape
    left_labels = [f"l_{h}" for h in _height_numbers(vocab, "l")]
    right_labels = [f"r_{h}" for h in _height_numbers(vocab, "r")]
    height_labels = left_labels + right_labels
    height_ids = np.array([vocab.obs_to_id[label] for label in height_labels], dtype=int)

    tuning = np.zeros((n_states, n_obs), float)
    counts = np.zeros(n_obs, float)
    gammas = decode_posteriors_filtered(train, model)

    start = 0
    for trial_gamma, length in zip(gammas, np.asarray(train.lengths, dtype=int)):
        trial_obs = train.obs[start:start + length]
        for step, gamma_t in enumerate(trial_gamma):
            obs_id = int(trial_obs[step])
            counts[obs_id] += 1
            tuning[:, obs_id] += gamma_t
        start += length

    nonzero = counts > 0
    tuning[:, nonzero] /= counts[nonzero]
    return tuning[:, height_ids], counts[height_ids], height_labels


def compute_state_tuning_all_obs(model, train, vocab):
    n_states, n_obs = model.emit.shape

    left_labels = [f"l_{h}" for h in _height_numbers(vocab, "l")]
    right_labels = [f"r_{h}" for h in _height_numbers(vocab, "r")]
    obs_labels = ["start"] + left_labels + ["gap"] + right_labels + ["reward", "no_reward", "end"]
    obs_ids = np.array([vocab.obs_to_id[label] for label in obs_labels], dtype=int)

    tuning = np.zeros((n_states, n_obs), float)
    counts = np.zeros(n_obs, float)
    gammas = decode_posteriors_filtered(train, model)

    start = 0
    for trial_gamma, length in zip(gammas, np.asarray(train.lengths, dtype=int)):
        trial_obs = train.obs[start:start + length]
        for step, gamma_t in enumerate(trial_gamma):
            obs_id = int(trial_obs[step])
            counts[obs_id] += 1
            tuning[:, obs_id] += gamma_t
        start += length

    nonzero = counts > 0
    tuning[:, nonzero] /= counts[nonzero]
    return tuning[:, obs_ids], counts[obs_ids], obs_labels


def compute_valid_height_pairs_from_dataset(
    dataset,
    vocab,
    trial_length: int = 6,
    obj1_idx: int = 1,
    obj2_idx: int = 3,
) -> set[tuple[int, int]]:
    valid_pairs: set[tuple[int, int]] = set()
    n_trials = len(dataset.obs) // trial_length

    for trial_idx in range(n_trials):
        trial_obs = dataset.obs[trial_idx * trial_length:(trial_idx + 1) * trial_length]
        label_1 = vocab.id_to_obs[int(trial_obs[obj1_idx])]
        label_2 = vocab.id_to_obs[int(trial_obs[obj2_idx])]
        if not (isinstance(label_1, str) and isinstance(label_2, str)):
            continue
        if not (label_1.startswith("l_") and label_2.startswith("r_")):
            continue

        height_left = _parse_height(label_1, "l")
        height_right = _parse_height(label_2, "r")
        if height_left is None or height_right is None:
            continue
        valid_pairs.add((height_left, height_right))

    return valid_pairs


def build_conditioned_obs_heatmap(
    obs,
    act,
    vocab,
    *,
    trial_len: int = 6,
    idx_start: int = 0,
    idx_L: int = 1,
    idx_gap: int = 2,
    idx_R: int = 3,
    idx_dec: int = 3,
    idx_rew: int = 4,
    left_code: int = 1,
    right_code: int = 0,
):
    obs = np.asarray(obs, dtype=int)
    act = np.asarray(act, dtype=int)
    n_trials = len(obs) // trial_len

    labels = []
    if "start" in vocab.obs_to_id:
        labels.append("start")
    labels += [f"l_{h}" for h in _height_numbers(vocab, "l")]
    if "gap" in vocab.obs_to_id:
        labels.append("gap")
    labels += [f"r_{h}" for h in _height_numbers(vocab, "r")]
    label_ids = [vocab.obs_to_id[label] for label in labels]

    col_to_step = []
    for label in labels:
        if label == "start":
            col_to_step.append(idx_start)
        elif label == "gap":
            col_to_step.append(idx_gap)
        elif label.startswith("l_"):
            col_to_step.append(idx_L)
        elif label.startswith("r_"):
            col_to_step.append(idx_R)
        else:
            raise ValueError(f"Unexpected token in x_labels: {label}")

    def row_index(decision: str, outcome: str) -> int:
        return {("R", "reward"): 0, ("L", "reward"): 1, ("R", "no_reward"): 2, ("L", "no_reward"): 3}[(decision, outcome)]

    correct = np.zeros((4, len(labels)), float)
    denom = np.zeros(4, float)
    id_to_obs = vocab.id_to_obs

    for i in range(n_trials):
        start = i * trial_len
        obs_trial = obs[start:start + trial_len]
        act_trial = act[start:start + trial_len]
        action = int(act_trial[idx_dec])
        if action == left_code:
            decision = "L"
        elif action == right_code:
            decision = "R"
        else:
            continue

        last_obs = id_to_obs[int(obs_trial[idx_rew])]
        if isinstance(last_obs, str) and last_obs.lower() == "reward":
            outcome = "reward"
        elif isinstance(last_obs, str) and last_obs.lower() in ("no_reward", "no-reward", "noreward"):
            outcome = "no_reward"
        else:
            continue

        row = row_index(decision, outcome)
        denom[row] += 1.0
        for j, (obs_id, step) in enumerate(zip(label_ids, col_to_step)):
            if step < len(obs_trial) and int(obs_trial[step]) == int(obs_id):
                correct[row, j] += 1.0

    heatmap = np.zeros_like(correct)
    for row in range(4):
        if denom[row] > 0:
            heatmap[row] = correct[row] / denom[row]
    return heatmap, labels


def sample_ahmm_nonstream(model, trial_len: int, n_trials: int, *, seed: int | None = None, s0=None, gamma0=None):
    rng = np.random.default_rng(seed)

    pi = np.asarray(model.pi, float)
    transitions = np.asarray(model.trans, float)
    policy = np.asarray(model.policy, float)
    emit = np.asarray(model.emit, float)

    n_states = pi.shape[0]
    n_actions = policy.shape[1]
    n_obs = emit.shape[1]
    total_steps = trial_len * n_trials

    states = np.empty(total_steps, dtype=int)
    actions = np.empty(total_steps, dtype=int)
    obs = np.empty(total_steps, dtype=int)

    if gamma0 is not None:
        p0 = np.asarray(gamma0, float)
        p0 = p0 / (p0.sum() + 1e-12)
    else:
        p0 = pi / (pi.sum() + 1e-12)

    offset = 0
    for _ in range(n_trials):
        if s0 is not None:
            state = int(s0)
        else:
            state = int(rng.choice(n_states, p=p0))

        action = int(rng.choice(n_actions, p=policy[state] / (policy[state].sum() + 1e-12)))
        obs_token = int(rng.choice(n_obs, p=emit[state] / (emit[state].sum() + 1e-12)))
        states[offset], actions[offset], obs[offset] = state, action, obs_token

        for local_t in range(1, trial_len):
            global_t = offset + local_t
            p_next = transitions[actions[global_t - 1], states[global_t - 1]]
            p_next = p_next / (p_next.sum() + 1e-12)
            state = int(rng.choice(n_states, p=p_next))
            action = int(rng.choice(n_actions, p=policy[state] / (policy[state].sum() + 1e-12)))
            obs_token = int(rng.choice(n_obs, p=emit[state] / (emit[state].sum() + 1e-12)))
            states[global_t], actions[global_t], obs[global_t] = state, action, obs_token

        offset += trial_len

    lengths = np.full(n_trials, trial_len, dtype=int)
    return states, actions, obs, lengths


def obs_heatmap_corr_for_session_and_model(ds, model, vocab, T_model=None, seed=None, corr_metric: str = "spearman") -> float:
    animal_heatmap, animal_labels = build_conditioned_obs_heatmap(ds.obs, ds.act, vocab)
    animal_heatmap = np.delete(animal_heatmap, [0, 11], axis=1)
    animal_labels = np.delete(animal_labels, [0, 11])

    if T_model is None:
        T_model = len(ds.obs)

    _, model_acts, model_obs, _ = sample_ahmm_nonstream(
        model,
        trial_len=6,
        n_trials=T_model,
        seed=seed or 0,
    )
    model_heatmap, model_labels = build_conditioned_obs_heatmap(model_obs, model_acts, vocab)
    model_heatmap = np.delete(model_heatmap, [0, 11], axis=1)
    model_labels = np.delete(model_labels, [0, 11])

    v1 = animal_heatmap.ravel()
    v2 = model_heatmap.ravel()
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 2:
        return 0.0

    if corr_metric == "pearson":
        r, _ = pearsonr(v1[mask], v2[mask])
    else:
        r, _ = spearmanr(v1[mask], v2[mask])
    if r is None or np.isnan(r):
        return 0.0
    return float(r)


def cross_session_on_selected_models(
    sessions_all,
    gen,
    vocab,
    best_model_by_session: dict[tuple[str, str], str],
    *,
    test_ratio: float = 0.4,
    valid_ratio: float = 0.5,
    T_model: int = 3000,
    random_state: int = 0,
) -> pd.DataFrame:
    rows = []
    for (train_animal, train_date), model_path in best_model_by_session.items():
        model_path = str(model_path).replace("\\", "/")
        if not os.path.isfile(model_path):
            continue

        model, meta = load_ahmm(model_path)
        for session in sessions_all:
            ds = records_to_dataset(session["records"], gen)
            _, ds_test_val = train_test_split_by_sequence(ds, test_ratio=test_ratio, seed=random_state)
            if valid_ratio > 0:
                ds_test, _ = train_test_split_by_sequence(ds_test_val, test_ratio=valid_ratio, seed=random_state)
            else:
                ds_test = ds_test_val

            heat_corr = obs_heatmap_corr_for_session_and_model(
                ds_test,
                model,
                vocab,
                T_model=T_model,
                seed=0,
                corr_metric="pearson",
            )
            rows.append(
                {
                    "train_animal": train_animal,
                    "train_date": train_date,
                    "test_animal": session["animal"],
                    "test_date": pd.to_datetime(session["exp_date"]).date().isoformat(),
                    "model_path": model_path,
                    "selected_seed": int(meta.get("seed", -1)) if isinstance(meta, dict) else None,
                    "selected_pde": float(meta.get("pde", np.nan)) if isinstance(meta, dict) else np.nan,
                    "model_corr_heatmap_similarity_score": float(heat_corr),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["train_animal", "train_date", "test_animal", "test_date"]).reset_index(drop=True)
    return out


def plot_neural_and_top10_models(
    all_neural: dict[tuple[str, str], np.ndarray],
    df_models: pd.DataFrame,
    *,
    df_used: pd.DataFrame | None = None,
    within_group_label: str = "AA",
    rep_col: str = "rep_matrix",
    pde_col: str = "pde",
    sim_col: str = "sim_rank",
    seed_col: str = "seed",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    topk: int = 10,
    rows_per_fig: int = 10,
    figsize_per_cell: tuple[float, float] = (1.2, 1.2),
    title_prefix: str = "",
):
    df = df_models.copy()
    df[train_date_col] = df[train_date_col].map(_date_iso)

    used_within = {}
    if df_used is not None and len(df_used) > 0:
        used_rows = df_used.copy()
        if "is_within_used" in used_rows.columns:
            mask = used_rows["is_within_used"].astype(bool)
        elif "group" in used_rows.columns:
            mask = used_rows["group"] == within_group_label
        else:
            mask = used_rows["test_key"] == used_rows["train_session_key"]
        used_rows = used_rows.loc[mask].copy()

        def seed_from_model_key(model_key):
            try:
                if isinstance(model_key, tuple) and len(model_key) >= 3:
                    return int(model_key[2])
            except Exception:
                return None
            return None

        used_rows["_seed_from_model_key"] = used_rows["model_key"].map(seed_from_model_key) if "model_key" in used_rows.columns else None
        for _, row in used_rows.iterrows():
            test_key = row.get("test_key", None)
            train_key = row.get("train_session_key", None)
            seed = row.get("_seed_from_model_key", None)
            if test_key is None or train_key is None or seed is None:
                continue
            used_within.setdefault(test_key, set()).add((train_key, seed))

    sessions = sorted(all_neural.keys(), key=lambda x: (x[0], x[1]))
    ncols = 1 + topk
    figs = []

    for start in range(0, len(sessions), rows_per_fig):
        block = sessions[start:start + rows_per_fig]
        fig, axes = plt.subplots(len(block), ncols, figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * len(block)), squeeze=False)
        for row_i, (animal, date) in enumerate(block):
            test_key = (animal, date)
            ax0 = axes[row_i, 0]
            ax0.imshow(np.asarray(all_neural[(animal, date)], float), interpolation="nearest")
            ax0.set_title(f"{title_prefix}{animal} {date}\nNeural", fontsize=7)
            ax0.set_xticks([])
            ax0.set_yticks([])

            sub = df[(df[train_animal_col] == animal) & (df[train_date_col] == date)]
            sub = sub.sort_values(pde_col, ascending=False).head(topk).reset_index(drop=True)
            used_set = used_within.get(test_key, set())

            for col_i in range(topk):
                ax = axes[row_i, 1 + col_i]
                ax.set_xticks([])
                ax.set_yticks([])
                if col_i >= len(sub):
                    ax.axis("off")
                    continue

                row = sub.iloc[col_i]
                mat = row.get(rep_col, None)
                if mat is None or (isinstance(mat, float) and np.isnan(mat)):
                    ax.axis("off")
                    continue

                ax.imshow(np.asarray(mat, float), interpolation="nearest")
                seed = row.get(seed_col, None)
                pde = row.get(pde_col, np.nan)
                sim = row.get(sim_col, np.nan)
                title = f"#{col_i + 1}\nseed={int(seed):03d}\nPDE={pde:.3f}"
                if pd.notna(sim):
                    title += f"\nsimRank={sim:.0f}"
                ax.set_title(title, fontsize=7)

                is_used = seed is not None and len(used_set) > 0 and ((test_key, int(seed)) in used_set)
                if is_used:
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            transform=ax.transAxes,
                            fill=False,
                            linewidth=5.0,
                            edgecolor="red",
                        )
                    )

        plt.tight_layout()
        plt.show()
        figs.append(fig)

    return figs


def model_action_distribution_all_sessions(
    sessions_all,
    gen,
    df_models: pd.DataFrame,
    vocab,
    *,
    trial_to_sample: int = 5000,
    trial_length: int = 6,
    obj1_idx: int = 1,
    obj2_idx: int = 3,
    dec_idx: int = 3,
    left_code: int = 1,
    skip_code: int = 2,
    seed_base: int = 0,
    load_fn=None,
):
    del load_fn
    pair_actions: dict[tuple[int, int], list[int]] = {}
    model_idx = 0

    for session in sessions_all:
        animal = session["animal"]
        exp_date = session["exp_date"]
        dataset = records_to_dataset(session["records"], gen)
        valid_pairs = compute_valid_height_pairs_from_dataset(
            dataset,
            vocab,
            trial_length=trial_length,
            obj1_idx=obj1_idx,
            obj2_idx=obj2_idx,
        )
        if not valid_pairs:
            continue

        try:
            model, _, _ = load_best_pde_model_for_session(df_models, animal, exp_date)
        except (KeyError, FileNotFoundError):
            continue

        _, acts, obs, _ = sample_ahmm(
            model,
            trial_len=trial_length,
            n_trials=trial_to_sample,
            seed=seed_base + model_idx,
        )
        model_idx += 1

        for trial_idx in range(trial_to_sample):
            start = trial_idx * trial_length
            trial_obs = obs[start:start + trial_length]
            trial_act = acts[start:start + trial_length]
            decision = int(trial_act[dec_idx])
            if decision == skip_code:
                continue

            label_1 = vocab.id_to_obs[int(trial_obs[obj1_idx])]
            label_2 = vocab.id_to_obs[int(trial_obs[obj2_idx])]
            if not (isinstance(label_1, str) and isinstance(label_2, str)):
                continue
            if not (label_1.startswith("l_") and label_2.startswith("r_")):
                continue

            height_left = _parse_height(label_1, "l")
            height_right = _parse_height(label_2, "r")
            if height_left is None or height_right is None:
                continue
            if (height_left, height_right) not in valid_pairs:
                continue

            pair_actions.setdefault((height_left, height_right), []).append(int(decision == left_code))

    if not pair_actions:
        raise ValueError("No valid (L_height, R_height) pairs found across sessions/models.")

    heights_left = sorted({pair[0] for pair in pair_actions})
    heights_right = sorted({pair[1] for pair in pair_actions})
    idx_left = {height: idx for idx, height in enumerate(heights_left)}
    idx_right = {height: idx for idx, height in enumerate(heights_right)}

    grid = np.full((len(heights_left), len(heights_right)), np.nan)
    counts = np.zeros((len(heights_left), len(heights_right)), dtype=int)
    for (height_left, height_right), actions in pair_actions.items():
        row = idx_left[height_left]
        col = idx_right[height_right]
        values = np.asarray(actions, dtype=int)
        if values.size == 0:
            continue
        grid[row, col] = float(values.mean())
        counts[row, col] = int(values.size)

    return grid, heights_left, heights_right, counts


def collect_left_tuning_all_sessions_from_df(
    sessions_all,
    gen,
    vocab,
    dfv: pd.DataFrame,
    df_models: pd.DataFrame,
    *,
    groups=("AA", "BA"),
    stream: bool = False,
    min_peak: float = 1e-3,
    model_key_col: str = "model_key",
    group_col: str = "group",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    seed_col: str = "seed",
    path_col: str = "save_path",
    pde_col: str = "pde",
    use_best: bool = False,
):
    dm = df_models.copy()
    dm[train_date_col] = pd.to_datetime(dm[train_date_col]).dt.strftime("%Y-%m-%d")
    dm[path_col] = dm[path_col].map(normalize_path)
    dm[seed_col] = dm[seed_col].astype(int)

    models_by_session: dict[tuple[str, str], list[str]] = {}
    missing = []

    if use_best:
        dm_best = (
            dm.sort_values(pde_col, ascending=False)
            .drop_duplicates([train_animal_col, train_date_col], keep="first")
        )
        for _, row in dm_best.iterrows():
            models_by_session[(str(row[train_animal_col]), str(row[train_date_col]))] = [os.path.normpath(row[path_col])]
    else:
        df_use = dfv[dfv[group_col].isin(groups)].copy()
        used_keys = {parse_model_key(model_key) for model_key in df_use[model_key_col]}
        key_to_path = {
            (str(row[train_animal_col]), str(row[train_date_col]), int(row[seed_col])): row[path_col]
            for _, row in dm.iterrows()
        }
        for key in used_keys:
            path = key_to_path.get(key)
            if path is None:
                missing.append(key)
                continue
            models_by_session.setdefault(key[:2], []).append(os.path.normpath(path))
        for key in models_by_session:
            models_by_session[key] = sorted(models_by_session[key])

    session_lookup = make_session_lookup(sessions_all)
    all_rows = []
    meta_rows = []
    left_labels = None

    for (animal, date_str), paths in models_by_session.items():
        session = session_lookup.get((animal, date_str))
        if session is None:
            continue
        dataset = records_to_dataset(session["records"], gen)

        for model_path in paths:
            if not os.path.exists(model_path):
                continue

            model, _ = load_ahmm(model_path)
            tuning_height, _, height_labels = compute_state_tuning_all(model, dataset, vocab, stream=stream)

            left_mask = np.array([label.startswith("l_") for label in height_labels])
            current_left_labels = [label for label in height_labels if label.startswith("l_")]
            if left_labels is None:
                left_labels = current_left_labels
            elif left_labels != current_left_labels:
                raise ValueError("Inconsistent left-bar labels across sessions/models.")

            tuning_left = tuning_height[:, left_mask]
            for state_idx in range(tuning_height.shape[0]):
                row_full = tuning_height[state_idx]
                peak_idx = int(np.argmax(row_full))
                peak_label = height_labels[peak_idx]
                peak_value = float(row_full[peak_idx])
                if peak_label.startswith("l_") and peak_value > min_peak:
                    all_rows.append(tuning_left[state_idx][None, :])
                    meta_rows.append(
                        {
                            "animal": animal,
                            "exp_date": session["exp_date"],
                            "state": state_idx,
                            "model_path": model_path,
                            "use_best": bool(use_best),
                        }
                    )

    if not all_rows:
        tuning_all = np.zeros((0, 0), float) if left_labels is None else np.zeros((0, len(left_labels)), float)
        meta_df = pd.DataFrame(columns=["animal", "exp_date", "state", "model_path", "use_best"])
    else:
        tuning_all = np.vstack(all_rows)
        meta_df = pd.DataFrame(meta_rows)

    if (not use_best) and missing:
        print(f"[collect_left_tuning_all_sessions_from_df] WARNING: {len(missing)} model_key(s) in dfv not found in df_models. First 5: {missing[:5]}")

    return tuning_all, left_labels, meta_df


def pca_ncomponents_95(tuning_all, max_components=None):
    X = np.asarray(tuning_all, float)
    _, n_features = X.shape
    if max_components is None:
        max_components = n_features
    pca = PCA(n_components=max_components)
    pca.fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_opt = int(np.searchsorted(cumulative, 0.95) + 1)
    return n_opt, cumulative


def pca_group_and_order_states_auto(tuning_all, variance_threshold: float = 0.95):
    del variance_threshold
    n_opt, _ = pca_ncomponents_95(tuning_all)
    pca = PCA(n_components=n_opt)
    X_pca = pca.fit_transform(tuning_all)
    labels = np.argmax(np.abs(X_pca), axis=1)

    order_list = []
    for pc_idx in range(n_opt):
        group_idx = np.where(labels == pc_idx)[0]
        if group_idx.size == 0:
            continue

        projections = X_pca[group_idx, pc_idx]
        pos_idx = group_idx[projections > 0]
        neg_idx = group_idx[projections < 0]

        if pos_idx.size > 0:
            pos_sorted = pos_idx[np.argsort(-projections[projections > 0])]
        else:
            pos_sorted = np.array([], dtype=int)

        if neg_idx.size > 0:
            neg_sorted = neg_idx[np.argsort(projections[projections < 0])]
        else:
            neg_sorted = np.array([], dtype=int)

        order_list.extend(np.concatenate([pos_sorted, neg_sorted]).tolist())

    return np.array(order_list, dtype=int), labels, X_pca, pca


def compute_lag_mean_pv_cov(ds, model, *, vocab, tower_heights=None, matrix_metric: str = "corr"):
    if tower_heights is None:
        tower_heights = _height_numbers(vocab, "l") or list(range(200, 651, 50))

    M, _, _ = show_pv_representation_ahmm_dataset(
        t=1,
        s=1,
        dataset=ds,
        model=model,
        tower_heights=tower_heights,
        id_to_obs=vocab.id_to_obs,
        plot=False,
        metric=matrix_metric,
    )

    lag_mean = np.zeros(M.shape[0] - 1)
    for lag in range(lag_mean.shape[0]):
        diffs = [np.cov(M[i], M[i + lag + 1])[0, 1] for i in range(M.shape[0] - lag - 1)]
        lag_mean[lag] = np.nanmean(diffs)

    x = np.arange(lag_mean.shape[0], dtype=float)
    mask = np.isfinite(lag_mean)
    y = lag_mean[mask]
    x_masked = x[mask]
    if len(y) < 2 or np.nanstd(y) < 1e-12:
        corr = 0.0
    else:
        r_tmp, _ = spearmanr(x_masked, y)
        corr = 0.0 if (r_tmp is None or np.isnan(r_tmp)) else float(r_tmp)
    return lag_mean, corr


def _cov_corr_from_M_only(M):
    n_heights = M.shape[0]
    if n_heights < 3:
        return 0.0

    lag_mean = np.zeros(n_heights - 1)
    for lag in range(len(lag_mean)):
        diffs = [np.cov(M[i], M[i + lag + 1])[0, 1] for i in range(n_heights - lag - 1)]
        lag_mean[lag] = np.nanmean(diffs)

    x = np.arange(len(lag_mean), dtype=float)
    mask = np.isfinite(lag_mean)
    if mask.sum() < 2 or np.nanstd(lag_mean[mask]) < 1e-12:
        return 0.0

    r, _ = spearmanr(x[mask], lag_mean[mask])
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def collect_pv_lag_curves_for_best_models(
    sessions_all,
    gen,
    df_models: pd.DataFrame,
    *,
    vocab=None,
    matrix_metric: str = "corr",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    path_col: str = "save_path",
    pde_col: str = "pde",
):
    if vocab is None:
        vocab = getattr(gen, "vocab", None)
    if vocab is None:
        raise ValueError("collect_pv_lag_curves_for_best_models requires `vocab` or `gen.vocab`.")
    dm = df_models.copy()
    dm[train_animal_col] = dm[train_animal_col].astype(str)
    dm[train_date_col] = dm[train_date_col].map(date_iso)
    dm[path_col] = dm[path_col].map(normalize_path)
    dm[pde_col] = pd.to_numeric(dm[pde_col], errors="coerce")
    dm_best = (
        dm.sort_values(pde_col, ascending=False)
        .drop_duplicates([train_animal_col, train_date_col], keep="first")
        .reset_index(drop=True)
    )

    session_lookup = make_session_lookup(sessions_all)
    rows = []
    for row in tqdm(dm_best.itertuples(index=False), total=len(dm_best)):
        animal = getattr(row, train_animal_col)
        date = getattr(row, train_date_col)
        model_path = getattr(row, path_col)
        pde = getattr(row, pde_col)
        session = session_lookup.get((animal, date))
        if session is None or not os.path.exists(model_path):
            continue

        dataset = records_to_dataset(session["records"], gen)
        model, _ = load_ahmm(model_path)
        lag_mean, lag_corr = compute_lag_mean_pv_cov(dataset, model, vocab=vocab, matrix_metric=matrix_metric)
        rows.append(
            {
                "animal": animal,
                "date": date,
                "model_path": model_path,
                "pde": float(pde) if np.isfinite(pde) else np.nan,
                "lag_mean": lag_mean,
                "lag_corr": float(lag_corr),
            }
        )

    lag_df = pd.DataFrame(rows)
    if not lag_df.empty:
        lag_df = lag_df.sort_values(["animal", "date"]).reset_index(drop=True)
    return lag_df


def get_real_and_shuffle_pv_cov_corr_for_session_using_yours(
    ds_session,
    model,
    vocab,
    *,
    n_shuffle: int = 200,
    seed: int = 0,
    matrix_metric: str = "corr",
):
    rng = np.random.default_rng(seed)
    tower_heights = _height_numbers(vocab, "l") or list(range(200, 651, 50))

    _, cov_real = compute_lag_mean_pv_cov(ds_session, model, vocab=vocab, tower_heights=tower_heights, matrix_metric=matrix_metric)
    M, _, _ = show_pv_representation_ahmm_dataset(
        t=1,
        s=1,
        dataset=ds_session,
        model=model,
        tower_heights=tower_heights,
        id_to_obs=vocab.id_to_obs,
        plot=False,
        metric=matrix_metric,
    )

    shuffle_scores = np.empty(n_shuffle, float)
    for idx in range(n_shuffle):
        perm = rng.permutation(M.shape[0])
        shuffle_scores[idx] = _cov_corr_from_M_only(M[perm, :])

    return float(cov_real), float(np.nanmean(shuffle_scores))


def build_within_session_pv_cov_df_best_models(
    sessions_all,
    gen,
    vocab,
    df_models: pd.DataFrame,
    *,
    n_shuffle: int = 200,
    seed: int = 0,
    matrix_metric: str = "corr",
    train_animal_col: str = "train_animal",
    train_date_col: str = "train_date",
    path_col: str = "save_path",
    pde_col: str = "pde",
):
    dm = df_models.copy()
    dm[train_animal_col] = dm[train_animal_col].astype(str)
    dm[train_date_col] = pd.to_datetime(dm[train_date_col]).dt.strftime("%Y-%m-%d")
    dm[path_col] = dm[path_col].map(normalize_path)
    dm[pde_col] = pd.to_numeric(dm[pde_col], errors="coerce")
    dm_best = (
        dm.sort_values(pde_col, ascending=False)
        .drop_duplicates([train_animal_col, train_date_col], keep="first")
        .reset_index(drop=True)
    )

    session_lookup = make_session_lookup(sessions_all)
    rows = []
    for row in tqdm(dm_best.itertuples(index=False), total=len(dm_best)):
        animal = getattr(row, train_animal_col)
        date_str = getattr(row, train_date_col)
        model_path = getattr(row, path_col)
        pde = getattr(row, pde_col)
        session = session_lookup.get((animal, date_str))
        if session is None or not os.path.exists(model_path):
            continue

        ds_session = records_to_dataset(session["records"], gen)
        model, _ = load_ahmm(model_path)
        cov_real, cov_shuffle = get_real_and_shuffle_pv_cov_corr_for_session_using_yours(
            ds_session,
            model,
            vocab,
            n_shuffle=n_shuffle,
            seed=seed,
            matrix_metric=matrix_metric,
        )
        rows.append(
            {
                "animal": animal,
                "exp_date": pd.to_datetime(session["exp_date"]),
                "cov_real": cov_real,
                "cov_shuffle": cov_shuffle,
                "model_path": model_path,
                "pde": float(pde) if np.isfinite(pde) else np.nan,
            }
        )

    pv_df = pd.DataFrame(rows)
    if not pv_df.empty:
        pv_df = pv_df.sort_values(["animal", "exp_date"]).reset_index(drop=True)
    return pv_df


def save_pv_df_to_mat(pv_df: pd.DataFrame, out_path: str = "Data/model_bar_correlation_data.mat"):
    sio.savemat(
        out_path,
        {
            "cov_real": pv_df["cov_real"].to_numpy(),
            "cov_shuffle": pv_df["cov_shuffle"].to_numpy(),
            "animals": pv_df["animal"].astype(str).to_numpy(),
            "date": pd.to_datetime(pv_df["exp_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        },
    )
    return out_path


def build_cat_ids_from_obs_labels(tuning, obs_labels):
    tuning = np.asarray(tuning, float)
    obs_labels = np.asarray(obs_labels, dtype=object)
    col_cat = np.full(len(obs_labels), -1, dtype=int)

    for idx, label in enumerate(obs_labels):
        label = str(label)
        lower = label.lower()
        if lower == "start":
            col_cat[idx] = 0
        elif label.startswith("l_"):
            col_cat[idx] = 1
        elif lower == "gap":
            col_cat[idx] = 2
        elif label.startswith("r_"):
            col_cat[idx] = 3
        elif lower == "reward":
            col_cat[idx] = 4
        elif lower in ("no_reward", "no reward", "noreward"):
            col_cat[idx] = 5
        elif lower == "end":
            col_cat[idx] = 6

    peak_col = np.nanargmax(tuning, axis=1)
    cat_ids = col_cat[peak_col]
    cat_ids[cat_ids < 0] = 2
    id_to_name = {0: "Start", 1: "Bar 1", 2: "Gap", 3: "Bar 2", 4: "Reward", 5: "No reward", 6: "End"}
    return cat_ids, col_cat, id_to_name


def blocks_from_category_ids(cat_ids_sorted, id_to_name):
    cat_ids_sorted = np.asarray(cat_ids_sorted, int)
    blocks = []
    start = 0
    n = len(cat_ids_sorted)
    for idx in range(1, n + 1):
        if idx == n or cat_ids_sorted[idx] != cat_ids_sorted[idx - 1]:
            cid = int(cat_ids_sorted[idx - 1])
            blocks.append({"name": str(id_to_name.get(cid, cid)), "start": start, "end": idx, "cid": cid})
            start = idx
    return blocks


def select_left_bar_tuning(tuning, obs_labels, prefix: str = "l_"):
    obs_labels = np.asarray(obs_labels, dtype=object)
    col_mask = np.array([str(label).startswith(prefix) for label in obs_labels])
    if not np.any(col_mask):
        raise ValueError(f"No observation labels start with '{prefix}'")
    return tuning[:, col_mask], obs_labels[col_mask].tolist()


def build_similarity_matrix(neural_s3, model_s3, neural_s4, model_s4):
    return pd.DataFrame(
        [
            [compare_dist_matrices(neural_s3, model_s3, method="pearson", on_constant="nan"),
             compare_dist_matrices(neural_s3, model_s4, method="pearson", on_constant="nan")],
            [compare_dist_matrices(neural_s4, model_s3, method="pearson", on_constant="nan"),
             compare_dist_matrices(neural_s4, model_s4, method="pearson", on_constant="nan")],
        ],
        index=["S01", "S02"],
        columns=["S01", "S02"],
    )


def prepare_obs_heatmap_crosscorr_2x2(
    ds_session,
    ds_session_diff,
    model_animal,
    model_diff,
    vocab,
    *,
    T_model: int = 3000,
    seed: int = 0,
    corr_metric: str = "pearson",
    ds_labels=("G375", "G506"),
    model_labels=("G375", "G506"),
    per_pair_kwargs=None,
):
    def _score(dataset, model, key):
        kwargs = dict(T_model=T_model, seed=seed, corr_metric=corr_metric)
        if per_pair_kwargs and key in per_pair_kwargs:
            kwargs.update(per_pair_kwargs[key])
        return float(
            obs_heatmap_corr_for_session_and_model(
                dataset,
                model,
                vocab,
                T_model=kwargs["T_model"],
                seed=kwargs["seed"],
                corr_metric=kwargs["corr_metric"],
            )
        )

    v_m0_ds0 = _score(ds_session, model_animal, ("ds0", "m0"))
    v_m0_ds1 = _score(ds_session_diff, model_animal, ("ds1", "m0"))
    v_m1_ds0 = _score(ds_session, model_diff, ("ds0", "m1"))
    v_m1_ds1 = _score(ds_session_diff, model_diff, ("ds1", "m1"))

    conf_df = pd.DataFrame(
        [[v_m0_ds0, v_m0_ds1], [v_m1_ds0, v_m1_ds1]],
        index=list(model_labels),
        columns=list(ds_labels),
    )
    raw = {
        "v_m0_ds0": v_m0_ds0,
        "v_m0_ds1": v_m0_ds1,
        "v_m1_ds0": v_m1_ds0,
        "v_m1_ds1": v_m1_ds1,
        "settings": {"T_model": T_model, "seed": seed, "corr_metric": corr_metric},
    }
    return conf_df, raw


def _as_key2(x, *, na_fallback=("OPT", "OPT")):
    if x is None or (isinstance(x, float) and np.isnan(x)) or (x is pd.NA):
        return na_fallback
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return str(x[0]), str(x[1])
    if isinstance(x, str):
        s = x.strip()
        try:
            value = ast.literal_eval(s)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return str(value[0]), str(value[1])
        except Exception:
            pass
    return str(x), ""


def _safe_datestr(series: pd.Series) -> pd.Series:
    as_str = series.astype(str)
    dt = pd.to_datetime(as_str, errors="coerce")
    return as_str.where(dt.isna(), dt.dt.strftime("%Y-%m-%d"))


def prep_dfv_for_heatmap(
    dfv: pd.DataFrame,
    *,
    test_col: str = "test_key",
    train_col: str = "train_session_key",
    na_train_fallback=("OPT", "OPT"),
) -> pd.DataFrame:
    out = dfv.copy()
    test_pairs = out[test_col].map(_as_key2)
    train_pairs = out[train_col].map(lambda x: _as_key2(x, na_fallback=na_train_fallback))
    out["test_animal"] = test_pairs.map(lambda t: t[0])
    out["test_date"] = _safe_datestr(test_pairs.map(lambda t: t[1]))
    out["train_animal"] = train_pairs.map(lambda t: t[0])
    out["train_date"] = _safe_datestr(train_pairs.map(lambda t: t[1]))
    return out


def model_eval_pipeline(model_summary_file: str, all_neural: dict[tuple[str, str], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = repair_model_paths(pd.read_pickle(resolve_data_path(model_summary_file)))
    df["train_date"] = df["train_date"].map(_date_iso)
    df["pde_rank"] = df.groupby(["train_animal", "train_date"])["pde"].rank(method="min", ascending=False)

    sim_df = build_model_x_neural_similarity_df(df, all_neural)
    is_within = (sim_df["train_animal"] == sim_df["test_animal"]) & (sim_df["train_date"] == sim_df["test_date"])
    has_within = is_within.groupby([sim_df["train_animal"], sim_df["train_date"]]).transform("any")
    sim_df_filtered = sim_df[has_within].reset_index(drop=True)
    sim_df_filtered["sim_rank"] = (
        sim_df_filtered.groupby(["train_animal", "train_date", "test_animal", "test_date"])["similarity"]
        .rank(method="min", ascending=False)
    )
    return df, sim_df, sim_df_filtered


def model_eval_pipeline_plot(sim_df_filtered: pd.DataFrame, n_states: int):
    ks = np.arange(1, 101)
    res1 = pd.DataFrame(
        [within_between_rank_sum_on_ranks(sim_df_filtered, k=int(k), n=1, alternative="less") for k in ks]
    )

    plt.figure(figsize=(6, 4))
    plt.plot(res1["k"], res1["p"], marker="o")
    plt.axhline(0.05, ls="--")
    plt.yscale("log")
    plt.xlabel("k (top-k by PDE rank)")
    plt.ylabel("Wilcoxon p (AA > BA)")
    plt.title(f"Similarity rank\n{n_states}-state models, random train/val split")
    k_min = res1.loc[res1["p"].idxmin(), "k"]
    p_min = res1["p"].min()
    plt.scatter([k_min], [p_min], zorder=3)
    plt.annotate(
        f"min p = {p_min:.2e}\n(k = {int(k_min)})",
        xy=(k_min, p_min),
        xytext=(k_min, p_min * 2),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        color="red",
    )
    plt.tight_layout()
    plt.show()

    res2 = pd.DataFrame(
        [within_between_similarity_on_ranks(sim_df_filtered, k=int(k), n=1, alternative="greater") for k in ks]
    )
    plt.figure(figsize=(6, 4))
    plt.plot(res2["k"], res2["p"], marker="o")
    plt.axhline(0.05, ls="--")
    plt.yscale("log")
    plt.xlabel("k (top-k by PDE rank)")
    plt.ylabel("Wilcoxon p (AA > BA)")
    plt.title(f"Similarity\n{n_states}-state models, random train/val split")
    k_min = res2.loc[res2["p"].idxmin(), "k"]
    p_min = res2["p"].min()
    plt.scatter([k_min], [p_min], zorder=3)
    plt.annotate(
        f"min p = {p_min:.2e}\n(k = {int(k_min)})",
        xy=(k_min, p_min),
        xytext=(k_min, p_min * 2),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        color="red",
    )
    plt.tight_layout()
    plt.show()


def model_eval_ks_search(sim_df_filtered: pd.DataFrame, n_states: int, k1: int = 30, k2: int = 80):
    k1s = np.arange(1, k1)
    k2s = np.arange(1, k2)
    result = sweep_k1_k2(sim_df_filtered, k1s, k2s, alternative="less")
    p_matrix = result.pivot(index="k1", columns="k2", values="p")

    plt.figure(figsize=(10, 6))
    image = plt.imshow(np.log10(p_matrix), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(image, label="log10(p-value)")
    plt.xlabel("k2 (upper PDE rank bound)")
    plt.ylabel("k1 (lower PDE rank bound)")
    plt.title(f"Within vs Between: p-value heatmap over (k1, k2)\n{n_states}-state models, random train/val split")
    plt.contour(np.log10(p_matrix), levels=[np.log10(0.05)], colors="red", linewidths=1)
    plt.tight_layout()
    plt.show()


def model_eval_behavior_cross_compare(
    df: pd.DataFrame,
    sessions_combined,
    gen,
    vocab,
    *,
    test_ratio: float = 0.4,
    valid_ratio: float = 0,
    T_model: int = 3000,
) -> pd.DataFrame:
    best_model_by_session, _ = select_rank1_pde_models(df)
    cg_df = cross_session_on_selected_models(
        sessions_combined,
        gen,
        vocab,
        best_model_by_session,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        T_model=T_model,
    )
    cg_df["model_corr_heatmap_similarity_percentile"] = (
        cg_df.groupby(["test_animal", "test_date"])["model_corr_heatmap_similarity_score"]
        .transform(lambda s: s.rank(pct=True, method="average") * 100)
    )
    return cg_df


def model_eval_behavior_plot(cg_df: pd.DataFrame, metric: str = "model_corr_heatmap_similarity_score", ascending: bool = True) -> pd.DataFrame:
    return plot_violin_within_between(cg_df, metric=metric, ascending=ascending)

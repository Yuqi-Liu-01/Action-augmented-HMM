import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib import gridspec, rcParams
from typing import Any, Dict, List, Tuple, Optional
import igraph
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import spearmanr

# RGB values from MATLAB parula, scaled to 0–1
parula_colors = np.array([
    [0.2081, 0.1663, 0.5292],
    [0.2116, 0.1898, 0.5777],
    [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771],
    [0.1959, 0.2645, 0.7279],
    [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303],
    [0.0591, 0.3598, 0.8683],
    [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828],
    [0.0165, 0.4266, 0.8786],
    [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641],
    [0.0629, 0.4737, 0.8554],
    [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384],
    [0.0793, 0.5200, 0.8312],
    [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240],
    [0.0488, 0.5772, 0.8228],
    [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135],
    [0.0239, 0.6287, 0.8038],
    [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768],
    [0.0267, 0.6642, 0.7607],
    [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254],
    [0.0843, 0.6928, 0.7062],
    [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646],
    [0.1801, 0.7177, 0.6424],
    [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954],
    [0.3022, 0.7376, 0.5712],
    [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244],
    [0.4420, 0.7481, 0.5033],
    [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661],
    [0.5709, 0.7485, 0.4494],
    [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188],
    [0.6834, 0.7435, 0.4044],
    [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768],
    [0.7858, 0.7356, 0.3633],
    [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360],
    [0.8824, 0.7274, 0.3217],
    [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886],
    [0.9739, 0.7314, 0.2666],
    [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164],
    [0.9955, 0.7861, 0.1967],
    [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633],
    [0.9697, 0.8481, 0.1475],
    [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132],
    [0.9598, 0.9218, 0.0948],
    [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538]
])

parula = LinearSegmentedColormap.from_list('parula', parula_colors)

DEFAULT_ACTION_COLORS = {
    "decision_L": "#1f77b4",
    "decision_R": "#d62728",
    "observation": "#000000",
}

from ahmm_utils import decode_posteriors_filtered

# ----------------- small helpers -----------------

def _get_trans_tensor(model) -> np.ndarray:
    """Return (A,S,S) action-conditioned transitions as np.ndarray."""
    T = getattr(model, "trans", None)
    if T is None:
        raise AttributeError("model.trans missing; expected shape (A,S,S) or dict[action]->(S,S).")
    if isinstance(T, dict):
        # sort keys to get deterministic stacking
        mats = [np.asarray(T[a]) for a in sorted(T.keys())]
        T = np.stack(mats, axis=0)
    T = np.asarray(T)
    if T.ndim != 3:
        raise ValueError("`model.trans` must have shape (A,S,S).")
    return T

def _decode_states(model, obs, acts) -> np.ndarray:
    """Use model.decode Viterbi and return only the state path (1D int array)."""
    obs  = np.asarray(obs,  dtype=int).ravel()
    acts = np.asarray(acts, dtype=int).ravel()
    try:
        states = model.decode(obs, acts, return_logp=False)
    except TypeError:
        out = model.decode(obs, acts)
        states = out[1] if isinstance(out, (list, tuple)) and len(out) >= 2 else out
    return np.asarray(states, dtype=int)

def _edge_contrib_by_action(
    model, states: np.ndarray, acts: np.ndarray, v: np.ndarray, *,
    mode: str = "model", action_weights: str | np.ndarray = "empirical",
    normalize_rows: bool = True
):
    """
    Returns:
      A_adj : (n,n) total adjacency
      C_act : (A,n,n) per-action contribution
    """
    T_all = _get_trans_tensor(model)          # (A,S,S)
    A_cnt = T_all.shape[0]
    n = len(v)

    if mode == "model":
        T_vis = T_all[:, v][:, :, v]          # (A,n,n)

        if isinstance(action_weights, str) and action_weights == "policy":
            # --- state-specific weighting by policy π(a|s) ---
            Pi = np.asarray(model.policy, float)          # (S,A)
            Pi_vis = Pi[v, :]                              # (n,A)
            # broadcast: for each action a, multiply row i by π(a|i)
            C_act = T_vis * Pi_vis.T[:, :, None]          # (A,n,n)
            A_adj = C_act.sum(axis=0)                     # (n,n)

        else:
            # previous behavior (global weights)
            if isinstance(action_weights, str):
                if action_weights == "uniform":
                    w = np.ones(A_cnt) / A_cnt
                elif action_weights == "empirical":
                    counts = np.bincount(np.asarray(acts, int), minlength=A_cnt).astype(float)
                    w = counts / counts.sum() if counts.sum() > 0 else np.ones(A_cnt)/A_cnt
                else:
                    raise ValueError("action_weights must be 'policy', 'empirical', 'uniform', or array.")
            else:
                w = np.asarray(action_weights, float); w = w / (w.sum() + 1e-12)

            C_act = (w[:, None, None] * T_vis)            # (A,n,n)
            A_adj  = C_act.sum(axis=0)

    elif mode == "path":
        C_act = np.zeros((A_cnt, n, n), float)
        idx = {s:i for i,s in enumerate(v)}
        for t in range(len(states) - 1):
            a = int(acts[t]); i = idx[states[t]]; j = idx[states[t+1]]
            C_act[a, i, j] += 1.0
        A_adj = C_act.sum(axis=0)
        if normalize_rows:
            rs = A_adj.sum(1, keepdims=True); nz = rs.squeeze() > 0
            if np.any(nz): A_adj[nz] /= rs[nz]

    else:
        raise ValueError("mode must be 'model' or 'path'")

    # keep C_act on the same scale as A_adj after any row normalization
    if normalize_rows:
        denom = C_act.sum(axis=0, keepdims=True) + 1e-12
        C_act = C_act / denom * (A_adj[None, :, :])

    return A_adj, C_act



def _layout_coords_from_igraph(A_adj: np.ndarray, layout: str = "fr",
                               layout_kwargs: Optional[Dict[str, Any]] = None,
                               spread: float = 1.6, seed: Optional[int] = None) -> Tuple[np.ndarray, igraph.Graph]:
    """Get 2D layout; normalize to [0,1], then expand by `spread`."""
    n = A_adj.shape[0]
    g = igraph.Graph.Adjacency((A_adj > 0).tolist(), mode=igraph.ADJ_DIRECTED)
    if g.vcount() != n:
        g.add_vertices(n - g.vcount())

    layout_kwargs = dict(layout_kwargs or {})
    if layout in ("fr", "fruchterman_reingold"):
        layout_kwargs.setdefault("niter", 3000)
        func = getattr(g, "layout_fruchterman_reingold", None)
        L = func(niter=layout_kwargs["niter"]) if func else g.layout("fr")
    elif layout in ("kk", "kamada_kawai"):
        func = getattr(g, "layout_kamada_kawai", None)
        L = func() if func else g.layout("kk")
    else:
        L = g.layout(layout)

    xy = np.array(L, dtype=float)
    mins, maxs = xy.min(0), xy.max(0)
    span = np.maximum(maxs - mins, 1e-9)
    xy = (xy - mins) / span
    xy = (xy - 0.5) * spread + 0.5
    return xy, g


def _draw_emission_pies(ax, xy, emit_rows, obs_labels, obs_colors,
                        radius=0.045, pie_min_frac=0.0, pie_top_k: Optional[int]=None,
                        label_text=None, label_size=9, label_halo=True):
    for i, (x, y) in enumerate(xy):
        p = np.asarray(emit_rows[i], float); p = p / (p.sum() + 1e-12)
        order = np.argsort(p)[::-1]

        if pie_top_k is not None:
            keep = order[:pie_top_k]
            fracs = list(p[keep]); cols = [obs_colors.get(obs_labels[j], "#999") for j in keep]
            rest = 1.0 - sum(fracs)
            if rest > 1e-12: fracs.append(rest); cols.append("#BBBBBB")
        else:
            mask = p >= pie_min_frac
            fracs = list(p[mask]); cols = [obs_colors.get(lbl, "#999") for lbl, m in zip(obs_labels, mask) if m]
            s = sum(fracs); 
            if s > 0: fracs = [f/s for f in fracs]

        base = Circle((x, y), radius=radius, facecolor="white",
                      edgecolor="#DDDDDD", linewidth=1.2, zorder=3)
        ax.add_patch(base)

        start = 90.0
        for frac, c in zip(fracs, cols):
            th = 360.0 * float(frac)
            if th <= 0: continue
            ax.add_patch(Wedge((x, y), r=radius, theta1=start, theta2=start+th,
                               facecolor=c, edgecolor="none", zorder=4))
            start += th

        if label_text is not None:
            t = ax.text(x, y, str(label_text[i]), ha="center", va="center",
                        fontsize=label_size, color="black", zorder=5)
            if label_halo:
                t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def _draw_weighted_arrows(ax, xy, A_adj, *, threshold=0.02, color="#3E3C3C",
                          node_radius=0.045, edge_colors=None,
                          weight_style="alpha",
                          alpha_min=0.15, alpha_max=0.95,
                          tail_width_min=2.0, tail_width_max=4.0,
                          gamma=0.6, head_min=10.0, head_max=16.0,
                          curvature=0.2):
    n = A_adj.shape[0]
    wmax = float(A_adj.max()) if A_adj.size else 0.0
    for i in range(n):
        for j in range(n):
            w = float(A_adj[i, j])
            if w < threshold: 
                continue
            shaft_color = edge_colors[i, j] if edge_colors is not None else color
            head_color  = shaft_color

            x1,y1 = xy[i]; x2,y2 = xy[j]
            dx,dy = x2-x1, y2-y1
            L = (dx*dx + dy*dy)**0.5 - 1e-12
            shrink = node_radius * 1.12
            sx,sy = x1 + shrink*dx/L, y1 + shrink*dy/L
            tx,ty = x2 - shrink*dx/L, y2 - shrink*dy/L

            wn = (w / (wmax + 1e-12)) if wmax > 0 else 0.0
            wn = wn**gamma
            if weight_style == "alpha":
                tail_w = tail_width_min
                alpha  = alpha_min + (alpha_max - alpha_min) * wn
            else:
                tail_w = tail_width_min + (tail_width_max - tail_width_min) * wn
                alpha  = alpha_max

            shaft = FancyArrowPatch((sx, sy), (tx, ty),
                                    arrowstyle='-',
                                    linewidth=tail_w, color=shaft_color, alpha=alpha,
                                    connectionstyle=f"arc3,rad={curvature}", zorder=2)
            ax.add_patch(shaft)

            ms = head_min + (head_max - head_min) * wn
            head = FancyArrowPatch((tx - 5e-7*dx, ty - 5e-7*dy), (tx, ty),
                                   arrowstyle='-|>', mutation_scale=ms,
                                   linewidth=0.0, facecolor=head_color, edgecolor=head_color,
                                   alpha=min(alpha + 0.2, 1.0),
                                   connectionstyle=f"arc3,rad={curvature}", zorder=3)
            ax.add_patch(head)


def _emission_legend_handles(obs_labels, obs_colors, markersize=8):
    handles = []
    for lbl in obs_labels:
        h = plt.Line2D(
            [0], [0],
            marker="o", linestyle="None", color="w",
            markerfacecolor=obs_colors.get(lbl, "#999999"),
            markeredgecolor="none",
            markersize=markersize,
        )
        handles.append(h)
    return handles, obs_labels


def make_obs_colors(
    obs_labels,
    *,
    height_cmap="Greens",
    left_cmap="Greens",
    right_cmap="Purples",
    shade_min=0.15,
    shade_max=0.90,
):
    colors = {}
    fixed_colors = {
        "start": "#FFFFFF",
        "gap": "#F5A000",
        "noreward": "#FF7676",
        "reward": "#24DEE8",
        "end": "#2D123B",
    }

    for label in obs_labels:
        key = label.replace("_", "").lower()
        if key in fixed_colors:
            colors[label] = fixed_colors[key]

    pat_shared = re.compile(r"(?i)^h_(\d+)$")
    pat_left = re.compile(r"(?i)^(?:l|h1)_?(\d+)$")
    pat_right = re.compile(r"(?i)^(?:r|h2)_?(\d+)$")

    shared_items, left_items, right_items = [], [], []
    for label in obs_labels:
        if (match := pat_left.match(label)) is not None:
            left_items.append((label, int(match.group(1))))
        elif (match := pat_right.match(label)) is not None:
            right_items.append((label, int(match.group(1))))
        elif (match := pat_shared.match(label)) is not None:
            shared_items.append((label, int(match.group(1))))

    def paint_group(items, cmap_name):
        if not items:
            return
        unique_heights = sorted({height for _, height in items})
        cmap = plt.get_cmap(cmap_name)
        values = np.linspace(shade_min, shade_max, len(unique_heights))
        color_by_height = {height: cmap(value) for height, value in zip(unique_heights, values)}
        for label, height in items:
            colors[label] = color_by_height[height]

    if left_items or right_items:
        paint_group(left_items, left_cmap)
        paint_group(right_items, right_cmap)
        paint_group(shared_items, height_cmap)
    else:
        paint_group(shared_items, height_cmap)

    return colors


def build_obs_raster_sorted(
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
    obs = np.asarray(obs, int)
    act = np.asarray(act, int)
    n_trials = len(obs) // trial_len

    x_labels = []
    if "start" in vocab.obs_to_id:
        x_labels.append("start")
    left_heights = sorted(int(tok.split("_")[1]) for tok in vocab.obs_to_id if str(tok).startswith("l_"))
    right_heights = sorted(int(tok.split("_")[1]) for tok in vocab.obs_to_id if str(tok).startswith("r_"))
    x_labels += [f"l_{h}" for h in left_heights]
    if "gap" in vocab.obs_to_id:
        x_labels.append("gap")
    x_labels += [f"r_{h}" for h in right_heights]
    x_ids = [vocab.obs_to_id[label] for label in x_labels]

    col_to_step = []
    for label in x_labels:
        if label == "start":
            col_to_step.append(idx_start)
        elif label == "gap":
            col_to_step.append(idx_gap)
        elif label.startswith("l_"):
            col_to_step.append(idx_L)
        else:
            col_to_step.append(idx_R)

    def _parse_height(token: str, side: str):
        match = re.match(rf"(?i)^{side}_(\d+)$", str(token).strip())
        return int(match.group(1)) if match else None

    trials = []
    id_to_obs = vocab.id_to_obs
    cond_order = ["R+reward", "L+reward", "R+no_reward", "L+no_reward"]
    order_map = {cond: idx for idx, cond in enumerate(cond_order)}
    for idx in range(n_trials):
        start = idx * trial_len
        obs_trial = obs[start:start + trial_len]
        act_trial = act[start:start + trial_len]
        decision_id = int(act_trial[idx_dec])
        if decision_id == left_code:
            decision = "L"
        elif decision_id == right_code:
            decision = "R"
        else:
            continue

        outcome_token = id_to_obs[int(obs_trial[idx_rew])]
        outcome_lower = outcome_token.lower() if isinstance(outcome_token, str) else ""
        if outcome_lower == "reward":
            outcome = "reward"
        elif outcome_lower in ("no_reward", "no-reward", "noreward"):
            outcome = "no_reward"
        else:
            continue

        left_height = _parse_height(id_to_obs[int(obs_trial[idx_L])], "l")
        right_height = _parse_height(id_to_obs[int(obs_trial[idx_R])], "r")
        row = np.zeros(len(x_labels), float)
        for col, (obs_id, step) in enumerate(zip(x_ids, col_to_step)):
            if step < len(obs_trial) and int(obs_trial[step]) == int(obs_id):
                row[col] = 1.0
        cond = f"{decision}+{outcome}"
        trials.append((cond, left_height, right_height, idx, row))

    trials.sort(key=lambda t: (order_map.get(t[0], 99), t[1] if t[1] is not None else 1e9, t[2] if t[2] is not None else 1e9))
    cond_sizes = {cond: 0 for cond in cond_order}
    rows = []
    row_order = []
    for cond, left_height, right_height, trial_idx, row in trials:
        rows.append(row)
        row_order.append((cond, left_height, right_height, trial_idx))
        cond_sizes[cond] += 1

    M = np.vstack(rows) if rows else np.zeros((0, len(x_labels)))
    return M, x_labels, row_order, cond_sizes


def plot_obs_raster_sorted(
    M,
    x_labels,
    cond_sizes,
    *,
    cmap="Greys",
    ax=None,
    title=None,
    vmin=0.0,
    vmax=1.0,
    figsize_cm=(14, 8),
    dpi=300,
    fontname=None,
    fontsize=None,
    tick_width=0.8,
    tick_length=3.0,
    axis_linewidth=0.6,
    capitalize_xticks=False,
    cond_order=("R+reward", "L+reward", "R+no_reward", "L+no_reward"),
    sep_color="#999999",
    sep_lw=1.0,
    show_block_last_row_ticks=True,
    block_tick_pad=90,
    cbar_size="3mm",
    cbar_pad=0.4,
    cbar_outline_width=0.6,
    cbar_outline_color="black",
):
    if fontsize is None:
        fontsize = {"ytick": 8, "title": 10, "cbar_label": 8, "cbar_tick_label": 5, "cond_label": 9}
    cond_label_map = {
        "R+reward": "Lick Right/Reward",
        "L+reward": "Lick Left/Reward",
        "R+no_reward": "Lick Right/No Reward",
        "L+no_reward": "Lick Left/No Reward",
    }

    if fontname is not None:
        plt.rcParams["font.family"] = fontname
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(np.asarray(M), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    x_labels_use = list(x_labels)
    if capitalize_xticks:
        x_labels_use = [label[:1].upper() + label[1:] if isinstance(label, str) and label else label for label in x_labels_use]
    ax.set_xticks([])
    ax.set_yticks([])

    y = 0
    centers = []
    labels = []
    block_bounds = []
    for cond in cond_order:
        n = int(cond_sizes.get(cond, 0))
        if n <= 0:
            continue
        y0, y1 = y, y + n
        if y0 > 0:
            ax.axhline(y0 - 0.5, color=sep_color, lw=sep_lw)
        centers.append((y0 + y1 - 1) / 2.0)
        labels.append(cond_label_map.get(cond, cond))
        block_bounds.append((y0, y1, n))
        y = y1
    if M.shape[0] > 0:
        ax.axhline(M.shape[0] - 0.5, color=sep_color, lw=sep_lw)

    if show_block_last_row_ticks and block_bounds:
        yticks = [y1 - 1 for _, y1, _ in block_bounds]
        yticklabels = [str(n) for _, _, n in block_bounds]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=fontsize["ytick"])

    if centers:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(centers)
        ax2.set_yticklabels(labels, va="center", ha="center", fontsize=fontsize["cond_label"])
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()
        ax2.tick_params(axis="y", length=0, pad=block_tick_pad, labelsize=fontsize["cond_label"])
        for spine in ("top", "right", "left"):
            ax2.spines[spine].set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize["title"])
    ax.tick_params(axis="both", width=tick_width, length=tick_length, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P(Observation)", fontsize=fontsize["cbar_label"])
    cb.ax.tick_params(width=tick_width, length=tick_length, labelsize=fontsize["cbar_tick_label"], direction="out")
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(cbar_outline_width)
    cb.outline.set_edgecolor(cbar_outline_color)
    sns.despine(ax=ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax


def plot_conditioned_obs_heatmap(
    H,
    x_labels,
    *,
    ax=None,
    title=None,
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
    figsize_cm=(12, 3),
    dpi=300,
    fontsize=None,
    axis_width=0.4,
    cbar_size=10,
    cbar_pad=0.02,
    cbar_outline_width=0.4,
    tick_width=0.8,
    tick_length=3.0,
):
    if fontsize is None:
        fontsize = {"xtick": 8, "ytick": 8, "title": 10, "cbar_label": 8, "cbar_tick_label": 8}
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([label[:1].upper() + label[1:] for label in x_labels], rotation=45, ha="right", fontsize=fontsize["xtick"])
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["Lick Right/Reward", "Lick Left/Reward", "Lick Right/No Reward", "Lick Left/No Reward"], fontsize=fontsize["ytick"])
    ax.tick_params(axis="both", width=tick_width, length=tick_length, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
    if title:
        ax.set_title(title, fontsize=fontsize["title"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(cbar_outline_width)
    cb.set_label("P(Obs | Decision, \nOutcome)", fontsize=fontsize["cbar_label"])
    cb.ax.tick_params(width=tick_width, length=tick_length, labelsize=fontsize["cbar_tick_label"])
    sns.despine(ax=ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax


def plot_confusion_heatmap(
    conf_df,
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    square=True,
    annot=True,
    fmt=".2f",
    figsize_cm=(12, 10),
    dpi=300,
    fontname="Arial",
    fontsize=None,
    xlabel="Dataset",
    ylabel="Model",
    title="Cross-Session Model Similarity (Obs-heatmap corr)",
    title_pad=6,
    xtick_rotation=0,
    ytick_rotation=0,
    tick_width=0.6,
    tick_length=3.0,
    show_cbar=True,
    cbar_size="3mm",
    cbar_pad=0.10,
    cbar_ticks=None,
    cbar_ticklabels=None,
    cbar_label="Similarity",
    use_constrained_layout=False,
):
    if fontsize is None:
        fontsize = {"title": 8, "axis_label": 7, "tick": 7, "annot": 7, "cbar_label": 7, "cbar_tick": 7}
    if fontname is not None:
        rcParams["font.family"] = fontname
        rcParams["pdf.fonttype"] = 42
        rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(
        figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54),
        dpi=dpi,
        constrained_layout=use_constrained_layout,
    )
    cax = None
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)

    data = conf_df.to_numpy(dtype=float)
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    sns.heatmap(
        conf_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=square,
        cbar=show_cbar,
        cbar_ax=cax,
        ax=ax,
        annot_kws=dict(size=fontsize["annot"]),
    )

    ax.set_xlabel(xlabel, fontsize=fontsize["axis_label"])
    ax.set_ylabel(ylabel, fontsize=fontsize["axis_label"])
    ax.set_title(title, fontsize=fontsize["title"], pad=title_pad)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, fontsize=fontsize["tick"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, fontsize=fontsize["tick"])
    ax.tick_params(axis="both", width=tick_width, length=tick_length, direction="out")

    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(cbar_label, fontsize=fontsize["cbar_label"], labelpad=8)
        cbar.ax.tick_params(width=tick_width, length=tick_length, labelsize=fontsize["cbar_tick"], direction="out")
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)
        cbar.outline.set_visible(False)

    sns.despine()
    return fig, ax


def plot_model_action_heatmap(
    grid,
    heights_L,
    heights_R,
    *,
    annotate=True,
    title=None,
    cmap="RdBu_r",
    origin="upper",
    show_diag=True,
    diag_style="--",
    diag_color="white",
    diag_lw=1.0,
    nan_color="black",
    nan_edgecolor="black",
    nan_edge_lw=0.0,
    nan_fill_value=0.5,
    equal_reward_value=0.5,
    figsize_cm=(8, 8),
    dpi=300,
    fontname="Arial",
    fontsize=None,
    xtick_rotation=0,
    ytick_rotation=0,
    tick_length=2.0,
    tick_width=0.4,
    axis_linewidth=0.4,
    show_spines=True,
    spine_color="black",
    annot_fmt="int_percent",
    annot_float_fmt="{:.2f}",
    annot_color_mode="auto",
    annot_color_fixed="white",
    annot_auto_thresh=(0.3, 0.7),
    annot_fontweight="normal",
    show_cbar=True,
    cbar_label="P(Lick Left)",
    cbar_ticks=np.linspace(0, 1, 6),
    cbar_ticklabels="percent",
    cbar_size=0.06,
    cbar_pad=0.20,
    cbar_outline=True,
    cbar_outline_width=0.4,
    cbar_outline_color="black",
    vmin=0.0,
    vmax=1.0,
):
    if fontsize is None:
        fontsize = {"title": 8, "axis_label": 7, "tick": 6, "annot": 6, "cbar_label": 7, "cbar_tick": 6}
    plt.rcParams["font.family"] = fontname
    fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)

    mask = ~np.isfinite(grid)
    data = np.array(grid, float, copy=True)
    data[mask] = nan_fill_value
    im = ax.imshow(data, origin=origin, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.invert_yaxis()

    if np.any(mask):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if mask[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=nan_color, edgecolor=nan_edgecolor, linewidth=nan_edge_lw, zorder=3))

    ax.set_xticks(np.arange(len(heights_R)))
    ax.set_yticks(np.arange(len(heights_L)))
    ax.set_xticklabels(heights_R, rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    ax.set_yticklabels(heights_L, rotation=ytick_rotation)
    ax.set_xlabel("Second bar (right)", fontsize=fontsize["axis_label"])
    ax.set_ylabel("First bar (left)", fontsize=fontsize["axis_label"])
    ax.tick_params(axis="both", width=tick_width, length=tick_length, labelsize=fontsize["tick"], direction="out")

    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)
        spine.set_edgecolor(spine_color)
        spine.set_visible(bool(show_spines))

    if show_diag:
        H, W = grid.shape
        ax.plot([-0.5, W - 0.5], [-0.5, H - 0.5], linestyle=diag_style, color=diag_color, linewidth=diag_lw, zorder=4)

    if annotate:
        lo, hi = annot_auto_thresh
        for i in range(len(heights_L)):
            for j in range(len(heights_R)):
                if mask[i, j]:
                    continue
                p_left = float(grid[i, j])
                hL = heights_L[i]
                hR = heights_R[j]
                if hL > hR:
                    p_reward = p_left
                elif hR > hL:
                    p_reward = 1.0 - p_left
                else:
                    p_reward = float(equal_reward_value)
                txt = f"{int(np.round(p_reward * 100.0))}" if annot_fmt == "int_percent" else annot_float_fmt.format(p_reward)
                color_txt = annot_color_fixed if annot_color_mode == "fixed" else ("white" if (p_left < lo or p_left > hi) else "black")
                ax.text(j, i, txt, ha="center", va="center", color=color_txt, fontsize=fontsize["annot"], fontweight=annot_fontweight, zorder=5)

    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{int(np.round(cbar_size * 100))}%", pad=cbar_pad)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cbar_label, fontsize=fontsize["cbar_label"])
        cb.ax.tick_params(width=tick_width, length=tick_length, labelsize=fontsize["cbar_tick"], direction="out")
        if cbar_ticks is not None:
            cb.set_ticks(list(cbar_ticks))
        if isinstance(cbar_ticklabels, (list, tuple, np.ndarray)):
            cb.set_ticklabels(list(cbar_ticklabels))
        elif cbar_ticklabels == "percent":
            ticks = cb.get_ticks()
            cb.set_ticklabels([f"{int(np.round(t * 100))}%" for t in ticks])
        if cbar_outline:
            cb.outline.set_visible(True)
            cb.outline.set_linewidth(cbar_outline_width)
            cb.outline.set_edgecolor(cbar_outline_color)
        else:
            cb.outline.set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize["title"])
    fig.tight_layout()
    return fig, ax


def plot_heatmap_and_pc_loadings(
    tuning_all,
    left_labels,
    *,
    pca_result=None,
    figsize_cm=(5, 2),
    dpi=600,
    cmap="GnBu",
    fontsize=None,
    seg_line_color="white",
    seg_line_width=0.8,
    pc_marker_size=8,
    pc_linewidth=0.8,
    pc_marker_edgewidth=0.8,
    pc_label_x=-0.7,
    pc_label_color="black",
    show_pc_loadings=True,
    pc_groups=None,
    pc_group_dash="|",
):
    if fontsize is None:
        fontsize = {"title": 7, "axis_label": 6, "tick": 6, "pc_label": 6, "pc_count": 5}
    if pca_result is None:
        max_components = tuning_all.shape[1]
        pca_test = PCA(n_components=max_components)
        pca_test.fit(np.asarray(tuning_all, float))
        cumulative = np.cumsum(pca_test.explained_variance_ratio_)
        n_opt = int(np.searchsorted(cumulative, 0.95) + 1)
        pca = PCA(n_components=n_opt)
        X_pca = pca.fit_transform(tuning_all)
        labels = np.argmax(np.abs(X_pca), axis=1)
        order_list = []
        for pc_idx in range(n_opt):
            group_idx = np.where(labels == pc_idx)[0]
            if group_idx.size == 0:
                continue
            proj = X_pca[group_idx, pc_idx]
            pos_idx = group_idx[proj > 0]
            neg_idx = group_idx[proj < 0]
            pos_sorted = pos_idx[np.argsort(-proj[proj > 0])] if pos_idx.size > 0 else np.array([], dtype=int)
            neg_sorted = neg_idx[np.argsort(proj[proj < 0])] if neg_idx.size > 0 else np.array([], dtype=int)
            order_list.extend(np.concatenate([pos_sorted, neg_sorted]).tolist())
        order = np.array(order_list, dtype=int)
    else:
        order, labels, _, pca = pca_result
    tuning_sorted = tuning_all[order, :]
    labels_sorted = labels[order]
    n_pc = pca.n_components_
    comps = pca.components_

    fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
    if show_pc_loadings:
        gs = gridspec.GridSpec(n_pc, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.25)
    else:
        gs = gridspec.GridSpec(n_pc, 1, wspace=0.15, hspace=0.15)

    ax_hm = fig.add_subplot(gs[:, 0])
    ax_hm.imshow(tuning_sorted, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_hm.set_xticks(np.arange(len(left_labels)))
    ax_hm.set_xticklabels(left_labels, rotation=45, ha="right", fontsize=fontsize["tick"])
    ax_hm.set_yticks([])
    ax_hm.set_xlabel("First bar height", fontsize=fontsize["axis_label"])
    ax_hm.set_title("States sorted by max PC projection", fontsize=fontsize["title"])

    pc_groups_0 = [(k, k) for k in range(n_pc)] if pc_groups is None else [(a - 1, b - 1) for a, b in pc_groups]
    start = 0
    pc_row_bounds = []
    for pc_idx in range(n_pc):
        n_group = int(np.sum(labels_sorted == pc_idx))
        if n_group == 0:
            continue
        end = start + n_group
        ax_hm.axhline(end - 0.5, color=seg_line_color, lw=seg_line_width)
        pc_row_bounds.append((pc_idx, start - 0.5, end - 0.5, n_group))
        start = end

    for g0, g1 in pc_groups_0:
        rows = [(p, yt, yb, n) for (p, yt, yb, n) in pc_row_bounds if g0 <= p <= g1]
        if not rows:
            continue
        y_top = rows[0][1]
        y_bot = rows[-1][2]
        y_mid = 0.5 * (y_top + y_bot)
        n_total = sum(n for _, _, _, n in rows)
        label = f"PC{g0 + 1}" if g0 == g1 else f"PC{g0 + 1}\n{pc_group_dash}\nPC{g1 + 1}"
        ax_hm.text(pc_label_x, y_mid, label, va="center", ha="right", fontsize=fontsize["pc_label"], color=pc_label_color, linespacing=0.9)
        ax_hm.text(-0.6, y_bot - 0.02 * (y_bot - y_top), f"{n_total}", va="top", ha="right", fontsize=fontsize["pc_count"], color=pc_label_color)

    if show_pc_loadings:
        x = np.arange(tuning_all.shape[1])
        for pc_idx in range(n_pc):
            ax_l = fig.add_subplot(gs[pc_idx, 1])
            ax_l.plot(x, comps[pc_idx], marker="o" if pc_marker_size > 0 else None, markersize=pc_marker_size, linewidth=pc_linewidth, markeredgewidth=pc_marker_edgewidth, color="black", markerfacecolor="none")
            ax_l.set_ylabel(f"PC{pc_idx + 1}", rotation=0, fontsize=fontsize["pc_label"], labelpad=8, va="center")
            ax_l.set_yticks([])
            ax_l.tick_params(axis="x", labelsize=fontsize["tick"])
            ax_l.spines["top"].set_visible(False)
            ax_l.spines["right"].set_visible(False)
            ax_l.set_ylim(np.min(comps[pc_idx]) - 0.2 * np.ptp(comps[pc_idx]), np.max(comps[pc_idx]) + 0.2 * np.ptp(comps[pc_idx]))
            if pc_idx < n_pc - 1:
                ax_l.set_xticks([])
            else:
                ax_l.set_xticks(x)
                ax_l.set_xticklabels(left_labels, rotation=45, ha="right", fontsize=fontsize["tick"])
                ax_l.set_xlabel("First bar height", fontsize=fontsize["axis_label"])
    return fig


def plot_pv_lag_mean_sem(
    lag_curves,
    *,
    normalize="minmax",
    label_corr=True,
    color="#B4559B",
    ylabel="Overlap(Normalized PV covariance)",
    figsize_cm=(5, 8),
    dpi=300,
    fontsize=None,
    marker_size=4,
    line_width=2,
    err_line_width=2,
    tick_width=2.0,
    tick_length=5.0,
):
    if fontsize is None:
        fontsize = {"corr": 22, "axis_label": 16, "tick": 14}
    lengths = [len(curve) for curve in lag_curves]
    L_max = max(lengths)
    C = np.vstack([np.asarray(curve, float) for curve in lag_curves if len(curve) == L_max])

    if normalize == "minmax":
        mins = C.min(axis=1, keepdims=True)
        maxs = C.max(axis=1, keepdims=True)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        C = (C - mins) / denom
    elif normalize == "first":
        denom = C[:, :1].copy()
        denom[denom == 0] = 1.0
        C = C / denom
    elif normalize == "max":
        denom = np.max(np.abs(C), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        C = C / denom
    elif normalize == "zscore":
        mu = C.mean(axis=1, keepdims=True)
        sd = C.std(axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        C = (C - mu) / sd
    elif normalize == "sum":
        denom = C.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        C = C / denom

    mean_curve = C.mean(axis=0)
    std_curve = C.std(axis=0, ddof=1)
    L = mean_curve.shape[0]
    x = np.arange(1, L + 1)
    x_for_corr = np.arange(L, 0, -1, dtype=float)
    r_mean = spearmanr(x_for_corr, mean_curve)[0] if label_corr else None

    fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
    ax.errorbar(x, mean_curve, yerr=std_curve, fmt="-o", color=color, ecolor=color, elinewidth=err_line_width, capsize=0, markersize=marker_size, linewidth=line_width)
    ax.invert_xaxis()
    if label_corr and r_mean is not None and not np.isnan(r_mean):
        ax.text(0.5, 1.02, f"Corr = {r_mean:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize["corr"])
    ax.set_xticks(x_for_corr)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("Bar height similarity", fontsize=fontsize["axis_label"])
    ax.set_ylabel(ylabel, fontsize=fontsize["axis_label"])
    ax.tick_params(axis="both", width=tick_width, length=tick_length, labelsize=fontsize["tick"], direction="out")
    ax.grid(False)
    sns.despine()
    plt.tight_layout()
    return fig, ax


def plot_violin_pv_cov_real_vs_shuffle(pv_df):
    long_df = pd.DataFrame([{"condition": "Discrimination", "score": row["cov_real"]} for _, row in pv_df.iterrows()] + [{"condition": "Shuffle", "score": row["cov_shuffle"]} for _, row in pv_df.iterrows()])
    fig, ax = plt.subplots(figsize=(3.0, 5.0))
    sns.violinplot(data=long_df, x="condition", y="score", hue="condition", palette={"Discrimination": "magenta", "Shuffle": "lightgray"}, inner="box", density_norm="width", bw_adjust=5, cut=0.5, linewidth=0, legend=False, alpha=0.4, ax=ax)
    med = long_df.groupby("condition")["score"].median()
    ax.scatter([0, 1], med.values, s=150, facecolor="white", edgecolor=["magenta", "gray"], linewidth=1.3, zorder=4)
    sns.swarmplot(data=long_df[long_df["condition"] == "Discrimination"], x="condition", y="score", color="magenta", dodge=False, size=6, alpha=0.8, ax=ax, legend=False, zorder=6)
    ax.set_ylabel("Corr PVs CoV and Bar Similarity", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Discrimination", "Shuffle"], rotation=45, ha="right")
    sns.despine(ax=ax)
    ax.grid(False)
    ymax = long_df["score"].max()
    line_y = ymax + 0.15
    ax.plot([0, 1], [line_y, line_y], color="black", linewidth=1.2)
    plt.tight_layout()
    return fig


def plot_pv_df_violin_sanity(
    pv_df,
    *,
    real_col="cov_real",
    shuf_col="cov_shuffle",
    labels=("Discrimination", "Shuffle"),
    ylabel="Corr PVs cov and Bar Similarity",
    figsize_cm=(4, 4),
    dpi=300,
    fontname="Arial",
    fontsize=6,
    violin_colors=("#ff4fd8", "#d9d9d9"),
    point_alpha=0.9,
    point_size=14,
    jitter=0.15,
    p_y_offset_frac=0.06,
    pbar_lw=1.0,
):
    df = pv_df.copy()
    real = pd.to_numeric(df[real_col], errors="coerce").to_numpy(float)
    shuf = pd.to_numeric(df[shuf_col], errors="coerce").to_numpy(float)
    real = real[np.isfinite(real)]
    shuf = shuf[np.isfinite(shuf)]
    long = pd.DataFrame({"group": np.r_[np.repeat(labels[0], real.size), np.repeat(labels[1], shuf.size)], "value": np.r_[real, shuf]})

    plt.rcParams["font.family"] = fontname
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
    sns.violinplot(data=long, x="group", y="value", order=list(labels), inner="box", cut=0, linewidth=0.8, palette={labels[0]: violin_colors[0], labels[1]: violin_colors[1]}, ax=ax)
    sns.stripplot(data=long, x="group", y="value", order=list(labels), jitter=jitter, size=np.sqrt(point_size), alpha=point_alpha, color=violin_colors[0], ax=ax)
    y_min = np.nanmin(long["value"].to_numpy())
    y_max = np.nanmax(long["value"].to_numpy())
    yr = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0
    y = y_max + p_y_offset_frac * yr
    ax.plot([0, 0, 1, 1], [y, y + 0.02 * yr, y + 0.02 * yr, y], color="black", lw=pbar_lw)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize, width=0.8, length=3)
    ax.grid(False)
    sns.despine(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
    plt.tight_layout()
    return fig, ax


def add_category_block_labels(
    ax,
    blocks,
    *,
    fontname="Arial",
    fontsize_label=8,
    fontsize_count=7,
    label_color="black",
    count_color="black",
    label_x=-2.5,
    label_ha="right",
    draw_separators=True,
    sep_color="white",
    sep_lw=0.5,
    show_counts=True,
    counts_side="left",
    counts_pad=6,
    count_x=None,
    count_ha="right",
):
    if draw_separators:
        for block in blocks:
            if block["start"] > 0:
                ax.axhline(block["start"] - 0.5, color=sep_color, lw=sep_lw)
    for block in blocks:
        y_mid = (block["start"] + block["end"] - 1) / 2.0
        ax.text(label_x, y_mid, block["name"], va="center", ha=label_ha, fontsize=fontsize_label, color=label_color, fontname=fontname, clip_on=False)
    if show_counts:
        y_pos = []
        cum = []
        running = 0
        for block in blocks:
            running += block["end"] - block["start"]
            y_pos.append(block["end"] - 1)
            cum.append(running)
        if counts_side == "left":
            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(c) for c in cum], fontname=fontname, fontsize=fontsize_count, color=count_color)
            ax.tick_params(axis="y", length=0, pad=counts_pad)
        elif counts_side == "right":
            if count_x is None:
                count_x = ax.get_xlim()[0] - 0.5
            for y, c in zip(y_pos, cum):
                ax.text(count_x, y, str(c), va="center", ha=count_ha, fontsize=fontsize_count, color=count_color, fontname=fontname, clip_on=False)
        else:
            raise ValueError("counts_side must be 'left' or 'right'.")


def plot_tuning_heatmap(
    tuning,
    obs_labels,
    *,
    blocks=None,
    cat_ids=None,
    id_to_name=None,
    enable_cat_blocks=True,
    state_mask=None,
    sort="peak",
    normalize=None,
    return_state_order=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    nan_color=None,
    ax=None,
    title=None,
    figsize_cm=(14, 18),
    dpi=300,
    fontname="Arial",
    fontsize=None,
    xtick_rotation=45,
    xtick_ha="right",
    tick_width=0.4,
    tick_length=2.0,
    axis_linewidth=0.4,
    show_spines=True,
    spine_color="black",
    xlabel="Observations",
    ylabel="HMM state",
    ylabel_pad=40,
    show_cbar=True,
    cbar_label=None,
    cbar_size=0.06,
    cbar_pad=0.12,
    cbar_ticks=None,
    cbar_ticklabels=None,
    cbar_outline=True,
    cbar_outline_width=0.4,
    cbar_outline_color="black",
    draw_cat_separators=True,
    cat_label_x=-2.5,
    cat_counts_pad=6,
    cat_counts_side="left",
):
    if fontsize is None:
        fontsize = {"title": 8, "axis_label": 7, "tick": 6, "cbar_label": 7, "cbar_tick": 6, "cat_label": 8, "cat_count": 7}
    tuning = np.asarray(tuning, float)
    S, O = tuning.shape
    obs_labels = list(obs_labels)
    selected_states = np.arange(S) if state_mask is None else (np.where(np.asarray(state_mask))[0] if np.asarray(state_mask).dtype == bool else np.asarray(state_mask, dtype=int))
    tuning_sel = tuning[selected_states]
    order = np.argsort(np.argmax(tuning_sel, axis=1)) if sort == "peak" else np.arange(tuning_sel.shape[0])
    tuning_plot = tuning_sel[order]
    state_ordered = selected_states[order]
    M = tuning_plot.copy()
    inferred_cb = "P(state | obs)"
    if normalize == "sum":
        rs = M.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        M = M / rs
    elif normalize == "max":
        mx = M.max(axis=1, keepdims=True)
        mx[mx == 0] = 1.0
        M = M / mx
    elif normalize == "zscore":
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        M = (M - mu) / sd
        inferred_cb = "z-scored"

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54), dpi=dpi)
        created = True
    else:
        fig = ax.figure

    if nan_color is not None:
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color=nan_color)
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    else:
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(O))
    ax.set_xticklabels(obs_labels, rotation=xtick_rotation, ha=xtick_ha, fontsize=fontsize["tick"], fontname=fontname)
    ax.set_yticks([])
    if enable_cat_blocks and cat_ids is not None and id_to_name is not None:
        cat_ids = np.asarray(cat_ids, int)
        cat_ids_sorted = cat_ids[state_ordered]
        inferred_blocks = []
        start = 0
        for idx in range(1, len(cat_ids_sorted) + 1):
            if idx == len(cat_ids_sorted) or cat_ids_sorted[idx] != cat_ids_sorted[idx - 1]:
                cid = int(cat_ids_sorted[idx - 1])
                inferred_blocks.append({"name": str(id_to_name.get(cid, cid)), "start": start, "end": idx, "cid": cid})
                start = idx
        add_category_block_labels(
            ax,
            inferred_blocks,
            fontname=fontname,
            fontsize_label=fontsize["cat_label"],
            fontsize_count=fontsize["cat_count"],
            draw_separators=draw_cat_separators,
            label_x=cat_label_x,
            counts_pad=cat_counts_pad,
            counts_side=cat_counts_side,
        )
        blocks = inferred_blocks
    elif blocks is not None:
        add_category_block_labels(ax, blocks, fontname=fontname, fontsize_label=fontsize["cat_label"], fontsize_count=fontsize["cat_count"])
    else:
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels([str(int(s)) for s in state_ordered], fontsize=fontsize["tick"], fontname=fontname)

    ax.set_xlabel(xlabel, fontsize=fontsize["axis_label"], fontname=fontname)
    ax.set_ylabel(ylabel, fontsize=fontsize["axis_label"], fontname=fontname, labelpad=ylabel_pad)
    if title:
        ax.set_title(title, fontsize=fontsize["title"], fontname=fontname)
    ax.tick_params(axis="both", width=tick_width, length=tick_length, direction="out")
    for spine in ax.spines.values():
        spine.set_visible(bool(show_spines))
        spine.set_linewidth(axis_linewidth)
        spine.set_edgecolor(spine_color)
    cb = None
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{float(cbar_size) * 100:.1f}%", pad=cbar_pad)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(inferred_cb if cbar_label is None else cbar_label, fontsize=fontsize["cbar_label"], fontname=fontname)
        cb.ax.tick_params(width=tick_width, length=tick_length, labelsize=fontsize["cbar_tick"], direction="out")
        if cbar_ticks is not None:
            cb.set_ticks(list(cbar_ticks))
        if cbar_ticklabels is not None:
            cb.set_ticklabels(list(cbar_ticklabels))
        if cbar_outline:
            cb.outline.set_visible(True)
            cb.outline.set_linewidth(cbar_outline_width)
            cb.outline.set_edgecolor(cbar_outline_color)
        else:
            cb.outline.set_visible(False)
    fig.tight_layout(pad=0.02)
    if return_state_order:
        if enable_cat_blocks and cat_ids is not None:
            cat_ids_sorted = cat_ids[state_ordered]
            return (fig if created else None), ax, state_ordered, cat_ids_sorted
        return (fig if created else None), ax, state_ordered, cb
    return (fig if created else None), ax, cb


def plot_obj1_tuning_heatmap(
    df_trials_all,
    date,
    *,
    exp_date_col="exp_date",
    session_id_fmt="%Y%m%d",
    win_col="fc3_window_obj1",
    h_col="object_1_h",
    start_col="object_1_start",
    end_col="object_1_end",
    active_frac_thr=0.2,
    use_positive_thr=0.0,
    require_any_activity=True,
    height_values=None,
    height_match="nearest",
    normalize="minmax",
    sort_by="pref",
    figsize=(5, 7),
    dpi=300,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    title=None,
    xlabel="Obj1 height",
    fontname=None,
    fontsize=None,
    xtick_rotation=45,
    xtick_ha="right",
    ax=None,
):
    if fontsize is None:
        fontsize = {"title": 10, "axis_label": 10, "xtick": 9, "ytick": 9}
    session_id = pd.to_datetime(date).strftime(session_id_fmt)
    df_session = df_trials_all[df_trials_all[exp_date_col] == session_id]
    if df_session.empty:
        raise ValueError(f"No rows found for session_id={session_id} in col '{exp_date_col}'")

    fc3_windows1 = df_session[win_col].values
    object_1_h = df_session[h_col].values
    object_1_start = df_session[start_col].values
    object_1_end = df_session[end_col].values
    ax_height_1 = np.sort(np.unique(object_1_h)) if height_values is None else np.asarray(height_values)

    n_total_neurons = fc3_windows1[0].shape[1]
    neuron_active = np.zeros(n_total_neurons, dtype=bool)
    for window in fc3_windows1:
        frac_active = np.mean(np.asarray(window) > use_positive_thr, axis=0)
        neuron_active |= frac_active > active_frac_thr
    active_idx = np.where(neuron_active)[0]
    if active_idx.size == 0:
        raise ValueError("No active neurons found under the given thresholds.")

    dF_sum_1 = np.zeros((active_idx.size, len(ax_height_1)), dtype=float)
    occupancy_1 = np.zeros((active_idx.size, len(ax_height_1)), dtype=float)
    for i in range(len(fc3_windows1)):
        h1 = float(object_1_h[i])
        if height_match == "nearest":
            idx1 = int(np.argmin(np.abs(h1 - ax_height_1)))
        elif height_match == "exact":
            matches = np.where(ax_height_1 == h1)[0]
            if matches.size == 0:
                continue
            idx1 = int(matches[0])
        else:
            raise ValueError("height_match must be 'nearest' or 'exact'")

        win1 = np.asarray(fc3_windows1[i])[:, active_idx]
        dur1 = float(object_1_end[i] - object_1_start[i])
        if require_any_activity and np.sum(win1) <= 0:
            continue
        dF_sum_1[:, idx1] += np.sum(win1, axis=0)
        occupancy_1[:, idx1] += dur1

    with np.errstate(divide="ignore", invalid="ignore"):
        tuning = np.nan_to_num(dF_sum_1 / occupancy_1, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize in (None, "none"):
        normed = tuning.copy()
        plot_vmin, plot_vmax = vmin, vmax
    elif normalize == "minmax":
        row_min = np.min(tuning, axis=1, keepdims=True)
        row_max = np.max(tuning, axis=1, keepdims=True)
        normed = (tuning - row_min) / (row_max - row_min + 1e-9)
        plot_vmin, plot_vmax = vmin, vmax
    elif normalize == "max":
        row_max = np.max(tuning, axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        normed = tuning / row_max
        plot_vmin, plot_vmax = vmin, vmax
    elif normalize == "zscore":
        mu = tuning.mean(axis=1, keepdims=True)
        sd = tuning.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        normed = (tuning - mu) / sd
        plot_vmin, plot_vmax = vmin, vmax
    else:
        raise ValueError("normalize must be None/'none', 'minmax', 'max', or 'zscore'.")

    if sort_by in (None, "none"):
        sort_idx = np.arange(active_idx.size)
    elif sort_by == "pref":
        sort_idx = np.argsort(np.argmax(normed, axis=1))
    elif sort_by == "peak_value":
        sort_idx = np.argsort(np.max(normed, axis=1))
    else:
        raise ValueError("sort_by must be None/'none', 'pref', or 'peak_value'.")

    sorted_tuning = normed[sort_idx, :]
    if fontname is not None:
        plt.rcParams["font.family"] = fontname

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created = True
    else:
        fig = ax.figure

    ax.imshow(sorted_tuning, aspect="auto", cmap=cmap, vmin=plot_vmin, vmax=plot_vmax, interpolation="nearest")
    ax.set_title(f"session {session_id}" if title is None else title, fontsize=fontsize.get("title", 10))
    ax.set_xlabel(xlabel, fontsize=fontsize.get("axis_label", 10))
    ax.set_xticks(np.arange(len(ax_height_1)))
    ax.set_xticklabels(ax_height_1, rotation=xtick_rotation, ha=xtick_ha, fontsize=fontsize.get("xtick", 9))
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if created:
        plt.tight_layout()

    out = {
        "session_id": session_id,
        "heights": ax_height_1,
        "active_idx": active_idx,
        "tuning": tuning,
        "normed": normed,
        "sort_idx": sort_idx,
        "sorted_tuning": sorted_tuning,
    }
    return fig, ax, out


def plot_metric_heatmap(
    cg_df,
    metric="similarity",
    agg="mean",
    normalize=False,
    title=None,
    cmap="viridis",
    annotate=True,
    vmin=None,
    vmax=None,
    annot_fmt=".3f",
    black_thresh=0.3,
    cbar_title=None,
    y_label="TRAIN session",
    x_label="TEST session",
    *,
    group_by="animal_date",
    figsize_cm=None,
    ax_rect=(0.22, 0.20, 0.70, 0.62),
    top_band_size="0.5%",
    left_band_size="0.5%",
    band_pad=0.6,
):
    def _bounds(labels):
        out, start = [], 0
        for idx in range(1, len(labels)):
            if labels[idx] != labels[idx - 1]:
                out.append((start, idx - 1))
                start = idx
        out.append((start, len(labels) - 1))
        return out

    if group_by == "animal_date":
        P = cg_df.pivot_table(index=("train_animal", "train_date"), columns=("test_animal", "test_date"), values=metric, aggfunc=agg).sort_index(axis=0).sort_index(axis=1)
        row_anim = P.index.get_level_values(0).to_numpy()
        col_anim = P.columns.get_level_values(0).to_numpy()
    else:
        P = cg_df.pivot_table(index=("train_animal",), columns=("test_animal",), values=metric, aggfunc=agg).sort_index(axis=0).sort_index(axis=1)
        row_anim = P.index.to_numpy()
        col_anim = P.columns.to_numpy()
    if normalize:
        row_max = P.max(axis=1, skipna=True).replace(0, np.nan)
        P = P.div(row_max, axis=0)

    figsize = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54) if figsize_cm is not None else (max(8, 0.42 * P.shape[1]), max(7, 0.42 * P.shape[0]))
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    ax = fig.add_axes(ax_rect)
    im = ax.imshow(P.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    ax_top = divider.new_horizontal(size=top_band_size, pad=band_pad, pack_start=True)
    ax_left = divider.new_vertical(size=left_band_size, pad=band_pad, pack_start=True)
    fig.add_axes(ax_top)
    fig.add_axes(ax_left)
    ax.set_xlabel(x_label, labelpad=6, fontsize=4)
    ax.set_ylabel(y_label, labelpad=6, fontsize=4)
    if title:
        fig.suptitle(title, y=0.96, fontsize=14, fontweight="bold")
    ax.grid(False)
    for k, (s, e) in enumerate(_bounds(row_anim)):
        if (e - s) >= 1 and k % 2 == 0:
            ax.axhspan(s - 0.5, e + 0.5, color="white", alpha=0.08, zorder=0)
    for k, (s, e) in enumerate(_bounds(col_anim)):
        if (e - s) >= 1 and k % 2 == 0:
            ax.axvspan(s - 0.5, e + 0.5, color="white", alpha=0.05, zorder=0)
    if annotate:
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:{annot_fmt}}", ha="center", va="center", fontsize=4, color="black" if v < black_thresh else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=25)
    if cbar_title is not None:
        cbar.set_label(cbar_title, fontsize=4)
    cbar.ax.tick_params(labelsize=4)
    return fig, ax, P


def _obs_at_step_s_dataset(dataset, s: int):
    """
    Group trials by obs token at within-trial step s.
    Returns:
      obs_s: (n_kept,) obs token at step s per trial (only trials with L>s)
      trial_idx: (n_kept,) trial indices
    """
    lengths = np.asarray(dataset.lengths, dtype=int)
    if len(lengths) < 2:
        trial_len = 6
        T = len(dataset.obs)
        n_trials = T // trial_len   # integer number of full trials

        lengths = np.full(n_trials, trial_len, dtype=int)
    starts = np.cumsum(np.r_[0, lengths[:-1]])  # compute starts here
    obs_s, trial_idx = [], []
    for i, L in enumerate(lengths):
        if s < L:  # trial long enough to have step s
            obs_s.append(int(dataset.obs[starts[i] + s]))
            trial_idx.append(i)
    return np.array(obs_s, dtype=int), np.array(trial_idx, dtype=int)


# ----------------- public API -----------------
def plot_transition_matrices(model=None, trans=None, action_labels=None,
                             state_order=None, axs=None, title=None,
                             cmap='viridis', vmin=0.0, vmax=None, annotate=False,
                             *,
                             state_labels=None,
                             per_action_vmax=False,
                             cbar='shared',          # 'shared' | 'per-axis' | 'none'
                             annot_fmt=".2f",
                             annot_min=1e-3,
                             mask_below=None,        # e.g., 1e-4 -> mask values < 1e-4
                             grid=False):
    """
    Visualize P(s' | s, a) for each action as separate panels.

    Parameters
    ----------
    model : object, optional
        If provided, uses `model.trans` (shape A x S x S).
    trans : np.ndarray, optional
        Transition tensor of shape (A, S, S). Used if `model` is None.
    action_labels : list[str], optional
        Names for each action; length must equal A if given.
    state_order : list[int], optional
        Permutation/order of states to show on both axes.
    axs : matplotlib Axes or list[Axes], optional
        Axes to draw into. If None, a new figure with A subplots is created.
    title : str, optional
        Figure-level title (when creating a new figure).
    cmap : str, optional
        Colormap for the heatmaps.
    vmin, vmax : float, optional
        Color limits. If `per_action_vmax=False` and vmax is None, uses global max.
    annotate : bool, optional
        If True, writes numeric probabilities in each cell.

    Keyword-only extras
    -------------------
    state_labels : list[str], optional
        Pretty labels for states; mapped after reordering.
    per_action_vmax : bool
        Use a separate vmax per action panel.
    cbar : {'shared','per-axis','none'}
        How to draw colorbars.
    annot_fmt : str
        Format string for annotations (e.g., '.2f', '.3f', '0.2e').
    annot_min : float
        Do not annotate values smaller than this threshold.
    mask_below : float or None
        If set, values < mask_below are shown as masked (uses colormap 'bad' color).
    grid : bool
        If True, draw faint gridlines between cells.
    """
    # Get transitions
    if trans is None:
        if model is None or not hasattr(model, 'trans'):
            raise ValueError("Provide `trans` or a `model` with `.trans` of shape (A, S, S).")
        trans = model.trans
    trans = np.asarray(trans)
    if trans.ndim != 3:
        raise ValueError(f"`trans` must have shape (A, S, S); got shape {trans.shape}.")

    A, S, S2 = trans.shape
    if S != S2:
        raise ValueError("Transition matrices must be square along the last two dims.")

    # State order
    if state_order is None:
        state_order = np.arange(S)
    else:
        state_order = np.asarray(state_order)
        if len(state_order) != S:
            raise ValueError("`state_order` must list all S states.")

    # Labels
    if action_labels is None:
        action_labels = [f"a={i}" for i in range(A)]
    if len(action_labels) != A:
        raise ValueError("`action_labels` length must equal number of actions A.")

    ticklabs = state_order if state_labels is None else [state_labels[i] for i in state_order]

    # Reorder states for each action
    Ts = trans[:, state_order][:, :, state_order]  # (A, S, S)

    # Color range
    if vmax is None:
        if per_action_vmax:
            vmaxs = []
            for a in range(A):
                m = float(np.nanmax(Ts[a])) if np.isfinite(Ts[a]).any() else 1.0
                vmaxs.append(1.0 if m == 0 else m)
        else:
            m = float(np.nanmax(Ts)) if np.isfinite(Ts).any() else 1.0
            vmax = 1.0 if m == 0 else m
            vmaxs = [vmax] * A
    else:
        vmaxs = [float(vmax)] * A

    # Axes / figure
    created_fig = False
    if axs is None:
        fig, axs = plt.subplots(1, A, figsize=(6*A + 2, 4 + 1), squeeze=False)
        axs = axs[0]
        created_fig = True
        if title:
            fig.suptitle(title)
    else:
        if isinstance(axs, np.ndarray):
            axs = axs.flatten().tolist()
        elif not isinstance(axs, (list, tuple)):
            axs = [axs]
        if len(axs) != A:
            fig, axs = plt.subplots(1, A, figsize=(6*A + 2, 4 + 1))
            created_fig = True
            if title:
                fig.suptitle(title)
        else:
            fig = axs[0].figure
            if title and created_fig:
                fig.suptitle(title)

    ims = []
    # Prepare colormap (and bad color for masking)
    cmap_obj = plt.get_cmap(cmap)
    if mask_below is not None:
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad(color='#EEEEEE')  # light gray for masked cells

    for a in range(A):
        ax = axs[a]
        M = Ts[a]
        M_plot = M.copy()
        if mask_below is not None:
            M_plot = M_plot.astype(float)
            M_plot[M_plot < mask_below] = np.nan

        im = ax.imshow(M_plot, origin='upper', aspect='auto',
                       cmap=cmap_obj, vmin=vmin, vmax=vmaxs[a])
        ims.append(im)
        ax.set_title(action_labels[a])
        ax.set_xlabel("next state $s'$")
        if a == 0:
            ax.set_ylabel("current state $s$")
        ax.set_xticks(np.arange(S))
        ax.set_yticks(np.arange(S))
        ax.set_xticklabels(ticklabs)
        ax.set_yticklabels(ticklabs)

        if grid:
            ax.set_xticks(np.arange(-.5, S, 1), minor=True)
            ax.set_yticks(np.arange(-.5, S, 1), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
            ax.tick_params(which='minor', bottom=False, left=False)

        if annotate:
            for i in range(S):
                for j in range(S):
                    val = M[i, j]
                    if not np.isfinite(val):
                        continue
                    if val >= max(annot_min, 0.0):
                        ax.text(j, i, format(val, annot_fmt),
                                ha='center', va='center', fontsize=7)

    # Colorbars
    if cbar != 'none':
        if cbar == 'per-axis':
            for a, im in enumerate(ims):
                cb = fig.colorbar(im, ax=axs[a], fraction=0.046, pad=0.02)
                cb.set_label("P($s'|s,a$)")
        else:  # 'shared'
            cb = fig.colorbar(ims[0], ax=axs, fraction=0.03, pad=0.02)
            cb.set_label("P($s'|s,a$)")

    return axs


def plot_emission_matrix(model, vocab, height_order=None, ax=None, title=None,
                         split_style="LR"):  # or "interleave"
    """
    Visualize P(o | s) (rows: hidden states, cols: observations).

    Column order:
      - shared: heights ascending (h_{h}) + ['gap','reward','no_reward','start','end']
      - split : by default L-then-R blocks:
                [l_{h} for h in ascending] + [r_{h} for h in ascending] + tail
                set split_style="interleave" to [l_h, r_h, l_h2, r_h2, ...] + tail
    """
    emit = model.emit  # (S, O)
    S, O = emit.shape

    # detect height tokens present
    obs_to_id = vocab.obs_to_id
    keys = list(obs_to_id.keys())
    pat = re.compile(r'^(?:h|h1|h2|l|r)_?(\d+)$', flags=re.IGNORECASE)

    # collect per-kind
    shared = {}   # h -> 'h_{h}'
    left   = {}   # h -> 'l_{h}'
    right  = {}   # h -> 'r_{h}'
    for tok in keys:
        m = pat.match(tok)
        if not m:
            continue
        h = int(m.group(1))
        t0 = tok.lower()
        if t0.startswith("l_"):
            left[h]  = tok
        elif t0.startswith("r_"):
            right[h] = tok
        elif t0.startswith("h"):
            shared[h] = tok

    is_split = bool(left or right)  # if any l_/r_ seen, treat as split

    # height ordering (numbers)
    if height_order is None:
        if is_split:
            nums = sorted(set(left.keys()) | set(right.keys()))
        else:
            nums = sorted(shared.keys())
    else:
        nums = [int(h) for h in height_order]

    # build height columns and labels
    height_cols = []
    height_labels = []
    if is_split:
        if split_style == "interleave":
            for h in nums:
                if h in left:
                    height_cols.append(obs_to_id[left[h]])
                    height_labels.append(left[h])
                if h in right:
                    height_cols.append(obs_to_id[right[h]])
                    height_labels.append(right[h])
        else:  # "LR" blocks
            for h in nums:
                if h in left:
                    height_cols.append(obs_to_id[left[h]])
                    height_labels.append(left[h])
            for h in nums:
                if h in right:
                    height_cols.append(obs_to_id[right[h]])
                    height_labels.append(right[h])
    else:
        for h in nums:
            tok = shared.get(h, f"h_{h}")
            if tok in obs_to_id:
                height_cols.append(obs_to_id[tok])
                height_labels.append(tok)

    # tail tokens
    tail = [t for t in ['gap','reward','no_reward','start','end'] if t in obs_to_id]
    tail_cols = [obs_to_id[t] for t in tail]

    cols = height_cols + tail_cols
    labels = height_labels + tail

    M = emit[:, cols]  # (S, len(cols))

    if ax is None:
        fig, ax = plt.subplots(figsize=(0.5*len(cols)+2, 0.4*S+2))
    im = ax.imshow(M, aspect='auto', cmap='viridis')
    ax.set_xlabel("Observations")
    ax.set_ylabel("Hidden states")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(S))
    ax.set_title(title or "Emission probability matrix")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(obs | state)")
    return fig,ax


def plot_policy_matrix(model=None, policy=None,
                       action_labels=None, state_labels=None,
                       state_order=None, ax=None, title=None,
                       cmap='Greens', vmin=0.0, vmax=1.0,
                       annotate=False, annot_fmt=".2f", annot_min=1e-3,
                       show_cbar=True, show_entropy=False):
    """
    Heatmap of π(a|s). Rows=states, Cols=actions.

    Parameters
    ----------
    model, policy : if model given, uses model.policy (S, A)
    action_labels : list[str] of length A
    state_labels  : list[str] of length S
    state_order   : list[int] permutation of states (applied to rows)
    annotate      : write numbers in cells above annot_min
    show_entropy  : draw a thin bar on the right with normalized entropy H(s)/log(A))
    """
    if policy is None:
        if model is None or not hasattr(model, "policy"):
            raise ValueError("Provide `policy` or a `model` with `.policy` (S, A).")
        policy = model.policy
    P = np.asarray(policy, float)  # (S, A)
    S, A = P.shape

    if action_labels is None:
        action_labels = [f"a={i}" for i in range(A)]
    if state_labels is None:
        state_labels = [f"s{i}" for i in range(S)]

    if state_order is None:
        state_order = np.arange(S)
    else:
        state_order = np.asarray(state_order)
        if len(state_order) != S:
            raise ValueError("`state_order` must list all S states.")

    P_ord = P[state_order, :].T  # transpose to make actions rows and states columns make actions rows and states columns
    xticklabs = [state_labels[i] for i in state_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.4*S + 2.5, 0.4*A + 2))

    im = ax.imshow(P_ord, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("State s")
    ax.set_ylabel("Action a")
    ax.set_xticks(np.arange(S)); ax.set_xticklabels(xticklabs, rotation=45, ha='right')
    ax.set_yticks(np.arange(A)); ax.set_yticklabels(action_labels)
    if title: ax.set_title(title)

    if annotate:
        for i in range(S):
            for j in range(A):
                val = P_ord[i, j]
                if val >= annot_min:
                    ax.text(j, i, format(val, annot_fmt), ha='center', va='center', fontsize=7)

    # optional colorbar
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(r"$\pi(a\,|\,s)$")

    # optional entropy bar
    if show_entropy:
        eps = 1e-12
        H = -(P_ord * np.log(P_ord + eps)).sum(axis=1)  # natural log
        H_norm = H / (np.log(A) if A > 1 else 1.0)     # 0..1
        divider = make_axes_locatable(ax)
        ax_h = divider.append_axes("right", size="3%", pad=0.2)
        ax_h.imshow(H_norm[:, None], aspect='auto', cmap='Greys', vmin=0.0, vmax=1.0)
        ax_h.set_xticks([]); ax_h.set_yticks([])
        ax_h.set_title("H", fontsize=9, pad=4)

    return fig, ax


def show_pv_representation_ahmm_dataset(
    t: int, s: int, dataset, model, tower_heights,
    id_to_obs=None, cmap=parula, ax=None,
    *, filter_states_by_height_emit: bool = True, emit_thresh: float = 1e-8,
    renormalize: bool = True, plot=True, metric: str = "cov"  # 'cov' | 'corr' | 'cosine'
):
    """
    Use decode_posteriors(dataset, model) to get gammas per trial.
    Group by obs at step s; average PV at time t per group; visualize pairwise
    similarity among groups using:
      - 'cov'   : covariance (default)
      - 'corr'  : Pearson correlation
      - 'cosine': cosine distance (as before)
    Optionally filter to states that have nontrivial emission mass on any height token.
    """
    gammas = decode_posteriors_filtered(dataset, model)

    # ---- build a state mask using emission over height tokens ----
    state_mask = None
    if filter_states_by_height_emit and hasattr(model, "emit") and (id_to_obs is not None):
        try:
            pat = re.compile(r'^(?:h|h1|h2|l|r)_?\d+$', flags=re.IGNORECASE)
            height_cols = [oid for oid, name in id_to_obs.items() if isinstance(name, str) and pat.match(name)]
            if len(height_cols) > 0:
                height_cols = np.array(sorted(height_cols), dtype=int)
                emit = np.asarray(model.emit, dtype=float)  # (S, O)
                height_mass = emit[:, height_cols].sum(axis=1)  # (S,)
                state_mask = (height_mass > emit_thresh)
                if not np.any(state_mask):
                    state_mask = None
        except Exception:
            state_mask = None

    # group by obs at step s
    obs_s, trial_idxs = _obs_at_step_s_dataset(dataset, s)
    if obs_s.size == 0:
        if ax is None: fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(f"No trials contain step s={s}")
        return None

    groups = np.unique(obs_s)
    obs_to_pvs = {g: [] for g in groups}

    for i, g in zip(trial_idxs, obs_s):
        if t < gammas[0].shape[0]:
            if len(dataset.lengths) < 2:
                pv = gammas[0][t+i*6]
            else:
                pv = gammas[i][t]  # (S,)
            if state_mask is not None:
                pv = pv[state_mask]
                if renormalize:
                    ssum = pv.sum()
                    if ssum > 0:
                        pv = pv / ssum
            obs_to_pvs[g].append(pv)

    # average PVs per group
    obs_to_mean = {g: np.mean(np.stack(pvs, 0), 0) for g, pvs in obs_to_pvs.items() if len(pvs) > 0}
    if not obs_to_mean:
        if ax is None: fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(f"No PVs available at t={t} for any group (s={s})")
        return None

    keys = sorted(obs_to_mean.keys())
    M = np.stack([obs_to_mean[g] for g in keys], 0)  # (G, S')
    good = ~np.isnan(M).any(axis=1)
    M, keys = M[good], np.array(keys)[good]
    if M.size == 0:
        if ax is None: fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title("All PV rows filtered/NaN after masking")
        return None

    # ---- compute similarity matrix according to metric ----
    metric = str(metric).lower()
    if metric == "cosine":
        D = cosine_distances(M)                 # distance in [0,2]
        matrix = D
        vmin, vmax = 0.0, float(np.nanmax(D)) if np.isfinite(D).any() else 1.0
        cbar_label = "Cosine distance"
        cm = cmap
    elif metric == "corr":
        X = M.copy().astype(float)
        X += 1e-6
        X /= X.sum(axis=1, keepdims=True)
        # print(X)
        EPS = 1e-12
        rowsum = X.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0          # avoid /0 if a row is all zeros
        X /= rowsum

        # --- NaN/inf safe, no warnings, matches corrcoef for non-degenerate rows ---
        Xc  = X - X.mean(axis=1, keepdims=True)                   # center rows
        std = Xc.std(axis=1, ddof=1, keepdims=True)               # row std (N-1)
        # z-score rows; where std==0, leave zeros (so correlations become 0)
        Z = np.divide(Xc, std, out=np.zeros_like(Xc), where=std > EPS)

        den = max(X.shape[1] - 1, 1)
        C = (Z @ Z.T) / den                                       # row-row corr
        # constant rows get 0 correlation with others; set their diagonal to 1
        np.fill_diagonal(C, 1.0)
        C = np.clip(C, -1.0, 1.0)

        # your distance + scaling (unchanged semantics)
        matrix = 1.0 - C
        mmax = matrix.max()
        matrix = matrix / mmax if mmax > 0 else np.zeros_like(matrix)
        cbar_label = "Correlation"
        cm = cmap
        vmin = 0
        vmax = 1
    elif metric == "cov":
        
        C = np.cov(M)                           # covariance among group vectors (rows)
        # np.cov returns scalar if G==1
        if np.ndim(C) == 0:
            C = np.array([[float(C)]])
        matrix = C
        # vmax_abs = float(np.nanmax(np.abs(C))) if np.isfinite(C).any() else 1.0
        vmin, vmax = float(np.nanmin(C)), float(np.nanmax(C))
        cbar_label = "Covariance"
        cm = plt.get_cmap(cmap).reversed() if isinstance(cmap, str) else cmap.reversed()
    else:
        raise ValueError("metric must be one of {'cov','corr','cosine'}")

    # labels
    if id_to_obs is not None:
        labels = [id_to_obs[int(x)] for x in keys]
    else:
        try:
            labels = np.asarray(tower_heights)[keys - 1]
        except Exception:
            labels = keys
    if plot:
        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        
        # cm = parula.reversed()
        im = ax.imshow(matrix, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(np.arange(len(keys))); ax.set_yticklabels(labels)
        ax.set_xlabel(f"Obs at t={s}"); ax.set_ylabel(f"Obs at t={s}")
        title_suffix = " (height-emitting states only)" if state_mask is not None else ""
        ax.set_title(f"AHMM {cbar_label} (PV @ t={t}){title_suffix}")
        ax.invert_yaxis()
        # when creating the image
        
        plt.colorbar(im, ax=ax)

    return M, matrix, gammas


def plot_decoded_graph_ahmm_with_pies(
    model, obs, acts, *,
    obs_labels, obs_colors,
    mode: str = "model", action_weights: str | np.ndarray = "empirical",
    layout: str = "fr", layout_kwargs: Optional[Dict[str, Any]] = None, spread: float = 1.6,
    vertex_size: float = 0.06, threshold: float = 0.02,
    pie_min_frac: float = 0.0, pie_top_k: Optional[int] = 8,
    label_vertices: bool = True, title: Optional[str] = None,
    ax: Optional[plt.Axes] = None, pad: float = 0.15,
    legend: bool = True, legend_title: str = "Emissions",
    legend_ncol: int = 3, legend_fontsize: int = 8, legend_title_fontsize: int = 10,
    legend_marker_size: int = 5, legend_handlelength: float = 0.8,
    legend_handletextpad: float = 0.35, legend_columnspacing: float = 0.75,
    action_legend_fontsize: Optional[int] = None, action_legend_linewidth: float = 2.0,
    dpi: Optional[int] = None, action_labels: Optional[List[str]] = None,
    action_colors: Optional[Dict[str, Any]] = None, edge_color_mode: str = "argmax", weight_style="alpha", tail_width_min=1.2, tail_width_max=2, head_min=10,head_max=16, save=False
):
    """
    Viterbi-decoded visited-state graph with per-state emission pies.
    Edges are colored by dominant contributing action.
    """
    # decode & index set
    states = np.asarray(_decode_states(model, obs, acts), int)
    v = np.unique(states)

    # adjacency and per-action contributions
    A_adj, C_act = _edge_contrib_by_action(
        model, states, acts, v, mode=mode, action_weights=action_weights, normalize_rows=True
    )
    np.fill_diagonal(A_adj, 0.0)
    if threshold > 0: A_adj[A_adj < threshold] = 0.0

    # layout & node radius
    xy, _ = _layout_coords_from_igraph(A_adj, layout=layout, layout_kwargs=layout_kwargs, spread=spread)
    span = (xy.max(0) - xy.min(0)).max()
    radius = float(vertex_size) * (span if span > 0 else 1.0)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi, constrained_layout=False)
    else:
        fig = ax.figure
        if dpi is not None: fig.set_dpi(dpi)
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    ax.set_aspect("equal"); ax.axis("off")

    # action colors & edge colors
    if action_labels is None:
        action_labels = [f"a{i}" for i in range(C_act.shape[0])]
    A_cnt = len(action_labels)
    if action_colors is None:
        import matplotlib as mpl
        tab = mpl.colormaps.get("tab10").colors
        action_colors = {
            action_labels[i]: DEFAULT_ACTION_COLORS.get(action_labels[i], tab[i % len(tab)])
            for i in range(A_cnt)
        }
    elif isinstance(action_colors, (list, tuple)):
        action_colors = {action_labels[i]: action_colors[i] for i in range(A_cnt)}

    edge_colors = np.full(A_adj.shape, "#3E3C3C", dtype=object)
    if edge_color_mode == "argmax":
        arg = np.argmax(C_act, axis=0)  # (n,n)
        for i in range(A_adj.shape[0]):
            for j in range(A_adj.shape[1]):
                if A_adj[i, j] > 0:
                    a = int(arg[i, j])
                    lbl = action_labels[a]
                    edge_colors[i, j] = action_colors.get(lbl, "#3E3C3C")
    else:
        raise ValueError("edge_color_mode currently supports only 'argmax'.")

    # draw edges & nodes
    _draw_weighted_arrows(
        ax, xy, A_adj, threshold=threshold, node_radius=radius,
        color="#3E3C3C", edge_colors=edge_colors,
        weight_style=weight_style, alpha_min=0.25, alpha_max=0.95,
        tail_width_min=tail_width_min, tail_width_max=tail_width_max, gamma=0.6,
        head_min=head_min, head_max=head_max, curvature=0.12
    )

    # --- annotate edges with transition probabilities ---
    for i in range(A_adj.shape[0]):
      for j in range(A_adj.shape[1]):
          if A_adj[i, j] > 0:
              # coordinates of the two nodes
              x0, y0 = xy[i]
              x1, y1 = xy[j]
              # mid-point of the arrow
              xm, ym = (0.6*x0 + 0.4*x1, 0.6*y0 + 0.4*y1)
              # probability value
              prob = A_adj[i, j]
              txt = ax.text(xm, ym, f"{prob:.2f}", fontsize=4,
                            color=edge_colors[i, j], ha="center", va="center")
              # add white border
              txt.set_path_effects([
                  pe.Stroke(linewidth=2.5, foreground="white"),
                  pe.Normal()
              ])


    emit_rows = np.asarray(model.emit)[v, :]
    _draw_emission_pies(
        ax, xy, emit_rows, obs_labels, obs_colors,
        radius=radius, pie_min_frac=pie_min_frac, pie_top_k=pie_top_k,
        label_text=[f"s{int(s)}" for s in v] if label_vertices else None,
        label_size=9, label_halo=True
    )

    # legends (small, top-right)
    if legend:
        em_handles, em_labels = _emission_legend_handles(
            obs_labels, obs_colors, markersize=legend_marker_size
        )
        fig.legend(
            em_handles, em_labels, title=legend_title, frameon=False,
            loc="upper right", bbox_to_anchor=(0.985, 0.985),
            ncol=legend_ncol,
            fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
            handlelength=legend_handlelength, handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing, borderaxespad=0.0,
        )
        if action_labels is not None and action_colors is not None:
            act_handles = [
                plt.Line2D([0], [0], color=action_colors[l], lw=action_legend_linewidth)
                for l in action_labels
            ]
            act_fs = legend_fontsize if action_legend_fontsize is None else action_legend_fontsize
            fig.legend(
                act_handles, action_labels, title="Actions", frameon=False,
                loc="upper right", bbox_to_anchor=(0.985, 0.76),
                ncol=1,
                fontsize=act_fs, title_fontsize=legend_title_fontsize,
                handlelength=max(1.0, legend_handlelength),
                handletextpad=legend_handletextpad, columnspacing=legend_columnspacing,
                borderaxespad=0.0,
            )

    # padding so nothing clips
    xmin, ymin = xy.min(0); xmax, ymax = xy.max(0)
    rngx = xmax - xmin if xmax > xmin else 1.0
    rngy = ymax - ymin if ymax > ymin else 1.0
    pad_x = pad * rngx + 1.6 * radius
    pad_y = pad * rngy + 1.6 * radius
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    if title: ax.set_title(title)
    if created_fig and not legend:
        fig.tight_layout()
    return fig, ax, A_adj, v, xy

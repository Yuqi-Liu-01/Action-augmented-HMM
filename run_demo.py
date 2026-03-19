import json
import os
import pickle
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ahmm_mplconfig"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ahmm_eval import (
    compute_nll_any,
    get_pde,
    nll_null_model,
    resolve_data_path,
)
from ahmm_plotting import (
    DEFAULT_ACTION_COLORS,
    make_obs_colors,
    plot_decoded_graph_ahmm_with_pies,
    plot_emission_matrix,
    plot_policy_matrix,
    plot_transition_matrices,
    show_pv_representation_ahmm_dataset,
)
from ahmm_utils import (
    SingleTrackConfig,
    SingleTrackGenerator,
    build_vocab,
    train_test_split_by_sequence,
    train_ahmm,
)


def _load_demo_session(root: Path, animal: str = "G375", session_date: str = "2023-02-28") -> dict:
    try:
        sessions_path = resolve_data_path("sessions_combined.pkl", data_roots=("demo_data", "data", root))
    except FileNotFoundError:
        sessions_path = root / "sessions_combined.pkl"
    with sessions_path.open("rb") as f:
        sessions = pickle.load(f)

    for session in sessions:
        exp_date = session.get("exp_date")
        exp_date_str = exp_date.strftime("%Y-%m-%d") if hasattr(exp_date, "strftime") else str(exp_date)
        if session.get("animal") == animal and exp_date_str == session_date:
            return session

    raise ValueError(f"Could not find session for animal={animal!r} on date={session_date!r}.")


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "demo_outputs"
    out_dir.mkdir(exist_ok=True)

    session = _load_demo_session(root, animal="G375", session_date="2023-02-28")
    records = list(session["records"])
    heights = sorted({int(h1) for h1, _, _ in records} | {int(h2) for _, h2, _ in records})

    vocab = build_vocab(heights, height_encoding="split")
    gen = SingleTrackGenerator(
        vocab,
        SingleTrackConfig(
            tower_heights=heights,
            height_encoding="split",
            p_gap=1.0,
            p_reward_given_correct=1.0,
            p_reward_given_incorrect=0.0,
        ),
    )

    ds_session = gen.sample_dataset_from_real(records, seed=0)
    ds_train, ds_holdout = train_test_split_by_sequence(ds_session, test_ratio=0.20, seed=0)
    ds_validate, ds_test = train_test_split_by_sequence(ds_holdout, test_ratio=0.5, seed=0)

    train_kwargs = {
        "tol": 1e-6,
        "early_stop": True,
        "patience": 5,
        "min_delta": 1e-6,
        "check_every": 5,
    }
    model, trace = train_ahmm(
        ds_train,
        n_states=25,
        n_iters=100000,
        tol=train_kwargs["tol"],
        seed=0,
        early_stop=train_kwargs["early_stop"],
        patience=train_kwargs["patience"],
        min_delta=train_kwargs["min_delta"],
        check_every=train_kwargs["check_every"],
    )

    nll_null = float(nll_null_model(ds_validate.obs, ds_validate.act))
    nll_model = float(compute_nll_any(model, dataset=ds_validate, mean=False))
    pde = float(get_pde(nll_model, nll_null))

    action_labels = [vocab.id_to_action[i] for i in range(len(vocab.id_to_action))]
    obs_labels = [k for k, _ in sorted(vocab.obs_to_id.items(), key=lambda kv: kv[1])]
    obs_colors = make_obs_colors(obs_labels)

    axs = plot_transition_matrices(
        model=model,
        action_labels=action_labels,
        title="G375 2023-02-28 Transition Matrices",
    )
    axs[0].figure.subplots_adjust(left=0.07, right=0.92, top=0.88, bottom=0.18, wspace=0.35)
    axs[0].figure.savefig(out_dir / "demo_transition_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(axs[0].figure)

    fig, ax = plot_policy_matrix(
        model=model,
        action_labels=action_labels,
        title="G375 2023-02-28 Policy Matrix",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "demo_policy_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_emission_matrix(model, vocab, title="G375 2023-02-28 Emission Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "demo_emission_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    show_pv_representation_ahmm_dataset(
        t=1,
        s=1,
        dataset=ds_session,
        model=model,
        tower_heights=heights,
        id_to_obs=vocab.id_to_obs,
        ax=ax,
        plot=True,
        metric="corr",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "demo_pv_representation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax, _, _, _ = plot_decoded_graph_ahmm_with_pies(
        model,
        ds_session.obs,
        ds_session.act,
        obs_labels=obs_labels,
        obs_colors=obs_colors,
        mode="model",
        action_weights="empirical",
        action_labels=action_labels,
        action_colors=DEFAULT_ACTION_COLORS,
        title="G375 2023-02-28 Transition Graph With Pies",
        legend=True,
    )
    fig.savefig(out_dir / "demo_transition_graph_pies.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "demo_animal": session["animal"],
        "demo_date": "2023-02-28",
        "split_method": "random trial split",
        "height_encoding": "split",
        "train_seed": 0,
        "test_ratio": 0.20,
        "valid_ratio": 0.5,
        "n_trials_total": int(len(ds_session.lengths)),
        "n_trials_train": int(len(ds_train.lengths)),
        "n_trials_validate": int(len(ds_validate.lengths)),
        "n_trials_test": int(len(ds_test.lengths)),
        "n_states": int(model.pi.shape[0]),
        "max_iters": 100000,
        "pde_validate": pde,
        "nll_model_validate": nll_model,
        "nll_null_validate": nll_null,
        "trace_length": int(len(trace)),
        "final_log_likelihood": float(trace[-1]),
        "demo_outputs": [
            "demo_summary.json",
            "demo_transition_matrices.png",
            "demo_policy_matrix.png",
            "demo_emission_matrix.png",
            "demo_pv_representation_matrix.png",
            "demo_transition_graph_pies.png",
        ],
    }

    with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

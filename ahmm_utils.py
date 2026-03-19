
"""
Utilities for AHMM data preparation, synthetic/real trial generation, and training.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import re
from enum import Enum
# -----------------------------
# Vocabulary / Encoding helpers
# -----------------------------
class HeightEncoding(Enum):
    SHARED = "shared"   # 'h_7' used for both objects
    SPLIT  = "split"    # 'l_7' for obj1, 'r_7' for obj2

# regex for parsing
_height_re_shared = re.compile(r"^h_(\d+)$")
_height_re_splitL = re.compile(r"^l_(\d+)$")
_height_re_splitR = re.compile(r"^r_(\d+)$")

def height_token(height: int, pos: int, scheme: HeightEncoding) -> str:
    """
    Return observation token for a height at object position pos∈{1,2}.
    SHARED: 'h_7'
    SPLIT : pos=1 -> 'l_7'; pos=2 -> 'r_7'
    """
    h = int(height)
    if scheme == HeightEncoding.SHARED:
        return f"h_{h}"
    elif scheme == HeightEncoding.SPLIT:
        if pos == 1: return f"l_{h}"
        if pos == 2: return f"r_{h}"
        raise ValueError(f"pos must be 1 or 2, got {pos}")
    else:
        raise ValueError(f"Unknown HeightEncoding: {scheme}")
    
def parse_height_token(tok: str) -> tuple[int | None, int | None]:
    """
    If tok encodes a height, return (pos, height).
    - SHARED tokens: 'h_7'  -> (None, 7)
    - SPLIT  tokens: 'l_7'  -> (1, 7); 'r_7' -> (2, 7)
    Else -> (None, None)
    """
    m = _height_re_shared.match(tok)
    if m: return (None, int(m.group(1)))
    m = _height_re_splitL.match(tok)
    if m: return (1, int(m.group(1)))
    m = _height_re_splitR.match(tok)
    if m: return (2, int(m.group(1)))
    return (None, None)


def is_height_token(tok: str) -> bool:
    pos, h = parse_height_token(tok)
    return h is not None

def hpos(tok: str) -> int | None:
    """Return 1/2 if token is positional height (split), else None."""
    pos, _ = parse_height_token(tok)
    return pos


@dataclass
class SessionEntry:
    session_id: str
    dataset: "SequenceDataset"
    meta: Dict[str, Any]

@dataclass(frozen=True)
class Vocab:
    obs_to_id: Dict[str, int]
    id_to_obs: Dict[int, str]
    action_to_id: Dict[str, int]
    id_to_action: Dict[int, str]

def build_vocab(tower_heights: Iterable[int],
                extra_obs: Optional[List[str]] = None,
                actions: Optional[List[str]] = None,
                *,
                height_encoding: Union[HeightEncoding, str] = HeightEncoding.SHARED) -> Vocab:
    if isinstance(height_encoding, str):
        height_encoding = HeightEncoding(height_encoding)

    if extra_obs is None:
        extra_obs = ['start', 'gap', 'reward', 'no_reward', 'end']
    if actions is None:
        actions = ['decision_R', 'decision_L', 'observation']

    heights = list(map(int, tower_heights))
    if height_encoding == HeightEncoding.SHARED:
        height_tokens = [f"h_{h}" for h in heights]
    elif height_encoding == HeightEncoding.SPLIT:
        height_tokens = [f"l_{h}" for h in heights] + [f"r_{h}" for h in heights]
    else:
        raise ValueError(f"Unknown height_encoding: {height_encoding}")
    
    obs_tokens = ['start'] + height_tokens + ['gap', 'reward', 'no_reward', 'end']
    for tok in (extra_obs or []):
        if tok not in obs_tokens:
            obs_tokens.append(tok)

    obs_to_id = {tok: i for i, tok in enumerate(obs_tokens)}
    id_to_obs = {i: tok for tok, i in obs_to_id.items()}
    action_to_id = {a: i for i, a in enumerate(actions)}
    id_to_action = {i: a for a, i in action_to_id.items()}
    return Vocab(obs_to_id, id_to_obs, action_to_id, id_to_action)

# -----------------------------
# Data containers
# -----------------------------

@dataclass
class SequenceDataset:
    obs: np.ndarray               # (T_total,)
    act: np.ndarray               # (T_total,)
    reward: Optional[np.ndarray]  # (T_total,) or None
    lengths: np.ndarray           # (n_seq,)
    stage_ids: Optional[np.ndarray] = None  # per-seq labels for curriculum stage

    @property
    def n_obs(self) -> int:
        return int(self.obs.max()) + 1 if self.obs.size else 0

    @property
    def n_actions(self) -> int:
        return int(self.act.max()) + 1 if self.act.size else 0

# -----------------------------
# Real-data preparation helpers
# -----------------------------

def concat_sequences(obs_seqs: List[np.ndarray],
                     act_seqs: List[np.ndarray],
                     reward_seqs: Optional[List[np.ndarray]] = None,
                     stage_ids: Optional[List[int]] = None) -> SequenceDataset:
    assert len(obs_seqs) == len(act_seqs)
    lengths = np.array([len(x) for x in obs_seqs], dtype=int)
    obs = np.concatenate(obs_seqs).astype(int)
    act = np.concatenate(act_seqs).astype(int)
    reward = None if reward_seqs is None else np.concatenate(reward_seqs).astype(int)
    stages = None if stage_ids is None else np.array(stage_ids, dtype=int)
    return SequenceDataset(obs, act, reward, lengths, stages)


def _normalize_side(x: str) -> str:
    side = str(x).strip().lower()
    if side in ("l", "left"):
        return "left"
    if side in ("r", "right"):
        return "right"
    return side


def _correct_side_from_heights(h1: int, h2: int) -> str:
    return "left" if int(h1) < int(h2) else "right"


def _to_lr_label(side_norm: str) -> str:
    if side_norm in ("left", "1"):
        return "L"
    if side_norm in ("right", "0"):
        return "R"
    return side_norm


def build_sessions_from_animal_df(df: pd.DataFrame) -> Tuple[List[dict], pd.DataFrame]:
    rows = []
    sessions = []

    for (animal, date), group in df.groupby(["animal", "exp_date"], dropna=False):
        valid = group.dropna(subset=["object_1_h", "object_2_h", "lick_side"])
        if valid.empty:
            continue

        h1s = valid["object_1_h"].astype(int).to_numpy()
        h2s = valid["object_2_h"].astype(int).to_numpy()
        licks_norm = valid["lick_side"].map(_normalize_side).to_numpy()

        if "correct_side" in valid.columns and not valid["correct_side"].isna().all():
            gold = valid["correct_side"].map(_normalize_side).to_numpy()
        else:
            gold = np.array([_correct_side_from_heights(h1, h2) for h1, h2 in zip(h1s, h2s)])

        n_trials = len(h1s)
        accuracy = float((licks_norm == gold).mean()) if n_trials else np.nan
        records = [(int(h1), int(h2), _to_lr_label(side)) for h1, h2, side in zip(h1s, h2s, licks_norm)]
        exp_date = pd.to_datetime(date)

        sessions.append(
            {
                "animal": animal,
                "exp_date": exp_date,
                "records": records,
                "accuracy": accuracy,
                "n_trials": n_trials,
            }
        )
        rows.append(
            {
                "animal": animal,
                "exp_date": exp_date,
                "n_trials": n_trials,
                "accuracy": accuracy,
                "h1_min": int(np.min(h1s)),
                "h1_max": int(np.max(h1s)),
                "h2_min": int(np.min(h2s)),
                "h2_max": int(np.max(h2s)),
            }
        )

    sessions = sorted(sessions, key=lambda session: session["accuracy"])
    summary_df = pd.DataFrame(rows).sort_values(["animal", "exp_date"]).reset_index(drop=True)
    return sessions, summary_df


def build_all_sessions(dfs: Dict[str, pd.DataFrame]) -> Tuple[List[dict], pd.DataFrame]:
    all_sessions = []
    all_rows = []
    for animal, df in dfs.items():
        sessions, summary = build_sessions_from_animal_df(df)
        all_sessions.extend(sessions)
        all_rows.append(summary)

    summary_df = (
        pd.concat(all_rows, ignore_index=True)
        .sort_values(["animal", "exp_date"])
        .reset_index(drop=True)
    )
    all_sessions = sorted(all_sessions, key=lambda session: (session["animal"], session["exp_date"]))
    return all_sessions, summary_df


def records_to_dataset(records: List[Tuple[int, int, str]], gen: "SingleTrackGenerator") -> SequenceDataset:
    return gen.sample_dataset_from_real(records)


def _hash_ndarray(x: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha1()
    digest.update(np.asarray(x).tobytes())
    return digest.hexdigest()[:10]


def save_ahmm(path: str, model: "AHMM", vocab: Vocab, meta: Dict[str, Any]) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        pi=model.pi.astype(np.float64),
        trans=model.trans.astype(np.float64),
        emit=model.emit.astype(np.float64),
        policy=model.policy.astype(np.float64),
        vocab_obs_order=np.array(
            [k for k, _ in sorted(vocab.obs_to_id.items(), key=lambda kv: kv[1])],
            dtype=object,
        ),
        vocab_act_order=np.array(
            [k for k, _ in sorted(vocab.action_to_id.items(), key=lambda kv: kv[1])],
            dtype=object,
        ),
    )
    header = dict(
        saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        pi_hash=_hash_ndarray(model.pi),
        trans_shape=tuple(model.trans.shape),
        emit_shape=tuple(model.emit.shape),
        policy_shape=tuple(model.policy.shape),
        **meta,
    )
    save_path.with_suffix(".json").write_text(json.dumps(header, indent=2))


def load_ahmm(path: str) -> Tuple["AHMM", Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    model = AHMM(data["pi"], data["trans"], data["emit"], data["policy"])
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def train_test_split_by_sequence(dataset: SequenceDataset,
                                 test_ratio: float = 0.2,
                                 seed: Optional[int] = None) -> Tuple[SequenceDataset, SequenceDataset]:
    rng = np.random.default_rng(seed)
    n = len(dataset.lengths)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    starts = np.cumsum(np.r_[0, dataset.lengths])
    def slice_idx(which):
        obsL, actL, rewL, lens, stages = [], [], [], [], []
        for i in sorted(which):
            s, e = starts[i], starts[i+1]
            obsL.append(dataset.obs[s:e]); actL.append(dataset.act[s:e])
            if dataset.reward is not None: rewL.append(dataset.reward[s:e])
            lens.append(e - s)
            if dataset.stage_ids is not None: stages.append(int(dataset.stage_ids[i]))
        reward = None if not rewL else np.concatenate(rewL)
        stage_ids = None if not stages else np.array(stages, dtype=int)
        return SequenceDataset(np.concatenate(obsL), np.concatenate(actL), reward, np.array(lens), stage_ids)
    return slice_idx(train_idx), slice_idx(test_idx)


def slice_dataset_by_indices(dataset: SequenceDataset, indices: Iterable[int]) -> SequenceDataset:
    starts = np.cumsum(np.r_[0, dataset.lengths])
    obs_list, act_list, reward_list, lengths, stages = [], [], [], [], []

    for idx in sorted(int(i) for i in indices):
        start, end = int(starts[idx]), int(starts[idx + 1])
        obs_list.append(dataset.obs[start:end])
        act_list.append(dataset.act[start:end])
        if dataset.reward is not None:
            reward_list.append(dataset.reward[start:end])
        lengths.append(end - start)
        if dataset.stage_ids is not None:
            stages.append(int(dataset.stage_ids[idx]))

    if not obs_list:
        reward = None if dataset.reward is None else np.array([], dtype=int)
        stage_ids = None if dataset.stage_ids is None else np.array([], dtype=int)
        return SequenceDataset(
            obs=np.array([], dtype=int),
            act=np.array([], dtype=int),
            reward=reward,
            lengths=np.array([], dtype=int),
            stage_ids=stage_ids,
        )

    reward = None if dataset.reward is None else np.concatenate(reward_list).astype(int)
    stage_ids = None if dataset.stage_ids is None else np.array(stages, dtype=int)
    return SequenceDataset(
        obs=np.concatenate(obs_list).astype(int),
        act=np.concatenate(act_list).astype(int),
        reward=reward,
        lengths=np.array(lengths, dtype=int),
        stage_ids=stage_ids,
    )


def train_test_split_random(
    dataset: SequenceDataset,
    test_ratio: float = 0.2,
    random_state: int | None = None,
) -> tuple[SequenceDataset, SequenceDataset]:
    rng = np.random.default_rng(random_state)
    n_trials = len(dataset.lengths)
    indices = np.arange(n_trials)
    rng.shuffle(indices)

    n_test = int(round(test_ratio * n_trials))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    return slice_dataset_by_indices(dataset, train_idx), slice_dataset_by_indices(dataset, test_idx)


def make_session_lookup(sessions_combined: Iterable[dict]) -> dict[tuple[str, str], dict]:
    return {
        (str(session["animal"]), pd.to_datetime(session["exp_date"]).strftime("%Y-%m-%d")): session
        for session in sessions_combined
    }


# -----------------------------
# Single-track generator with curriculum
# -----------------------------

@dataclass
class SingleTrackConfig:
    tower_heights: List[int]
    p_gap: float = 1.0
    p_reward_given_correct: float = 1.0
    p_reward_given_incorrect: float = 0.0
    start_token: str = 'start'
    gap_token: str = 'gap'
    end_token: str = 'end'
    height_encoding: Union[HeightEncoding, str] = HeightEncoding.SHARED

class SingleTrackGenerator:
    def __init__(self, vocab: Vocab, cfg: SingleTrackConfig):
        self.vocab, self.cfg = vocab, cfg
        if isinstance(self.cfg.height_encoding, str):
            self.cfg.height_encoding = HeightEncoding(self.cfg.height_encoding)

    @staticmethod
    def _correct_action(h1: int, h2: int) -> str:
        if h1 < h2: return 'decision_R'
        if h1 > h2: return 'decision_L'
        return 'observation'  # tie
    
    def _side_to_action(self, lick_side: str) -> str:
        s = str(lick_side).strip().lower()
        table = {
            'l': 'decision_L', 'left': 'decision_L',
            'r': 'decision_R', 'right': 'decision_R'
        }
        if s not in table:
            raise ValueError(f"Unrecognized lick_side: {lick_side!r}")
        return table[s]

    def make_trial_from_real(self, h1: int, h2: int, lick_side: str, *, rng: np.random.Generator | None = None):
        """
        Build a single trial from real data (h1, h2, animal's licked side).
        Returns (obs:int[], act:int[], reward:int[]).
        - Actions are all 'observation' except one decision token at obj2 time.
        - Reward is drawn per cfg.p_reward_given_{correct,incorrect}.
        """
        v, c = self.vocab, self.cfg
        if rng is None:
            rng = np.random.default_rng()  # reproducible if caller passes a seeded Generator

        # heights: shared ('h_{h}') or split ('l_{h}','r_{h}')
        tok_h1 = height_token(int(h1), pos=1, scheme=c.height_encoding)
        tok_h2 = height_token(int(h2), pos=2, scheme=c.height_encoding)

        # build observations up to obj2
        obs = [v.obs_to_id[c.start_token], v.obs_to_id[tok_h1]]
        if rng.random() < c.p_gap:
            obs.append(v.obs_to_id[c.gap_token])
        obs.append(v.obs_to_id[tok_h2])

        # index of the obj2 token we just appended
        decision_index = len(obs) - 1

        # animal's decision from lick_side
        decision_name = self._side_to_action(lick_side)
        a_id = v.action_to_id[decision_name]

        # correctness vs rule
        gold = self._correct_action(int(h1), int(h2))
        correct = (decision_name == gold)
        p_rew = c.p_reward_given_correct if correct else c.p_reward_given_incorrect
        rew_tok = v.obs_to_id['reward' if rng.random() < p_rew else 'no_reward']

        # finish obs with outcome + end
        obs.extend([rew_tok, v.obs_to_id[c.end_token]])

        # actions timeline: one decision at obj2 index
        acts = (
            [v.action_to_id['observation']]*decision_index
            + [a_id]
            + [v.action_to_id['observation']] * (len(obs) - decision_index - 1)
        )

        # rewards array: only the outcome token is 1 if reward
        rewards = np.zeros(len(obs), dtype=int)
        rewards[-2] = 1 if rew_tok == v.obs_to_id['reward'] else 0

        return np.array(obs, int), np.array(acts, int), rewards


    def sample_dataset_from_real(self, records, stage_id: int | None = None, *, seed: int | None = None):
        """
        records: iterable of (h1, h2, lick_side).
        Returns SequenceDataset; optionally tags all sequences with 'stage_id'.
        """
        rng = np.random.default_rng(seed)
        obsL, actL, rewL, lens, stages = [], [], [], [], []
        for (h1, h2, lick) in records:
            o, a, r = self.make_trial_from_real(h1, h2, lick, rng=rng)
            obsL.append(o); actL.append(a); rewL.append(r); lens.append(len(o))
            if stage_id is not None:
                stages.append(stage_id)
        stage_ids = None if stage_id is None else stages
        return concat_sequences(obsL, actL, rewL, stage_ids=stage_ids)

    


# ---------- AHMM (discrete EM) ----------
def _normalize(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True) + eps
    return x / s

def _safe_log(x, eps: float = 1e-300):
    x = np.asarray(x, dtype=float)
    return np.log(np.clip(x, eps, 1.0))


@dataclass
class AHMM:
    pi: np.ndarray            # (S,)
    trans: np.ndarray         # (A,S,S)
    emit: np.ndarray          # (S,O)
    policy: np.ndarray        # (S,A)

    def decode(self, obs: np.ndarray, acts: np.ndarray, *, return_logp: bool = True):
        """
        Viterbi path for a single (obs, acts) sequence.
        Uses log-domain scores with:
            score_t(s) = max_{s'} score_{t-1}(s') + log T[a_{t-1}][s', s]
                        + log policy[s, a_t] + log emit[s, o_t]
        t=0 base: log pi[s] + log policy[s, a_0] + log emit[s, o_0]
        Returns (logp, states) if return_logp else states.
        """
        obs = np.asarray(obs, dtype=int)
        acts = np.asarray(acts, dtype=int)
        Tlen = int(len(obs))
        if Tlen == 0:
            return (float("-inf"), np.array([], dtype=int)) if return_logp else np.array([], dtype=int)

        S = int(self.pi.shape[0])
        # log parameters
        logpi   = _safe_log(self.pi)                  # (S,)
        logemit = _safe_log(self.emit)                # (S,O)
        logpol  = _safe_log(self.policy)              # (S,A)
        logTr   = _safe_log(self.trans)               # (A,S,S)

        delta = np.full((Tlen, S), -np.inf, dtype=float)
        psi   = np.full((Tlen, S), -1, dtype=int)

        # t = 0
        a0, o0 = int(acts[0]), int(obs[0])
        delta[0, :] = logpi + logpol[:, a0] + logemit[:, o0]

        # t >= 1
        for t in range(1, Tlen):
            a_prev = int(acts[t-1])
            a_t    = int(acts[t])
            o_t    = int(obs[t])
            # scores from all prev states to each current state
            # shape (S_prev, S_curr)
            scores = delta[t-1][:, None] + logTr[a_prev]
            psi[t, :]   = np.argmax(scores, axis=0)
            delta[t, :] = np.max(scores, axis=0) + logpol[:, a_t] + logemit[:, o_t]

        # backtrack
        last_state = int(np.argmax(delta[-1]))
        path = np.empty(Tlen, dtype=int)
        path[-1] = last_state
        for t in range(Tlen-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        best_logp = float(delta[-1, last_state])
        return (best_logp, path) if return_logp else path


    def decode_sequences(dataset: SequenceDataset, model: AHMM, *, return_logp: bool = True):
        """
        Run Viterbi per trial in a SequenceDataset.
        Returns a list of paths (and optionally logps) aligned to dataset.lengths.
        """
        O = dataset.obs.astype(int)
        A = dataset.act.astype(int)
        Ls = dataset.lengths.astype(int)
        starts = np.cumsum(np.r_[0, Ls[:-1]])
        paths = []
        logps = [] if return_logp else None
        for s, L in zip(starts, Ls):
            o = O[s:s+L]; a = A[s:s+L]
            out = model.decode(o, a, return_logp=return_logp)
            if return_logp:
                lp, st = out
                logps.append(lp); paths.append(st)
            else:
                paths.append(out)
        return (paths, np.array(logps, float)) if return_logp else paths



def train_ahmm(
    dataset: SequenceDataset, n_states: int,
    n_obs: Optional[int] = None, n_actions: Optional[int] = None,
    n_iters: int = 200, tol: float = 1e-4, seed: int = 0,
    init_model: Optional[AHMM] = None,
    *,
    early_stop: bool = True,
    patience: int = 5,
    min_delta: float = 1e-4,
    check_every: int = 1,
) -> Tuple[AHMM, List[float]]:
    """
    EM for AHMM; supports warm start via init_model and early stopping.
    Stops when:
      - classic |LL_t - LL_{t-1}| < tol (legacy), OR
      - early_stop=True and 'patience' checks pass without >= min_delta improvement
    """
    rng = np.random.default_rng(seed)
    O, A, Ls = dataset.obs.astype(int), dataset.act.astype(int), dataset.lengths.astype(int)
    O_card = int(n_obs if n_obs is not None else (O.max() + 1 if O.size else 0))
    A_card = int(n_actions if n_actions is not None else (A.max() + 1 if A.size else 0))
    S = int(n_states)

    # --- init (warm start if provided) ---
    if init_model is not None:
        pi     = init_model.pi.copy()
        trans  = init_model.trans.copy()
        emit   = init_model.emit.copy()
        policy = init_model.policy.copy()
        if emit.shape != (S, O_card):   emit   = _normalize(rng.random((S, O_card)), axis=1)
        if policy.shape != (S, A_card): policy = _normalize(rng.random((S, A_card)), axis=1)
        if trans.shape != (A_card, S, S):
            trans = np.stack([_normalize(rng.random((S, S)), axis=1) for _ in range(A_card)], axis=0)
        if pi.shape != (S,):            
            pi = _normalize(rng.random(S))
    else:
        pi = _normalize(rng.random(S))
        trans = np.stack([_normalize(rng.random((S, S)), axis=1) for _ in range(A_card)], axis=0)
        emit  = _normalize(rng.random((S, O_card)), axis=1)
        policy= _normalize(rng.random((S, A_card)), axis=1)

    trace: List[float] = []
    no_improve = 0  # for patience

    def fb(seq_O, seq_A):
        T = len(seq_O)
        alpha = np.zeros((T, S))
        beta = np.zeros((T, S))
        a0 = seq_A[0]
        alpha[0] = pi * policy[:, a0] * emit[:, seq_O[0]]
        c0 = alpha[0].sum() + 1e-12
        alpha[0] /= c0; logc = np.log(c0)
        for t in range(1, T):
            M = trans[seq_A[t-1]]
            alpha[t] = (alpha[t-1] @ M) * policy[:, seq_A[t]] * emit[:, seq_O[t]]
            ct = alpha[t].sum() + 1e-12; alpha[t] /= ct; logc += np.log(ct)
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            M = trans[seq_A[t]]
            beta[t] = (M @ (policy[:, seq_A[t+1]] * emit[:, seq_O[t+1]] * beta[t+1]))
            bt = beta[t].sum() + 1e-12
            beta[t] /= bt
        g = alpha * beta
        g /= g.sum(axis=1, keepdims=True) + 1e-12
        xi_sum = np.zeros((A_card, S, S))
        for t in range(T-1):
            a = seq_A[t]; M = trans[a]
            tmp = (alpha[t][:, None] * M) * (policy[:, seq_A[t+1]][None, :] * emit[:, seq_O[t+1]][None, :] * beta[t+1][None, :])
            tmp /= tmp.sum() + 1e-12
            xi_sum[a] += tmp
        return g, xi_sum, logc

    for it in range(1, n_iters + 1):
        gamma0 = np.zeros(S)
        emitC  = np.zeros((S, O_card))
        polC   = np.zeros((S, A_card))
        trC    = np.zeros((A_card, S, S))
        total  = 0.0

        s = 0
        for L in Ls:
            e = s + L; seq_O, seq_A = O[s:e], A[s:e]
            g, xi, ll = fb(seq_O, seq_A)
            total += ll
            gamma0 += g[0]
            for t in range(L):
                emitC[:, seq_O[t]] += g[t]
                polC[:, seq_A[t]] += g[t]
            trC += xi
            s = e

        # M-step
        pi = _normalize(gamma0)
        emit = _normalize(emitC, axis=1)
        policy = _normalize(polC, axis=1)
        for a in range(A_card):
            trC[a] = _normalize(trC[a], axis=1)
        trans = trC

        # bookkeeping
        trace.append(total)

        # legacy tol stopping
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
            break

        # patience-based early stopping
        if early_stop and (it % check_every == 0):
            if len(trace) == 1:
                best = trace[-1]; no_improve = 0
            else:
                best = max(trace[:-1])  # best before current
                if (trace[-1] - best) >= min_delta:
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

    return AHMM(pi, trans, emit, policy), trace


def sweep_save_all_and_log(
    sessions_all: List[dict],
    vocab: Vocab,
    gen: "SingleTrackGenerator",
    *,
    animals: Optional[Tuple[str, ...]] = ("G375", "G386", "G402", "G410", "G420", "G506"),
    models_root: str = "ahmm_models",
    seeds=range(0, 101),
    test_ratio: float = 0.20,
    valid_ratio: float = 0.5,
    n_states: int = 25,
    max_iters: int = 100000,
    train_kwargs: Optional[Dict[str, Any]] = None,
    log_path: Optional[str] = None,
    checkpoint_every: int = 1,
    overwrite: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    from tqdm import tqdm
    from ahmm_eval import compute_nll_any, get_pde, nll_null_model

    if train_kwargs is None:
        train_kwargs = dict(early_stop=True, patience=5, min_delta=1e-6, check_every=5, tol=1e-6)

    if resume and log_path is not None and os.path.exists(log_path):
        ext = os.path.splitext(log_path)[1].lower()
        if ext == ".parquet":
            log_df = pd.read_parquet(log_path)
        elif ext in (".pkl", ".pickle"):
            log_df = pd.read_pickle(log_path)
        elif ext == ".csv":
            log_df = pd.read_csv(log_path)
        else:
            log_df = pd.read_pickle(log_path)
        rows = log_df.to_dict("records")
    else:
        rows = []

    def checkpoint(df: pd.DataFrame) -> None:
        if log_path is None:
            return
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        ext = os.path.splitext(log_path)[1].lower()
        if ext == ".parquet":
            df.to_parquet(log_path, index=False)
        elif ext in (".pkl", ".pickle"):
            df.to_pickle(log_path)
        elif ext == ".csv":
            df.to_csv(log_path, index=False)
        else:
            df.to_pickle(log_path)

    rows_since_checkpoint = 0
    animal_filter = set(animals) if animals is not None else None

    for session in sessions_all:
        animal = session["animal"]
        if animal_filter is not None and animal not in animal_filter:
            continue

        exp_date = str(pd.to_datetime(session["exp_date"]).date())
        ds_session = records_to_dataset(session["records"], gen)
        ds_train, ds_holdout = train_test_split_by_sequence(ds_session, test_ratio=test_ratio, seed=0)
        ds_validate, ds_test = train_test_split_by_sequence(ds_holdout, test_ratio=valid_ratio, seed=0)
        nll_null = float(nll_null_model(ds_validate.obs, ds_validate.act))

        for seed in tqdm(seeds, desc=f"Training seeds for {animal} {exp_date}"):
            save_path = os.path.join(models_root, animal, exp_date, f"{exp_date}_seed{int(seed):03d}_ahmm.npz")
            if not overwrite and os.path.exists(save_path):
                continue

            model, trace = train_ahmm(
                ds_train,
                n_states=n_states,
                n_iters=max_iters,
                tol=train_kwargs.get("tol", 1e-6),
                seed=seed,
                early_stop=train_kwargs.get("early_stop", True),
                patience=train_kwargs.get("patience", 5),
                min_delta=train_kwargs.get("min_delta", 1e-6),
                check_every=train_kwargs.get("check_every", 5),
            )

            nll_model = float(compute_nll_any(model, dataset=ds_validate, mean=False))
            pde = float(get_pde(nll_model, nll_null))
            meta = dict(
                animal=animal,
                exp_date=exp_date,
                seed=int(seed),
                n_states=int(model.emit.shape[0]),
                test_ratio=float(test_ratio),
                valid_ratio=float(valid_ratio),
                pde=pde,
                nll_model=nll_model,
                nll_null=nll_null,
                trace_first=float(trace[0]) if trace else np.nan,
                trace_last=float(trace[-1]) if trace else np.nan,
                animal_accuracy=session.get("accuracy"),
                n_trials_train=int(len(ds_train.lengths)),
                n_trials_validate=int(len(ds_validate.lengths)),
                n_trials_test=int(len(ds_test.lengths)),
            )
            save_ahmm(save_path, model, vocab, meta)

            rows.append(
                dict(
                    train_animal=animal,
                    train_date=exp_date,
                    seed=int(seed),
                    test_ratio=float(test_ratio),
                    valid_ratio=float(valid_ratio),
                    pde=pde,
                    nll_model=nll_model,
                    nll_null=nll_null,
                    save_path=save_path,
                    n_states=int(model.emit.shape[0]),
                    n_trials_train=int(len(ds_train.lengths)),
                    n_trials_validate=int(len(ds_validate.lengths)),
                    n_trials_test=int(len(ds_test.lengths)),
                )
            )

            rows_since_checkpoint += 1
            if checkpoint_every and rows_since_checkpoint % checkpoint_every == 0:
                checkpoint(pd.DataFrame(rows))
                rows_since_checkpoint = 0

    log_df = pd.DataFrame(rows)
    checkpoint(log_df)
    return log_df


def _lik(model, a_t, o_t):
    return model.policy[:, a_t] * model.emit[:, o_t]


def decode_posteriors_filtered(dataset, model):
    """
    Returns per-trial list of FILTERED posteriors (online γ_t),
    i.e., γ_t = P(z_t | o_{1:t}, a_{1:t}).
    """
    O, A, Ls = dataset.obs.astype(int), dataset.act.astype(int), dataset.lengths.astype(int)
    S = model.pi.shape[0]
    out = []
    s0 = 0
    for L in Ls:
        e = s0 + L
        seq_O, seq_A = O[s0:e], A[s0:e]
        T = len(seq_O)

        alpha = np.zeros((T, S))
        c = np.zeros(T)  # scaling factors

        # t = 0
        Et = _lik(model, seq_A[0], seq_O[0])
        alpha[0] = model.pi * Et
        c[0] = alpha[0].sum() + 1e-12
        alpha[0] /= c[0]

        # t >= 1
        for t in range(1, T-1):
            M = model.trans[seq_A[t-1]]            # T(z_t | z_{t-1}, a_{t-1})
            Et = _lik(model, seq_A[t], seq_O[t])   # P(a_t|z_t) * P(o_t|z_t)   
            alpha[t] = (alpha[t-1] @ M) * Et
            c[t] = alpha[t].sum() + 1e-12
            alpha[t] /= c[t]
        alpha[t+1] = (alpha[t] @ model.trans[seq_A[t]]) * model.emit[:, seq_O[t+1]]
        c[t+1] = alpha[t+1].sum() + 1e-12
        alpha[t+1] /= c[t+1]

        # filtered γ is just normalized alpha
        out.append(alpha)
        s0 = e
    return out


def h2_index(seq_O: np.ndarray, vocab) -> int | None:
    names = [vocab.id_to_obs[int(x)] for x in seq_O]
    height_idxs = [i for i, tok in enumerate(names) if is_height_token(tok)]
    if not height_idxs:
        return None
    h2_candidates = [i for i in height_idxs if hpos(names[i]) == 2]
    if h2_candidates:
        return h2_candidates[0]
    return height_idxs[1] if len(height_idxs) >= 2 else None

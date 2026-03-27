"""
regime_hmm.py
=============
Volatility-regime detector based on a 4-state Gaussian HMM.

States
------
  CALM          Low VIX, low VVIX — benign, trending environment.
  STRESS        Elevated VIX, rising VVIX — risk is building but not acute.
  CRISIS        High VIX *and* high VVIX — acute dislocation / spike.
  RECOVERY      Elevated VIX but VVIX falling — vol mean-reverting post-shock.

Design goals for live strategy use
------------------------------------
* No look-ahead: annotate() uses the *filtered* (forward-only) posterior,
  matching what is actually available at each bar.  The smoothed posterior
  (forward+backward) is available separately for research / attribution.
* Stable row alignment: feature construction tracks dropped warmup rows so
  regime labels are always date-aligned with the input DataFrame.
* Deterministic labelling: states are mapped to Regime enum by their
  economic interpretation (VIX level + VVIX level), not arbitrary HMM
  state indices, so the mapping is stable across re-fits.
* Regime smoothing: a configurable holding period prevents the model from
  whipsawing between regimes on noise — important for position sizing.
* Convergence guard: non-converged models are warned on; the best
  converged model is preferred over the best overall model.
* Serialisation: save / load via joblib so a fitted model survives sessions.
"""



import logging
import warnings
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .util import suppress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "vix_close",
    "vvix_close",
    "vix3m_close",
    "spy_close",
    "tip_close",
    "ief_close",
})
RV_WINDOW = 20           # rolling window for realised vol — sets the warmup period
CONTEXT_WINDOW = 126     # longer-horizon context for z-score style features
ANNUALISE = np.sqrt(252)


class Regime(IntEnum):
    CALM     = 0   # low VIX, low VVIX  — benign / trending
    STRESS   = 1   # elevated VIX, rising VVIX — risk building
    CRISIS   = 2   # high VIX *and* high VVIX — acute dislocation
    RECOVERY = 3   # elevated VIX, VVIX falling — post-shock mean reversion


K = len(Regime)
NORM_ENTROPY = np.log(K)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pl.DataFrame, context: str = "") -> None:
    tag = f"[{context}] " if context else ""

    if df.is_empty():
        raise ValueError(f"{tag}DataFrame is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{tag}Missing required columns: {sorted(missing)}")

    min_rows = max(RV_WINDOW, CONTEXT_WINDOW) + K + 1
    if len(df) < min_rows:
        raise ValueError(
            f"{tag}Need at least {min_rows} rows for feature construction; "
            f"got {len(df)}."
        )

    for col in REQUIRED_COLUMNS - {"spy_close"}:
        n_nonpos = (df[col] <= 0).sum()
        if n_nonpos:
            raise ValueError(
                f"{tag}Column '{col}' has {n_nonpos} non-positive value(s). "
                "Log transform requires strictly positive inputs."
            )
        null_pct = df[col].null_count() / len(df)
        if null_pct > 0.05:
            warnings.warn(
                f"{tag}Column '{col}' is {null_pct:.1%} null. "
                "Regime estimates may be unreliable.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(df: pl.DataFrame) -> tuple[np.ndarray, int]:
    """
    Build the feature matrix for regime inference.

    Returns
    -------
    X : np.ndarray, shape (n_valid, n_features)
        Rows with any remaining null/NaN have been dropped from the front.
    n_dropped : int
        Number of leading rows consumed by warmup / null removal.
        Callers use this to re-align results with the original DataFrame index.

    Features (column order)
    -----------------------
    0  vix_log          log(VIX)                 — level of implied vol
    1  term_slope       (VIX3M - VIX) / VIX      — term structure shape
    2  vvix_log         log(VVIX)                — vol-of-vol
    3  iv_rv_ratio      log(VIX) - log(RV_20)    — IV premium over realised vol
    4  vix_chg5         VIX 5-day % change       — momentum
    5  vvix_chg5        VVIX 5-day % change      — vol-of-vol momentum
    6  vvix_vix_spread  log(VVIX) - log(VIX)     — vol-of-vol premium
    7  term_slope_chg5  5-day change in slope    — curve acceleration
    8  vix_z126         z-score of log(VIX)      — long-horizon stress context
    9  rv_log_chg5      5-day change in RV_20    — realised vol trend
    10 infl_be_proxy    log(TIP) - log(IEF)      — breakeven inflation proxy
    11 infl_be_chg21    21-day change in proxy   — inflation pressure momentum
    """
    # Fill short provider gaps first; otherwise a single missing print can
    # invalidate long rolling windows (e.g., z-score over 126 bars).
    base = (
        df.with_columns(
            pl.col("vix_close").fill_nan(None),
            pl.col("vvix_close").fill_nan(None),
            pl.col("vix3m_close").fill_nan(None),
            pl.col("spy_close").fill_nan(None),
            pl.col("tip_close").fill_nan(None),
            pl.col("ief_close").fill_nan(None),
        )
        .with_columns(
            pl.col("vix_close").forward_fill(limit=3).backward_fill(limit=1),
            pl.col("vvix_close").forward_fill(limit=3).backward_fill(limit=1),
            pl.col("vix3m_close").forward_fill(limit=3).backward_fill(limit=1),
            pl.col("spy_close").forward_fill(limit=3).backward_fill(limit=1),
            pl.col("tip_close").forward_fill(limit=3).backward_fill(limit=1),
            pl.col("ief_close").forward_fill(limit=3).backward_fill(limit=1),
        )
    )

    feat = (
        base.with_columns(
            vix_log=pl.col("vix_close").log(),
            vvix_log=pl.col("vvix_close").log(),
            tip_log=pl.col("tip_close").log(),
            ief_log=pl.col("ief_close").log(),
            term_slope=(pl.col("vix3m_close") - pl.col("vix_close")) / pl.col("vix_close"),
            rv_log=(pl.col("spy_close").rolling_std(RV_WINDOW) * ANNUALISE).log(),
        )
        .with_columns(
            iv_rv_ratio=pl.col("vix_log") - pl.col("rv_log"),
            vix_chg5=pl.col("vix_close").pct_change(5),
            vvix_chg5=pl.col("vvix_close").pct_change(5),
            vvix_vix_spread=pl.col("vvix_log") - pl.col("vix_log"),
            term_slope_chg5=pl.col("term_slope").diff(5),
            vix_z126=(
                (pl.col("vix_log") - pl.col("vix_log").rolling_mean(CONTEXT_WINDOW))
                / (pl.col("vix_log").rolling_std(CONTEXT_WINDOW) + 1e-6)
            ),
            rv_log_chg5=pl.col("rv_log").diff(5),
            infl_be_proxy=pl.col("tip_log") - pl.col("ief_log"),
            infl_be_chg21=(pl.col("tip_log") - pl.col("ief_log")).diff(21),
        )
        .select(
            [
                "vix_log",
                "term_slope",
                "vvix_log",
                "iv_rv_ratio",
                "vix_chg5",
                "vvix_chg5",
                "vvix_vix_spread",
                "term_slope_chg5",
                "vix_z126",
                "rv_log_chg5",
                "infl_be_proxy",
                "infl_be_chg21",
            ]
        )
        .fill_nan(None)
        # Forward-fill gaps but cap at 3 bars to avoid stale propagation
        .with_columns(pl.all().forward_fill(limit=3))
        .drop_nulls()
    )

    n_dropped = len(df) - len(feat)
    return feat.to_numpy().astype(float), n_dropped


# ---------------------------------------------------------------------------
# State → Regime labelling
# ---------------------------------------------------------------------------

def _label_states(means_orig: np.ndarray) -> dict[int, Regime]:
    """
    Map raw HMM state indices to Regime labels using original-scale means.

    Strategy
    --------
    Features:  0=vix_log  1=term_slope  2=vvix_log  3=iv_rv_ratio
               4=vix_chg5  5=vvix_chg5  6=vvix_vix_spread
               7=term_slope_chg5  8=vix_z126  9=rv_log_chg5
               10=infl_be_proxy  11=infl_be_chg21

    Uses multi-signal scoring instead of a pure VIX sort so that:
    - CRISIS prefers high VIX/high VVIX with rising risk momentum.
    - RECOVERY prefers cooling VVIX momentum and improving curve shape.
    - CALM prefers low VIX + low VVIX profile.
    - STRESS is the residual elevated-risk state.
    """
    if means_orig.shape[0] != K:
        raise ValueError(
            f"Expected {K} HMM states but got {means_orig.shape[0]}. "
            "Re-fit with n_components=K."
        )

    idx = np.arange(K)
    vix = means_orig[:, 0]
    vvix = means_orig[:, 2]
    vix_chg = means_orig[:, 4]
    vvix_chg = means_orig[:, 5]
    spread = means_orig[:, 6]
    term_slope = means_orig[:, 1]
    vix_z = means_orig[:, 8]
    infl_be = means_orig[:, 10]
    infl_be_chg = means_orig[:, 11]

    calm_score = -(vix + 0.55 * vvix + 0.15 * np.maximum(vix_z, 0.0) + 0.10 * np.maximum(infl_be, 0.0))
    calm_idx = int(idx[np.argmax(calm_score)])

    crisis_score = (
        vix
        + 0.85 * vvix
        + 0.30 * np.maximum(vvix_chg, 0.0)
        + 0.20 * np.maximum(vix_chg, 0.0)
        + 0.20 * np.maximum(vix_z, 0.0)
        + 0.10 * np.maximum(infl_be_chg, 0.0)
    )
    crisis_order = np.argsort(crisis_score)[::-1]
    crisis_idx = int(next(i for i in crisis_order if i != calm_idx))

    remaining = [i for i in idx.tolist() if i not in {calm_idx, crisis_idx}]
    recovery_scores = {
        i: (-vvix_chg[i]) + 0.45 * term_slope[i] + 0.35 * spread[i] - 0.25 * max(vix_chg[i], 0.0) - 0.10 * max(infl_be_chg[i], 0.0)
        for i in remaining
    }
    recovery_idx = max(recovery_scores, key=lambda i: recovery_scores[i])
    stress_idx = next(i for i in remaining if i != recovery_idx)

    logger.debug(
        "State map diagnostics | CALM=%d CRISIS=%d RECOVERY=%d STRESS=%d",
        calm_idx,
        crisis_idx,
        recovery_idx,
        stress_idx,
    )

    return {
        calm_idx:     Regime.CALM,
        stress_idx:   Regime.STRESS,
        crisis_idx:   Regime.CRISIS,
        recovery_idx: Regime.RECOVERY,
    }


def _reorder_proba(raw_proba: np.ndarray, label_map: dict[int, Regime]) -> np.ndarray:
    """Reorder columns from HMM state order → Regime enum order."""
    out = np.empty_like(raw_proba)
    for raw_state, regime in label_map.items():
        out[:, regime.value] = raw_proba[:, raw_state]
    return out


def _build_transition_penalty_matrix(scale: float) -> np.ndarray:
    """Return a regime-ordered penalty matrix for unlikely one-step transitions."""
    penalty = np.zeros((K, K), dtype=float)

    # Discourage semantic skips while keeping all transitions technically possible.
    penalty[Regime.CALM.value, Regime.RECOVERY.value] = 0.95
    penalty[Regime.RECOVERY.value, Regime.CRISIS.value] = 0.55
    penalty[Regime.CALM.value, Regime.CRISIS.value] = 0.35
    penalty[Regime.STRESS.value, Regime.RECOVERY.value] = 0.25

    return penalty * max(scale, 0.0)


def _decode_with_soft_transition_penalty(proba: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """
    Decode labels with soft transition penalties.

    The decoder remains fully causal and only nudges the MAP choice away from
    semantically unlikely jumps when posterior differences are small.
    """
    if len(proba) == 0:
        return np.array([], dtype=np.int8)

    logp = np.log(np.clip(proba, 1e-12, 1.0))
    labels = np.empty(len(proba), dtype=np.int8)

    labels[0] = np.int8(np.argmax(logp[0]))
    for t in range(1, len(proba)):
        prev = int(labels[t - 1])
        labels[t] = np.int8(np.argmax(logp[t] - penalty[prev]))

    return labels


def _confidence_metrics(proba: np.ndarray, low_conf_threshold: float) -> dict[str, np.ndarray]:
    """Compute confidence diagnostics from regime-ordered posterior probabilities."""
    n = len(proba)
    max_prob = np.full(n, np.nan)
    margin = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    low_conf = np.zeros(n, dtype=bool)

    if n == 0:
        return {
            "max_prob": max_prob,
            "margin": margin,
            "entropy": entropy,
            "confidence": confidence,
            "is_low_confidence": low_conf,
        }

    valid = ~np.isnan(proba).all(axis=1)
    if valid.any():
        p = np.clip(proba[valid], 1e-12, 1.0)
        max_prob[valid] = np.max(p, axis=1)

        sorted_p = np.sort(p, axis=1)
        margin[valid] = sorted_p[:, -1] - sorted_p[:, -2]

        entropy[valid] = -np.sum(p * np.log(p), axis=1)
        confidence[valid] = 1.0 - (entropy[valid] / NORM_ENTROPY)
        low_conf[valid] = confidence[valid] < low_conf_threshold

    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "confidence": confidence,
        "is_low_confidence": low_conf,
    }


def _approx_diag_hmm_params(n_features: int) -> int:
    """Approximate number of free scalars for a K-state diagonal-covariance HMM."""
    means = K * n_features
    covars = K * n_features
    transition = K * (K - 1)
    start = K - 1
    return means + covars + transition + start


# ---------------------------------------------------------------------------
# Holding-period filter  (causal)
# ---------------------------------------------------------------------------

def _apply_holding_period(labels: np.ndarray, min_hold: int) -> np.ndarray:
    """
    Suppress regime transitions that last fewer than ``min_hold`` bars.

    Fully causal — only past bars are used to decide whether a transition is
    confirmed.  A new regime is accepted only after it has persisted for
    ``min_hold`` consecutive bars; until then the previous regime is reported.
    """
    if min_hold <= 1:
        return labels.copy()

    smoothed   = labels.copy()
    confirmed  = int(labels[0])
    candidate  = confirmed
    run_length = 1

    for i in range(1, len(labels)):
        if labels[i] == candidate:
            run_length += 1
            if run_length >= min_hold and candidate != confirmed:
                # Retrospectively confirm from the start of this run
                smoothed[i - run_length + 1 : i + 1] = candidate
                confirmed  = candidate
                run_length = 1
        else:
            # New candidate; suppress unconfirmed run so far
            start = i - run_length
            smoothed[start:i] = confirmed
            candidate  = int(labels[i])
            run_length = 1

    # Handle tail: unconfirmed run at end
    if candidate != confirmed:
        smoothed[len(labels) - run_length :] = confirmed

    return smoothed


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RegimeHMM:
    """
    Fits a 4-state Gaussian HMM on VIX-family features and maps the states
    to economically labelled volatility regimes.

    Parameters
    ----------
    n_init : int
        Number of random restarts; the best model by log-likelihood is kept.
    n_iter : int
        Maximum EM iterations per restart.
    random_state : int
        Base seed; restart i uses ``random_state + i``.
    min_holding_period : int
        Minimum bars a regime must persist before a transition is accepted.
        Set to 0 or 1 to disable.  Recommended: 3–5 for daily bars.
    transition_penalty : float
        Soft penalty strength for semantically unlikely one-step jumps.
        Set to 0 to disable transition penalties.
    low_confidence_threshold : float
        Threshold for diagnostics-only low confidence flag. Must be in [0, 1].
    """

    def __init__(
        self,
        n_init: int = 10,
        n_iter: int = 1_000,
        random_state: int = 42,
        min_holding_period: int = 5,
        transition_penalty: float = 0.20,
        low_confidence_threshold: float = 0.55,
    ) -> None:
        self.n_init             = n_init
        self.n_iter             = n_iter
        self.random_state       = random_state
        self.min_holding_period = min_holding_period
        self.transition_penalty = transition_penalty
        self.low_confidence_threshold = low_confidence_threshold

        if self.n_init < 1:
            raise ValueError("n_init must be >= 1")
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self.min_holding_period < 0:
            raise ValueError("min_holding_period must be >= 0")
        if self.transition_penalty < 0:
            raise ValueError("transition_penalty must be >= 0")
        if not (0.0 <= self.low_confidence_threshold <= 1.0):
            raise ValueError("low_confidence_threshold must be in [0, 1]")

        self._hmm:       Optional[GaussianHMM] = None
        self._scaler:    StandardScaler        = StandardScaler()
        self._label_map: dict[int, Regime]     = {}
        self._is_fitted: bool                  = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted; nothing to save.")
        joblib.dump(self, path)
        logger.info("RegimeHMM saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        """Load a previously saved model."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RegimeHMM, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before predicting.")

    def _preprocess(self, df: pl.DataFrame, *, fit: bool = False) -> tuple[np.ndarray, int]:
        X, n_dropped = build_features(df)
        if X.shape[0] == 0:
            raise ValueError(
                "Feature construction produced zero valid rows after warmup/null filtering."
            )
        X = self._scaler.fit_transform(X) if fit else self._scaler.transform(X)
        return X, n_dropped

    # ------------------------------------------------------------------
    # Filtered posterior  (causal — safe for live use)
    # ------------------------------------------------------------------

    def _filtered_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Forward-pass posterior P(s_t | x_{1:t}).

        This is what is genuinely available at time t — no future data.
        hmmlearn does not expose this directly, so we run the forward
        algorithm manually using the fitted model parameters.
        """
        assert self._hmm is not None
        log_B  = self._hmm._compute_log_likelihood(X_scaled)       # (T, K)
        log_A  = np.log(self._hmm.transmat_   + 1e-300)            # (K, K)
        log_pi = np.log(self._hmm.startprob_  + 1e-300)            # (K,)

        T          = len(X_scaled)
        log_alpha  = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            # (K_from, 1) + (K_from, K_to) → marginalise over K_from
            log_alpha[t] = (
                np.logaddexp.reduce(log_alpha[t - 1, :, None] + log_A, axis=0)
                + log_B[t]
            )

        # Normalise rows to probabilities
        log_alpha -= np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha)

    def _smoothed_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Smoothed posterior P(s_t | x_{1:T})  — uses all data.

        For research / attribution only.  Do NOT use for live signals.
        """
        assert self._hmm is not None
        return self._hmm.predict_proba(X_scaled)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame, max_fit_days: Optional[int] = None) -> "RegimeHMM":
        """
        Fit the HMM on ``df``.

        Among ``n_init`` random restarts, converged models are preferred; the
        best by log-likelihood within that set is retained.  If none converge,
        the best non-converged model is used with a warning.
        """
        _validate_dataframe(df, context="fit")
        X_scaled, _ = self._preprocess(df, fit=True)
        if max_fit_days is not None and len(X_scaled) > max_fit_days:
            X_scaled = X_scaled[-max_fit_days:]

        n_obs, n_features = X_scaled.shape
        approx_params = _approx_diag_hmm_params(n_features)
        if n_obs < approx_params:
            warnings.warn(
                "Training sample may be too small for the current feature set "
                f"(n_obs={n_obs}, approx_free_params={approx_params}). "
                "Regime estimates can be unstable; consider a longer training window.",
                UserWarning,
                stacklevel=2,
            )

        converged_models:    list[tuple[float, GaussianHMM]] = []
        unconverged_models:  list[tuple[float, GaussianHMM]] = []

        for i in range(self.n_init):
            hmm = GaussianHMM(
                n_components=K,
                covariance_type="diag",
                random_state=self.random_state + i,
                n_iter=self.n_iter,
                implementation="log",
                tol=1e-4,
                min_covar=1e-4,
            )
            try:
                with warnings.catch_warnings():
                    with suppress(logging.WARNING, module="hmmlearn"):
                        warnings.filterwarnings("ignore", module="hmmlearn")
                        hmm.fit(X_scaled)
                        score = hmm.score(X_scaled)
            except Exception as exc:
                logger.debug("Restart %d failed: %s", i, exc)
                continue

            bucket = converged_models if hmm.monitor_.converged else unconverged_models
            bucket.append((score, hmm))

        if not converged_models and not unconverged_models:
            raise RuntimeError(
                "All HMM fitting restarts failed. Check input data."
            )

        if not converged_models:
            warnings.warn(
                "No HMM restart converged. Regime estimates may be unreliable. "
                "Consider increasing n_iter or reviewing data quality.",
                UserWarning,
                stacklevel=2,
            )
            pool = unconverged_models
        else:
            logger.info(
                "%d/%d restarts converged.",
                len(converged_models), self.n_init,
            )
            pool = converged_models

        best_score, best_model = max(pool, key=lambda x: x[0])
        logger.info("Best log-likelihood: %.4f", best_score)

        self._hmm       = best_model
        means_orig      = self._scaler.inverse_transform(self._hmm.means_)
        self._label_map = _label_states(means_orig)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pl.DataFrame,
        *,
        smoothed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute posterior probabilities and MAP regime labels for ``df``.

        Parameters
        ----------
        df : pl.DataFrame
        smoothed : bool
            Use smoothed (forward-backward) posterior.  Research only —
            this looks ahead.  Default False (filtered / causal).

        Returns
        -------
        proba : np.ndarray, shape (len(df), K)
            Posterior probabilities in Regime enum order.
            Leading ``n_dropped`` rows are NaN (insufficient warmup data).
        labels : np.ndarray, shape (len(df),)
            Argmax regime index (int8).  -1 for warmup rows.
        n_dropped : int
            Number of leading rows that could not be scored.
        """
        self._check_fitted()
        _validate_dataframe(df, context="predict")

        X_scaled, n_dropped = self._preprocess(df)

        raw_proba     = self._smoothed_proba(X_scaled) if smoothed else self._filtered_proba(X_scaled)
        proba_ordered = _reorder_proba(raw_proba, self._label_map)

        # Build full-length arrays, padding the warmup head with NaN / -1
        n_total    = len(df)
        proba_full = np.full((n_total, K), np.nan)
        proba_full[n_dropped:] = proba_ordered

        labels_raw = np.argmax(proba_ordered, axis=1).astype(np.int8)
        if self.transition_penalty > 0:
            penalty = _build_transition_penalty_matrix(self.transition_penalty)
            labels_raw = _decode_with_soft_transition_penalty(proba_ordered, penalty)

        if self.min_holding_period > 1:
            labels_raw = _apply_holding_period(labels_raw, self.min_holding_period)

        labels_full             = np.full(n_total, -1, dtype=np.int8)
        labels_full[n_dropped:] = labels_raw

        return proba_full, labels_full, n_dropped

    # ------------------------------------------------------------------
    # Annotator
    # ------------------------------------------------------------------

    def annotate(
        self,
        df: pl.DataFrame,
        *,
        smoothed: bool = False,
        drop_warmup: bool = False,
        add_diagnostics: bool = True,
    ) -> pl.DataFrame:
        """
        Return ``df`` with regime columns appended.

        Added columns
        -------------
        regime              int8   Regime enum value; -1 = warmup / unknown
        regime_name         str    Human-readable label
        p_calm              f64  ]
        p_stress            f64  ] Posterior probability per regime
        p_crisis            f64  ]
        p_recovery          f64  ]
        regime_max_prob     f64    Max posterior probability
        regime_margin       f64    Top-1 minus top-2 posterior gap
        regime_entropy      f64    Posterior entropy
        regime_confidence   f64    1 - normalized entropy
        is_low_confidence   bool   confidence < low_confidence_threshold

        Parameters
        ----------
        smoothed : bool
            Passed through to predict(); research use only.
        drop_warmup : bool
            Drop rows where regime == -1 from the output.
        add_diagnostics : bool
            Include confidence and uncertainty columns.
        """
        proba, labels, _ = self.predict(df, smoothed=smoothed)
        diagnostics = _confidence_metrics(proba, self.low_confidence_threshold)

        _name = {r.value: r.name for r in Regime}
        _name[-1] = "UNKNOWN"

        out = df.with_columns(
            pl.Series("regime",      labels,                       dtype=pl.Int8),
            pl.Series("regime_name", [_name[int(v)] for v in labels], dtype=pl.Utf8),
        )
        for r in Regime:
            out = out.with_columns(
                pl.Series(f"p_{r.name.lower()}", proba[:, r.value], dtype=pl.Float64)
            )

        if add_diagnostics:
            out = out.with_columns(
                pl.Series("regime_max_prob", diagnostics["max_prob"], dtype=pl.Float64),
                pl.Series("regime_margin", diagnostics["margin"], dtype=pl.Float64),
                pl.Series("regime_entropy", diagnostics["entropy"], dtype=pl.Float64),
                pl.Series("regime_confidence", diagnostics["confidence"], dtype=pl.Float64),
                pl.Series("is_low_confidence", diagnostics["is_low_confidence"], dtype=pl.Boolean),
            )

        if drop_warmup:
            out = out.filter(pl.col("regime") >= 0)

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def regime_stats(self, df: pl.DataFrame, *, smoothed: bool = False) -> pl.DataFrame:
        """
        Per-regime summary: bar count, mean/std VIX, mean posterior probability.

        Use this after fitting to verify that the label mapping makes
        economic sense (e.g. CRISIS should have the highest mean VIX).
        """
        ann = self.annotate(df, smoothed=smoothed, drop_warmup=True)
        return (
            ann.group_by("regime_name")
            .agg(
                pl.len().alias("n_bars"),
                pl.col("vix").mean().round(2).alias("mean_vix"),
                pl.col("vix").std().round(2).alias("std_vix"),
                *[
                    pl.col(f"p_{r.name.lower()}")
                    .mean()
                    .round(3)
                    .alias(f"mean_p_{r.name.lower()}")
                    for r in Regime
                ],
            )
            .sort("mean_vix")
        )

    def transition_matrix(self, df: pl.DataFrame, *, smoothed: bool = False) -> pl.DataFrame:
        """
        Empirical 1-step regime transition matrix (rows = from, cols = to).

        Compare with ``self._hmm.transmat_`` to sanity-check that the label
        mapping and holding-period filter are behaving as expected.
        """
        _, labels, _ = self.predict(df, smoothed=smoothed)
        valid = labels[labels >= 0]

        mat = np.zeros((K, K), dtype=float)
        for t in range(len(valid) - 1):
            mat[valid[t], valid[t + 1]] += 1

        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat /= row_sums

        names = [r.name for r in Regime]
        return (
            pl.DataFrame(
                {names[i]: mat[i].tolist() for i in range(K)},
                schema={n: pl.Float64 for n in names},
            )
            .with_columns(pl.Series("from_regime", names))
            .select(["from_regime"] + names)
        )


class WalkForwardRegimeHMM:
    """
    Walk-forward orchestrator for RegimeHMM.

    This class keeps scheduling and data-splitting logic separate from
    RegimeHMM, which remains responsible for single-window fit/predict.

    Parameters
    ----------
    training_window : str
        Polars duration string for rolling train lookback, e.g. "1y", "18mo".
    walk_forward_window : str
        Polars duration string for retrain cadence / prediction block, e.g. "1mo", "2w".
    n_init, n_iter, random_state, min_holding_period
        Forwarded to RegimeHMM for each retrain.
    """

    def __init__(
        self,
        training_window: str = "3y",
        walk_forward_window: str = "1mo",
        *,
        n_init: int = 10,
        n_iter: int = 1_000,
        random_state: int = 42,
        min_holding_period: int = 5,
        transition_penalty: float = 0.20,
        low_confidence_threshold: float = 0.55,
    ) -> None:
        self.training_window = training_window
        self.walk_forward_window = walk_forward_window
        self.n_init = n_init
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_holding_period = min_holding_period
        self.transition_penalty = transition_penalty
        self.low_confidence_threshold = low_confidence_threshold

    @staticmethod
    def _offset_date(value: object, interval: str, *, backward: bool) -> object:
        sign = "-" if backward else ""
        shifted = (
            pl.DataFrame({"date": [value]})
            .with_columns(pl.col("date").dt.offset_by(f"{sign}{interval}").alias("shifted"))
        )
        return shifted["shifted"][0]

    def run(
        self,
        df: pl.DataFrame,
        *,
        oos_callback: Callable[[pl.DataFrame], pl.DataFrame] = lambda x: x,
        smoothed: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Interval-driven retrain / prediction walk-forward.

        Returns
        -------
        predictions : pl.DataFrame
            Out-of-sample rows with regime columns.
        metadata : pl.DataFrame
            Per-window diagnostics for audit and debugging.
        """
        if "date" not in df.columns:
            raise ValueError("Input df must include a 'date' column.")

        if smoothed:
            warnings.warn(
                "smoothed=True uses look-ahead posterior and is research-only.",
                UserWarning,
                stacklevel=2,
            )

        df_sorted = df.sort("date")
        n_total = len(df_sorted)
        min_rows = max(RV_WINDOW, CONTEXT_WINDOW) + K + 1

        if n_total < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows for walk-forward; got {n_total}."
            )

        # Validate interval strings early with a no-op offset on first date.
        first_date = df_sorted["date"][0]
        try:
            _ = self._offset_date(first_date, self.training_window, backward=True)
            _ = self._offset_date(first_date, self.walk_forward_window, backward=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Invalid interval string. Expected Polars duration syntax like "
                "'1y', '6mo', '2w', '10d'."
            ) from exc

        df_indexed = df_sorted.with_row_index("idx")
        windows = (
            df_indexed.group_by_dynamic(
                "date",
                every=self.walk_forward_window,
                period=self.walk_forward_window,
                closed="left",
                label="left",
            )
            .agg(
                pl.col("idx").min().alias("test_start_idx"),
                (pl.col("idx").max() + 1).alias("test_end_idx"),
                pl.col("date").min().alias("test_start_date"),
                pl.col("date").max().alias("test_end_date"),
            )
            .drop_nulls(subset=["test_start_idx", "test_end_idx", "test_start_date", "test_end_date"])
            .sort("test_start_idx")
        )

        if windows.is_empty() or len(windows) < 2:
            raise ValueError(
                "Not enough windows for walk-forward. Increase data span or reduce walk_forward_window."
            )

        context_bars = max(min_rows - 1, RV_WINDOW + 5)

        pred_blocks: list[pl.DataFrame] = []
        meta_rows: list[dict[str, object]] = []

        retrain_id = 0
        for w in windows.iter_rows(named=True):
            test_start = int(w["test_start_idx"])
            test_end = int(w["test_end_idx"])
            if test_end <= test_start:
                continue

            window_id = retrain_id
            retrain_id += 1

            test_start_date = w["test_start_date"]
            train_start_date = self._offset_date(test_start_date, self.training_window, backward=True)
            train_df = df_sorted.filter(
                (pl.col("date") >= pl.lit(train_start_date))
                & (pl.col("date") < pl.lit(test_start_date))
            )

            if len(train_df) < min_rows:
                continue

            score_start = max(0, test_start - context_bars)
            score_df = df_sorted.slice(score_start, test_end - score_start)
            
            model = RegimeHMM(
                n_init=self.n_init,
                n_iter=self.n_iter,
                random_state=self.random_state + window_id,
                min_holding_period=self.min_holding_period,
                transition_penalty=self.transition_penalty,
                low_confidence_threshold=self.low_confidence_threshold,
            )

            try:
                model = model.fit(train_df)
            except ValueError as exc:
                logger.warning(
                    "Skipping window retrain_id=%s because training data is not usable: %s",
                    window_id,
                    exc,
                )
                continue

            try:
                scored_block = model.annotate(score_df, smoothed=smoothed)
            except ValueError as exc:
                logger.warning(
                    "Skipping window retrain_id=%s because scoring data is not usable: %s",
                    window_id,
                    exc,
                )
                continue

            test_offset = test_start - score_start
            oos_block = scored_block.slice(test_offset, test_end - test_start)
            if oos_block.is_empty():
                logger.warning(
                    "Skipping window retrain_id=%s because OOS block is empty after scoring.",
                    window_id,
                )
                continue

            pred_blocks.append(
                oos_callback(oos_block.with_columns(pl.lit(window_id).alias("retrain_id")))
            )

            unknown_count = int((oos_block["regime"] < 0).sum())
            meta_rows.append(
                {
                    "retrain_id": window_id,
                    "train_start": train_df["date"][0],
                    "train_end": train_df["date"][-1],
                    "test_start": oos_block["date"][0],
                    "test_end": oos_block["date"][-1],
                    "n_train": len(train_df),
                    "n_test": len(oos_block),
                    "n_unknown": unknown_count,
                    "training_window": self.training_window,
                    "walk_forward_window": self.walk_forward_window,
                }
            )

        if not pred_blocks:
            raise RuntimeError(
                "No walk-forward windows were produced. Increase data span or adjust interval settings."
            )

        predictions = pl.concat(pred_blocks).sort("date")
        predictions = predictions.unique(subset=["date"], keep="first").sort("date")
        metadata = pl.DataFrame(meta_rows).sort("retrain_id")
        return predictions, metadata
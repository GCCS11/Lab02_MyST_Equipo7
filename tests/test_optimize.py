"""Pruebas de la función objetivo, del gate del walk-forward, del walk-forward por régimen y de freeze_theta."""
import json

import numpy as np
import pandas as pd

from src.backtest import backtest_regimes
from src.data import segment_ids
from src.optimize import (
    NEVER,
    PENALTY,
    freeze_theta,
    make_folds,
    regime_n_min,
    regime_window_objective,
    walk_forward,
    walk_forward_regimes,
    window_objective,
)
from src.regimes import regime_features
from tests.test_golden import regime_path
from tests.test_regimes import regime_prices
from tests.test_signals import make_prices

P = {
    "n_donchian": 30, "n_roc": 10, "n_ema": 15, "n_atr": 10, "k": 1.0,
    "m": 4.0, "r": 2.0, "max_hold": 100, "signal_exit_after": NEVER, "risk_frac": 0.01,
}


def small_folds(df):
    return make_folds(segment_ids(df), train_bars=1200, test_bars=300, step_bars=300)


def test_objective_penalizes_too_few_trades():
    df = make_prices()
    assert window_objective(df, P, n_min=10**6) < PENALTY / 2


def test_objective_returns_calmar_when_enough_trades():
    df = make_prices()
    assert window_objective(df, P, n_min=1) > PENALTY / 2


def test_walk_forward_gate_keeps_cash_when_no_edge_in_sample():
    """Con un gate inalcanzable no se opera ninguna semana y el equity queda plano."""
    df = make_prices(3000)
    table, oos, _ = walk_forward(df, small_folds(df), n_trials=3, gate=np.inf)
    assert not table["traded"].any()
    assert (table["oos_trades"] == 0).all()
    assert (oos.equity == 1_000_000.0).all()
    assert oos.total_costs == 0.0


def test_walk_forward_without_gate_trades_every_week():
    df = make_prices(3000)
    table, _, _ = walk_forward(df, small_folds(df), n_trials=3)
    assert table["traded"].all()


def test_regime_objective_penalizes_a_regime_that_never_occurs():
    df = make_prices()
    labels = pd.Series(1, index=df.index)  # todas las barras son del régimen 1
    assert regime_window_objective(df, segment_ids(df), labels, 2, P, n_min=1) < PENALTY / 2


def test_regime_min_trades_scale_with_regime_share():
    assert regime_n_min(5000, 10000, 10) == 5
    assert regime_n_min(100, 10000, 10) == 3  # nunca menos de 3


def test_trades_record_the_regime_they_were_opened_in():
    df, seg, labels, specs = regime_path()
    res = backtest_regimes(df, seg, labels, specs)
    assert res.trades["regime"].tolist() == [1]


def regime_ohlc():
    """Precios sintéticos de tres regímenes con las cuatro columnas OHLC."""
    close = regime_prices()["Close"]
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.0004,
            "Low": np.minimum(open_, close) * 0.9996,
            "Close": close,
        }
    )


def regime_setup():
    df = regime_ohlc()
    seg = segment_ids(df)
    feats = regime_features(df, seg, 300)
    return df, seg, feats, make_folds(seg, train_bars=3000, test_bars=700, step_bars=700)


def test_walk_forward_regimes_keeps_cash_when_gate_is_unreachable():
    df, seg, feats, folds = regime_setup()
    table, oos, _ = walk_forward_regimes(df, feats, seg, folds, n_trials=3, gate=np.inf)
    assert (table["regimes_traded"] == 0).all()
    assert (oos.equity == 1_000_000.0).all()
    assert {"bars_reversion", "bars_tendencia", "bars_crisis"} <= set(table.columns)


def test_walk_forward_regimes_trades_and_labels_each_trade_with_a_regime():
    df, seg, feats, folds = regime_setup()
    table, oos, _ = walk_forward_regimes(df, feats, seg, folds, n_trials=5, gate=-1e9)
    assert len(oos.trades) > 0, "la prueba sería vacía sin operaciones"
    assert set(oos.trades["regime"]) <= {0, 1, 2}


def test_freeze_theta_stores_none_when_gate_is_unreachable():
    df, seg, feats, _ = regime_setup()
    theta = freeze_theta(df, feats, seg, n_trials=3, n_min=1, gate=np.inf)
    assert theta["single"]["params"] is None
    assert all(r["params"] is None for r in theta["regimes"].values())


def test_freeze_theta_is_serializable_and_complete_when_gate_is_permissive():
    df, seg, feats, _ = regime_setup()
    theta = freeze_theta(df, feats, seg, n_trials=3, n_min=1, gate=-np.inf)
    restored = json.loads(json.dumps(theta))
    assert set(restored["single"]["params"]) == set(P)
    assert restored["meta"]["train_bars"] == len(df)
    assert set(restored["regimes"]) == {"reversion", "tendencia", "crisis"}



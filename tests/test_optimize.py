"""Pruebas de la función objetivo de una ventana y del gate del walk-forward."""
from src.data import segment_ids
from src.optimize import NEVER, PENALTY, make_folds, walk_forward, window_objective
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
    table, oos, _ = walk_forward(df, small_folds(df), n_trials=3, gate=1e9)
    assert not table["traded"].any()
    assert (table["oos_trades"] == 0).all()
    assert (oos.equity == 1_000_000.0).all()
    assert oos.total_costs == 0.0


def test_walk_forward_without_gate_trades_every_week():
    df = make_prices(3000)
    table, _, _ = walk_forward(df, small_folds(df), n_trials=3)
    assert table["traded"].all()


    
"""Pruebas de la evaluación final con θ congelado: sin operaciones antes del test y sin fugas."""
import pandas as pd

from src.data import segment_ids
from src.optimize import NEVER, buy_and_hold, evaluate_frozen
from tests.test_optimize import regime_ohlc

PARAMS = {
    "n_donchian": 30, "n_roc": 10, "n_ema": 15, "n_atr": 10, "k": 1.0,
    "m": 4.0, "r": 2.0, "max_hold": 100, "signal_exit_after": NEVER, "risk_frac": 0.01,
}
THETA = {
    "single": {"params": PARAMS},
    "rules": {"vol_crisis": 0.0016, "r2_trend": 0.5},
    "regimes": {
        "reversion": {"params": PARAMS},
        "tendencia": {"params": {**PARAMS, "n_donchian": 60, "m": 6.0}},
        "crisis": {"params": None},
    },
}


def split(n_train=9000):
    df = regime_ohlc()
    return df.iloc[:n_train], df.iloc[n_train:]


def test_no_trade_is_opened_before_the_test_starts():
    train, test = split()
    res = evaluate_frozen(train, test, THETA)
    for key in ("theta_unico", "theta_por_regimen"):
        trades = res[key].trades
        assert len(trades) > 0, f"la prueba sería vacía sin operaciones ({key})"
        assert (trades["entry_time"] >= test.index[0]).all()
        assert len(res[key].equity) == len(test)


def test_a_strategy_without_params_stays_in_cash():
    train, test = split()
    theta = {**THETA, "single": {"params": None},
             "regimes": {k: {"params": None} for k in THETA["regimes"]}}
    res = evaluate_frozen(train, test, theta)
    for key in ("theta_unico", "theta_por_regimen"):
        assert (res[key].equity == 1_000_000.0).all()
        assert len(res[key].trades) == 0


def closed_before(result, t_time):
    """Operaciones ya cerradas antes de t_time (vacío si la estrategia no operó)."""
    trades = result.trades
    if trades.empty:
        return trades
    return trades[trades["exit_time"] < t_time].reset_index(drop=True)


def test_evaluation_does_not_use_data_after_t():
    """Recortar el test en t no cambia el equity previo a t ni las operaciones ya cerradas."""
    train, test = split()
    full = evaluate_frozen(train, test, THETA)
    for key in ("theta_unico", "theta_por_regimen"):
        assert len(full[key].trades) > 0, f"la prueba sería vacía sin operaciones ({key})"
    for t in (800, 2000, 3500):
        part = evaluate_frozen(train, test.iloc[: t + 1], THETA)
        for key in ("theta_unico", "theta_por_regimen"):
            pd.testing.assert_series_equal(part[key].equity.iloc[:-1], full[key].equity.iloc[:t])
            done_full = closed_before(full[key], test.index[t])
            done_part = closed_before(part[key], test.index[t])
            assert len(done_part) == len(done_full)
            if len(done_full):
                pd.testing.assert_frame_equal(done_part, done_full)


def test_buy_and_hold_ignores_the_jump_between_segments():
    df = regime_ohlc().iloc[:1000].copy()
    df2 = regime_ohlc().iloc[1000:2000].copy()
    df2.index = df2.index + pd.Timedelta(days=30)  # hueco largo: abre un tramo nuevo
    test = pd.concat([df, df2])
    res = buy_and_hold(test, segment_ids(test))
    first = df["Close"].iloc[-1] / df["Open"].iloc[0]
    second = df2["Close"].iloc[-1] / df2["Open"].iloc[0]
    assert abs(res.equity.iloc[-1] - 1_000_000.0 * first * second) < 1e-6

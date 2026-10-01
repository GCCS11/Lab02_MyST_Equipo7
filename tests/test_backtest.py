"""Pruebas del motor: salida stop-primero, sizing, estado y contabilidad."""
import pandas as pd
import pytest

from src.backtest import FEE, Portfolio, Position, check_exit, size_position

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def test_stop_wins_when_both_levels_in_same_bar():
    """(a) Largo con stop y target dentro de la misma barra: se cierra como stop_loss."""
    long_pos = Position(1, 1.0, 100.0, 95.0, 105.0, T0)
    assert check_exit(long_pos, 100.0, 106.0, 94.0) == ("stop_loss", 95.0)
    short_pos = Position(-1, 1.0, 100.0, 105.0, 95.0, T0)
    assert check_exit(short_pos, 100.0, 106.0, 94.0) == ("stop_loss", 105.0)


def test_gap_fills_at_open():
    """Si la barra abre más allá del stop, se llena a la apertura, no al nivel."""
    long_pos = Position(1, 1.0, 100.0, 95.0, 105.0, T0)
    assert check_exit(long_pos, 94.0, 96.0, 93.0) == ("stop_loss", 94.0)


def test_sizing_returns_budgeted_dollar_risk():
    """(b) Sin tope de capital, qty * distancia al stop es exactamente el riesgo presupuestado."""
    equity, risk_frac, stop_dist = 1_000_000, 0.001, 1500.0
    qty = size_position(equity, price=30_000.0, stop_distance=stop_dist, risk_frac=risk_frac)
    assert qty * stop_dist == pytest.approx(risk_frac * equity)


def test_sizing_never_exceeds_capital():
    """Con stop muy corto el riesgo presupuestado pediría apalancamiento: manda el tope."""
    equity = 1_000_000
    qty = size_position(equity, price=30_000.0, stop_distance=50.0, risk_frac=0.01)
    assert qty * 30_000.0 * (1 + FEE) <= equity * (1 + 1e-12)


def test_never_two_open_positions():
    """(c) La máquina de estados no admite dos posiciones abiertas a la vez."""
    p = Portfolio(1_000_000)
    pos = Position(1, 1.0, 100.0, 95.0, 105.0, T0)
    p.open(pos)
    with pytest.raises(RuntimeError):
        p.open(pos)


def test_accounting_matches_trades():
    """Sin posición abierta, el valor es el efectivo y refleja el P&L neto de comisiones."""
    p = Portfolio(1_000_000)
    p.open(Position(1, 2.0, 100.0, 95.0, 105.0, T0))
    p.close(110.0, T0, "take_profit")
    expected_pnl = 2.0 * 10.0 - FEE * (2.0 * 100.0 + 2.0 * 110.0)
    assert p.trades[0]["pnl"] == pytest.approx(expected_pnl)
    assert p.equity(110.0) == pytest.approx(1_000_000 + expected_pnl)
    assert p.total_costs == pytest.approx(FEE * p.traded_notional)
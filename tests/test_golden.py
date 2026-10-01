import pandas as pd
import pytest

from src.backtest import BacktestParams, backtest


def test_backtest_matches_hand_calculation():
    # m=2, r=2, risk_frac=1%, comision 0.125%, capital 1,000,000. ATR = 1 en todas las barras.
    # Señal al cierre de cada barra: [+1, 0, -1, 0, 0, 0]
    idx = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 101, 104, 104.5, 106.5],
            "High": [101, 102, 105, 105, 107, 107],
            "Low": [99, 99, 100, 103, 104, 106],
            "Close": [100, 101, 104, 104.5, 106.5, 106.5],
        },
        index=idx,
        dtype=float,
    )
    signal = pd.Series([1, 0, -1, 0, 0, 0], index=idx)
    atr = pd.Series(1.0, index=idx)
    seg = pd.Series(0, index=idx)

    res = backtest(df, signal, atr, seg, BacktestParams(m=2, r=2))

    # Trade 1 (largo): entra en la apertura de la barra 1 a 100. Stop = 2, qty = 10,000/2 = 5,000.
    #   Nocional 500,000, comisión 625, efectivo = 1,000,000 - 500,000 - 625 = 499,375.
    #   Cierre barra 1: 499,375 + 5,000*101 = 1,004,375. Target = 104.
    # Barra 2: el máximo (105) toca el target 104 y el mínimo (100) no toca el stop 98.
    #   Salida a 104: +520,000 - 650 de comisión => efectivo = 1,018,725.
    # Trade 2 (corto): señal -1 al cierre de la barra 2, entra en la apertura de la barra 3 a 104.
    #   qty = 0.01*1,018,725/2 = 5,093.625. Nocional 529,737, comisión 662.17125.
    #   efectivo = 1,018,725 + 529,737 - 662.17125 = 1,547,799.82875.
    #   Cierre barra 3 (104.5): 1,547,799.82875 - 5,093.625*104.5 = 1,015,516.01625.
    # Barra 4: el máximo (107) supera el stop 106 y el target 100 no se toca => stop en 106.
    #   Recompra: 539,924.25, comisión 674.9053125 => efectivo = 1,007,200.6734375.
    expected_equity = [
        1_000_000.0,
        1_004_375.0,
        1_018_725.0,
        1_015_516.01625,
        1_007_200.6734375,
        1_007_200.6734375,
    ]
    assert res.equity.tolist() == pytest.approx(expected_equity, rel=1e-12)
    assert res.trades["reason"].tolist() == ["take_profit", "stop_loss"]
    assert res.total_costs == pytest.approx(2_612.0765625)
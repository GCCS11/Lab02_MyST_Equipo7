"""Datos sintéticos compartidos por las pruebas y la prueba de la regla de confirmación 2 de 3."""
import numpy as np
import pandas as pd

from src.signals import confirm_signal


def make_prices(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Precios OHLC sintéticos (caminata aleatoria) con un hueco de ~10.8 h que parte la serie en dos tramos."""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0},
        index=idx,
    )
    return df.drop(df.index[1000:1130])


def test_confirmation_rule():
    """Un indicador a favor no abre posición; dos o más sí."""
    signals = pd.DataFrame(
        [(1, 0, 0), (0, 0, -1), (1, -1, 0), (1, 1, 0), (1, 1, 1), (-1, -1, 0), (1, 1, -1)],
        columns=["a", "b", "c"],
    )
    expected = [0, 0, 0, 1, 1, -1, 1]
    assert confirm_signal(signals).tolist() == expected

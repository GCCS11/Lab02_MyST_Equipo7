"""Prueba de truncamiento de las variables de régimen, columna por columna."""
import numpy as np

from src.data import segment_ids
from src.regimes import regime_features
from tests.test_signals import make_prices

WINDOW = 120


def test_regime_features_are_causal_in_every_column():
    """Recalcular sobre df[:t+1] no cambia el valor de ninguna columna en t."""
    df = make_prices()
    full = regime_features(df, segment_ids(df), WINDOW)
    assert full.notna().all(axis=1).sum() > 500, "la prueba sería vacía sin valores"
    for t in (300, 900, 1100, 1500, 1800):
        cut = df.iloc[: t + 1]
        part = regime_features(cut, segment_ids(cut), WINDOW)
        for col in full.columns:
            np.testing.assert_allclose(
                part[col].iloc[-1], full[col].iloc[t], rtol=1e-8, atol=1e-12, err_msg=f"{col} en t={t}"
            )
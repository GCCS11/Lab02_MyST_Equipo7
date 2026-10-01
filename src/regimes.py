"""Detección de régimen de mercado: variables sobre ventana móvil de 1 semana."""
import numpy as np
import pandas as pd

WEEK = 7 * 288  # 1 semana en barras de 5 minutos


def regime_features(df: pd.DataFrame, seg: pd.Series, window: int = WEEK) -> pd.DataFrame:
    """Tres variables de régimen, causales y calculadas dentro de cada tramo continuo:

    vol        volatilidad realizada: desviación estándar de los retornos logarítmicos.
    trend_r2   fuerza de tendencia: R^2 de la regresión lineal del log-precio contra el
               tiempo (cercano a 1 si el precio avanza en línea recta).
    autocorr1  fuerza de reversión: autocorrelación de orden 1 de los retornos
               (negativa si los movimientos tienden a revertirse).

    Son NaN hasta completar la primera ventana de cada tramo."""
    parts = []
    for _, d in df.groupby(seg, sort=False):
        logp = np.log(d["Close"])
        ret = logp.diff()
        t = pd.Series(np.arange(len(d), dtype=float), index=d.index)
        parts.append(
            pd.DataFrame(
                {
                    "vol": ret.rolling(window).std(),
                    "trend_r2": logp.rolling(window).corr(t) ** 2,
                    "autocorr1": ret.rolling(window).corr(ret.shift(1)),
                }
            )
        )
    return pd.concat(parts)
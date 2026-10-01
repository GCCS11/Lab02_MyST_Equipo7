import numpy as np
import pandas as pd

BARS_PER_YEAR = 365 * 288  # BTC opera 24/7; barras de 5 minutos


def bar_returns(equity: pd.Series) -> pd.Series:
    """Retorno simple por barra."""
    return equity.pct_change().dropna()


def years_of_data(equity: pd.Series) -> float:
    """Tiempo operado en años, contando barras (no días de calendario), para no
    contar los huecos de datos como tiempo transcurrido."""
    return (len(equity) - 1) / BARS_PER_YEAR


def annualized_return(equity: pd.Series) -> float:
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years_of_data(equity)) - 1


def annualized_vol(equity: pd.Series) -> float:
    return bar_returns(equity).std() * np.sqrt(BARS_PER_YEAR)


def sharpe(equity: pd.Series) -> float:
    r = bar_returns(equity)
    return r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)


def sortino(equity: pd.Series) -> float:
    r = bar_returns(equity)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return r.mean() / downside * np.sqrt(BARS_PER_YEAR)


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Caída porcentual respecto al máximo previo (valores <= 0)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_curve(equity).min()


def calmar(equity: pd.Series) -> float:
    """Retorno anualizado entre |máximo drawdown|; NaN si no hubo drawdown."""
    mdd = abs(max_drawdown(equity))
    return annualized_return(equity) / mdd if mdd > 0 else float("nan")


def win_rate(trades: pd.DataFrame) -> float:
    """Fracción de operaciones con P&L neto positivo."""
    return float((trades["pnl"] > 0).mean()) if len(trades) else float("nan")


def annual_turnover(equity: pd.Series, traded_notional: float) -> float:
    """Nocional operado (entradas y salidas) por año, en múltiplos del equity medio."""
    return traded_notional / equity.mean() / years_of_data(equity)


def summary(result) -> dict:
    """Todas las métricas de un BacktestResult en un diccionario."""
    eq = result.equity
    return {
        "retorno_total": eq.iloc[-1] / eq.iloc[0] - 1,
        "retorno_anual": annualized_return(eq),
        "volatilidad_anual": annualized_vol(eq),
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(eq),
        "n_operaciones": len(result.trades),
        "win_rate": win_rate(result.trades),
        "turnover_anual": annual_turnover(eq, result.traded_notional),
        "costos_totales": result.total_costs,
    }



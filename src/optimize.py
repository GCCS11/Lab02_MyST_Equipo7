"""Optimización de hiperparámetros y walk-forward."""
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from src.backtest import BacktestParams, BacktestResult, run_strategy
from src.metrics import calmar
from src.signals import SignalParams

BARS_PER_DAY = 288
N_MIN_TRADES = 10  # mínimo de operaciones por ventana de entrenamiento
N_TRIALS = 150  # el lab pide entre 100 y 200 pruebas por ventana
SEED = 42
PENALTY = -1e6  # valor del objetivo para configuraciones inválidas
NEVER = 10**9  # signal_exit_after = NEVER equivale a no salir por señal opuesta


@dataclass(frozen=True)
class Fold:
    """Posiciones (iloc) en el DataFrame limpio. El train va de train_start a train_end
    (exclusivo) y el test, inmediatamente después, de train_end a test_end (exclusivo)."""

    train_start: int
    train_end: int
    test_end: int


def make_folds(
    seg: pd.Series,
    train_bars: int = 30 * BARS_PER_DAY,
    test_bars: int = 7 * BARS_PER_DAY,
    step_bars: int = 7 * BARS_PER_DAY,
) -> list[Fold]:
    """Ventanas del walk-forward (1 mes de train, 1 semana de test, paso semanal)
    dentro de cada tramo continuo: ninguna ventana cruza un corte de datos."""
    seg_v = seg.to_numpy()
    folds = []
    for s in pd.unique(seg_v):
        pos = np.flatnonzero(seg_v == s)
        start, stop = pos[0], pos[-1] + 1
        a = start
        while a + train_bars + test_bars <= stop:
            folds.append(Fold(a, a + train_bars, a + train_bars + test_bars))
            a += step_bars
    return folds


def suggest_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda (ver SPEC, sección 9). Las ventanas van en escala log."""
    return {
        "n_donchian": trial.suggest_int("n_donchian", 12, 864, log=True),
        "n_roc": trial.suggest_int("n_roc", 12, 864, log=True),
        "n_ema": trial.suggest_int("n_ema", 12, 864, log=True),
        "n_atr": trial.suggest_int("n_atr", 12, 864, log=True),
        "k": trial.suggest_float("k", 1.0, 3.0),
        "m": trial.suggest_float("m", 10.0, 60.0),
        "r": trial.suggest_float("r", 1.5, 4.0),
        "max_hold": trial.suggest_int("max_hold", 1 * BARS_PER_DAY, 10 * BARS_PER_DAY),
        "signal_exit_after": trial.suggest_categorical("signal_exit_after", [0, BARS_PER_DAY, NEVER]),
        "risk_frac": trial.suggest_float("risk_frac", 0.0025, 0.03, log=True),
    }


def to_params(p: dict) -> tuple[SignalParams, BacktestParams]:
    """Convierte un diccionario de hiperparámetros en los objetos del pipeline."""
    sp = SignalParams(p["n_donchian"], p["n_roc"], p["n_ema"], p["n_atr"], p["k"])
    bp = BacktestParams(
        m=p["m"],
        r=p["r"],
        risk_frac=p["risk_frac"],
        max_hold=p["max_hold"],
        signal_exit_after=p["signal_exit_after"],
    )
    return sp, bp


def window_objective(df: pd.DataFrame, p: dict, n_min: int = N_MIN_TRADES) -> float:
    """Calmar de la ventana. Si hay menos de n_min operaciones (o el Calmar no está
    definido) devuelve una penalización graduada por número de operaciones, para que
    el optimizador tenga hacia dónde moverse en lugar de una meseta plana."""
    sp, bp = to_params(p)
    res = run_strategy(df, sp, bp)
    n = len(res.trades)
    if n < n_min:
        return PENALTY + n
    c = calmar(res.equity)
    return c if np.isfinite(c) else PENALTY + n


def optimize_window(
    df: pd.DataFrame, n_trials: int = N_TRIALS, n_min: int = N_MIN_TRADES, seed: int = SEED
) -> optuna.Study:
    """Busca los hiperparámetros que maximizan el Calmar en una ventana (TPE)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: window_objective(df, suggest_params(t), n_min), n_trials=n_trials)
    return study


def run_oos(df: pd.DataFrame, fold: Fold, p: dict):
    """Evalúa los hiperparámetros congelados p en la semana de prueba de un fold.

    Los indicadores se calculan sobre datos que empiezan `warm` barras antes de la
    prueba (calentamiento, solo pasado) y no se abre ninguna posición antes de ella.
    Devuelve (resultado del backtest, posición donde empieza la prueba en la ventana)."""
    sp, bp = to_params(p)
    warm = max(sp.n_donchian, sp.n_roc, sp.n_ema, sp.n_atr) + 2
    lo = max(fold.train_start, fold.train_end - warm)
    window = df.iloc[lo : fold.test_end]
    start = fold.train_end - lo
    return run_strategy(window, sp, bp, start=start), start


def walk_forward(
    df: pd.DataFrame,
    folds: list[Fold],
    n_trials: int = N_TRIALS,
    n_min: int = N_MIN_TRADES,
    gate: float | None = None,
):
    """Optimiza en cada ventana de entrenamiento y evalúa con los parámetros congelados
    en la semana siguiente. Las semanas fuera de muestra se encadenan por retornos.

    Si gate no es None, la semana solo se opera cuando el mejor Calmar de entrenamiento
    supera gate; si no, se queda en efectivo (el optimizador no encontró ventaja ni
    dentro de muestra). Con gate=None todas las semanas se operan.

    Devuelve (tabla por fold, resultado OOS encadenado, benchmark comprar y mantener)."""
    cash0 = BacktestParams(m=1, r=1).cash
    level, bench_level = cash0, cash0
    rows, equity_parts, bench_parts, trades_parts = [], [], [], []
    costs = notional = 0.0
    for i, f in enumerate(folds):
        study = optimize_window(df.iloc[f.train_start : f.train_end], n_trials, n_min)
        p = study.best_params
        test = df.iloc[f.train_end : f.test_end]
        traded = gate is None or study.best_value > gate
        if traded:
            res, start = run_oos(df, f, p)
            path = res.equity.iloc[start:] / cash0
            week_trades = res.trades.copy()
            costs += res.total_costs
            notional += res.traded_notional
        else:
            path = pd.Series(1.0, index=test.index)
            week_trades = pd.DataFrame()
        equity_parts.append(level * path)
        level = equity_parts[-1].iloc[-1]
        bench = test["Close"] / test["Open"].iloc[0]
        bench_parts.append(bench_level * bench)
        bench_level = bench_parts[-1].iloc[-1]
        if len(week_trades):
            week_trades["fold"] = i
            trades_parts.append(week_trades)
        rows.append(
            {
                "fold": i,
                "test_start": test.index[0],
                "is_calmar": study.best_value,
                "traded": traded,
                "oos_return": path.iloc[-1] - 1,
                "oos_trades": len(week_trades),
                **{f"p_{k}": v for k, v in p.items()},
            }
        )
    oos = BacktestResult(
        equity=pd.concat(equity_parts).rename("equity"),
        trades=pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame(),
        total_costs=costs,
        traded_notional=notional,
    )
    benchmark = BacktestResult(pd.concat(bench_parts).rename("equity"), pd.DataFrame(), 0.0, 0.0)
    return pd.DataFrame(rows), oos, benchmark



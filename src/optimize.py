""""Optimización de hiperparámetros y walk-forward (con un solo θ o con un θ por régimen)."""
import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from src.backtest import (
    INITIAL_CASH,
    BacktestParams,
    BacktestResult,
    RegimeSpec,
    backtest_regimes,
    run_strategy,
)
from src.metrics import calmar
from src.regimes import NAMES, RuleRegimes, hold_labels
from src.signals import SignalParams, strategy_signal

BARS_PER_DAY = 288
N_MIN_TRADES = 10  # mínimo de operaciones por ventana de entrenamiento
N_TRIALS = 150  # el lab pide entre 100 y 200 pruebas por ventana
SEED = 42
PENALTY = -1e6  # valor del objetivo para configuraciones inválidas
NEVER = 10**9  # signal_exit_after = NEVER equivale a no salir por señal opuesta
B_MIN_BARS = BARS_PER_DAY  # barras mínimas de un régimen en la ventana para optimizar su θ
FINAL_N_MIN = 30  # mínimo de operaciones al ajustar θ una sola vez sobre todo el train


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


def _score(res: BacktestResult, n_min: int) -> float:
    """Calmar del resultado; si hay menos de n_min operaciones (o el Calmar no está
    definido) devuelve una penalización graduada por número de operaciones, para que
    el optimizador tenga hacia dónde moverse en lugar de una meseta plana."""
    n = len(res.trades)
    if n < n_min:
        return PENALTY + n
    c = calmar(res.equity)
    return c if np.isfinite(c) else PENALTY + n


def _tpe_study(objective, n_trials: int, seed: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: objective(suggest_params(t)), n_trials=n_trials)
    return study


def window_objective(df: pd.DataFrame, p: dict, n_min: int = N_MIN_TRADES) -> float:
    """Calmar de una ventana con un solo conjunto de parámetros."""
    sp, bp = to_params(p)
    return _score(run_strategy(df, sp, bp), n_min)


def optimize_window(
    df: pd.DataFrame, n_trials: int = N_TRIALS, n_min: int = N_MIN_TRADES, seed: int = SEED
) -> optuna.Study:
    """Busca los hiperparámetros que maximizan el Calmar en una ventana (TPE)."""
    return _tpe_study(lambda p: window_objective(df, p, n_min), n_trials, seed)


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


# ---------------------------------------------------------------------------
# Un θ por régimen
# ---------------------------------------------------------------------------
def regime_n_min(bars_regime: int, bars_window: int, n_min: int = N_MIN_TRADES) -> int:
    """Mínimo de operaciones exigido a un régimen: la misma densidad que el mínimo de la
    ventana completa, proporcional a las barras que el régimen ocupa (al menos 3)."""
    return max(3, int(np.ceil(n_min * bars_regime / bars_window)))


def regime_window_objective(df, seg, labels, regime: int, p: dict, n_min: int) -> float:
    """Calmar de la ventana operando solo mientras el régimen vigente sea `regime`."""
    sp, bp = to_params(p)
    signal, atr_series = strategy_signal(df, seg, sp)
    spec = RegimeSpec(signal, atr_series, bp)
    return _score(backtest_regimes(df, seg, labels, {regime: spec}), n_min)


def optimize_regime_window(
    df, seg, labels, regime: int, n_trials: int = N_TRIALS, n_min: int = N_MIN_TRADES, seed: int = SEED
) -> optuna.Study:
    """θ_j = argmax Calmar(backtest(ventana | régimen = j, θ)) con TPE."""
    return _tpe_study(lambda p: regime_window_objective(df, seg, labels, regime, p, n_min), n_trials, seed)


def run_regime_oos(df, seg, labels, fold: Fold, thetas: dict, close_on_change: bool = True):
    """Evalúa en la semana de prueba los θ congelados de cada régimen (thetas: régimen -> params).
    Los indicadores arrancan `warm` barras antes de la prueba y no se abre nada antes de ella."""
    parsed = {r: to_params(p) for r, p in thetas.items()}
    warm = max(max(sp.n_donchian, sp.n_roc, sp.n_ema, sp.n_atr) for sp, _ in parsed.values()) + 2
    lo = max(fold.train_start, fold.train_end - warm)
    window, seg_w = df.iloc[lo : fold.test_end], seg.iloc[lo : fold.test_end]
    specs = {}
    for r, (sp, bp) in parsed.items():
        signal, atr_series = strategy_signal(window, seg_w, sp)
        specs[r] = RegimeSpec(signal, atr_series, bp)
    start = fold.train_end - lo
    res = backtest_regimes(window, seg_w, labels.iloc[lo : fold.test_end], specs, start, close_on_change)
    return res, start


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def _chain_folds(df: pd.DataFrame, folds: list[Fold], fold_fn, verbose: bool = False):
    """Encadena por retornos las semanas fuera de muestra que produce fold_fn(i, fold), que
    devuelve (trayectoria normalizada de la semana, operaciones, costos, nocional, fila de la tabla)."""
    level, bench_level = INITIAL_CASH, INITIAL_CASH
    rows, equity_parts, bench_parts, trades_parts = [], [], [], []
    costs = notional = 0.0
    t0 = time.time()
    for i, f in enumerate(folds):
        path, week_trades, week_costs, week_notional, row = fold_fn(i, f)
        test = df.iloc[f.train_end : f.test_end]
        equity_parts.append(level * path)
        level = equity_parts[-1].iloc[-1]
        bench = test["Close"] / test["Open"].iloc[0]
        bench_parts.append(bench_level * bench)
        bench_level = bench_parts[-1].iloc[-1]
        if len(week_trades):
            week_trades = week_trades.copy()
            week_trades["fold"] = i
            trades_parts.append(week_trades)
        costs += week_costs
        notional += week_notional
        rows.append({"fold": i, "test_start": test.index[0], "oos_return": path.iloc[-1] - 1,
                     "oos_trades": len(week_trades), **row})
        if verbose and (i + 1) % 10 == 0:
            print(f"  ventana {i + 1} de {len(folds)}  ({time.time() - t0:.0f} s)")
    oos = BacktestResult(
        equity=pd.concat(equity_parts).rename("equity"),
        trades=pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame(),
        total_costs=costs,
        traded_notional=notional,
    )
    benchmark = BacktestResult(pd.concat(bench_parts).rename("equity"), pd.DataFrame(), 0.0, 0.0)
    return pd.DataFrame(rows), oos, benchmark


def walk_forward(
    df: pd.DataFrame,
    folds: list[Fold],
    n_trials: int = N_TRIALS,
    n_min: int = N_MIN_TRADES,
    gate: float | None = None,
    verbose: bool = False,
):
    """Optimiza en cada ventana de entrenamiento y evalúa con los parámetros congelados
    en la semana siguiente. Las semanas fuera de muestra se encadenan por retornos.

    Si gate no es None, la semana solo se opera cuando el mejor Calmar de entrenamiento
    supera gate; si no, se queda en efectivo (el optimizador no encontró ventaja ni
    dentro de muestra). Con gate=None todas las semanas se operan.

    Devuelve (tabla por fold, resultado OOS encadenado, benchmark comprar y mantener)."""

    def fold_fn(i, f):
        study = optimize_window(df.iloc[f.train_start : f.train_end], n_trials, n_min)
        p = study.best_params
        traded = gate is None or study.best_value > gate
        test_index = df.index[f.train_end : f.test_end]
        row = {"is_calmar": study.best_value, "traded": traded,
               **{f"p_{k}": v for k, v in p.items()}}
        if not traded:
            return pd.Series(1.0, index=test_index), pd.DataFrame(), 0.0, 0.0, row
        res, start = run_oos(df, f, p)
        path = res.equity.iloc[start:] / INITIAL_CASH
        return path, res.trades, res.total_costs, res.traded_notional, row

    return _chain_folds(df, folds, fold_fn, verbose)


def walk_forward_regimes(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    seg: pd.Series,
    folds: list[Fold],
    n_trials: int = N_TRIALS,
    n_min: int = N_MIN_TRADES,
    gate: float = 0.0,
    close_on_change: bool = True,
    verbose: bool = False,
):
    """Walk-forward con un θ por régimen.

    En cada fold los umbrales del régimen (reglas) se ajustan solo con las variables hasta el
    final de la ventana de entrenamiento y quedan fijos para la semana de prueba. Cada régimen
    con al menos B_MIN_BARS barras en la ventana optimiza su propio θ solo sobre las barras
    en que rige; si su mejor Calmar no supera `gate`, no se opera en ese régimen esa semana.
    Una posición se cierra al cambiar el régimen (si close_on_change).

    Devuelve (tabla por fold, resultado OOS encadenado, benchmark comprar y mantener)."""

    def fold_fn(i, f):
        rules = RuleRegimes.fit(feats.iloc[: f.train_end])
        labels = hold_labels(rules.classify(feats.iloc[: f.test_end]), seg.iloc[: f.test_end])
        train_df = df.iloc[f.train_start : f.train_end]
        train_seg = seg.iloc[f.train_start : f.train_end]
        train_lab = labels.iloc[f.train_start : f.train_end]
        thetas, row = {}, {}
        for code, name in NAMES.items():
            bars = int((train_lab == code).sum())
            row[f"bars_{name}"] = bars
            row[f"is_{name}"] = np.nan
            if bars < B_MIN_BARS:
                continue
            study = optimize_regime_window(
                train_df, train_seg, train_lab, code, n_trials, regime_n_min(bars, len(train_df), n_min)
            )
            row[f"is_{name}"] = study.best_value
            if study.best_value > gate:
                thetas[code] = study.best_params
        row["regimes_traded"] = len(thetas)
        test_index = df.index[f.train_end : f.test_end]
        if not thetas:
            return pd.Series(1.0, index=test_index), pd.DataFrame(), 0.0, 0.0, row
        res, start = run_regime_oos(df, seg, labels, f, thetas, close_on_change)
        path = res.equity.iloc[start:] / INITIAL_CASH
        return path, res.trades, res.total_costs, res.traded_notional, row

    return _chain_folds(df, folds, fold_fn, verbose)


# ---------------------------------------------------------------------------
# Congelar θ antes de la evaluación final
# ---------------------------------------------------------------------------
def freeze_theta(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    seg: pd.Series,
    n_trials: int = N_TRIALS,
    n_min: int = FINAL_N_MIN,
    gate: float = 0.0,
) -> dict:
    """Optimiza una sola vez sobre todo el train el θ único, los umbrales de régimen y el θ de
    cada régimen. Devuelve un diccionario serializable. Un θ cuyo mejor Calmar no supera
    `gate` se guarda como None: esa estrategia no opera (el optimizador no encontró ventaja
    ni dentro de muestra). Esto se hace y se guarda antes de abrir el archivo de test."""
    single = optimize_window(df, n_trials, n_min)
    rules = RuleRegimes.fit(feats)
    labels = hold_labels(rules.classify(feats), seg)
    regimes = {}
    for code, name in NAMES.items():
        bars = int((labels == code).sum())
        entry = {"bars": bars, "calmar": None, "params": None}
        if bars >= B_MIN_BARS:
            study = optimize_regime_window(
                df, seg, labels, code, n_trials, regime_n_min(bars, len(df), n_min)
            )
            entry["calmar"] = float(study.best_value)
            entry["params"] = study.best_params if study.best_value > gate else None
        regimes[name] = entry
    return {
        "meta": {
            "n_trials": n_trials,
            "n_min_trades": n_min,
            "gate": gate,
            "seed": SEED,
            "train_start": str(df.index[0]),
            "train_end": str(df.index[-1]),
            "train_bars": len(df),
        },
        "single": {
            "calmar": float(single.best_value),
            "params": single.best_params if single.best_value > gate else None,
        },
        "rules": {"vol_crisis": rules.vol_crisis, "r2_trend": rules.r2_trend},
        "regimes": regimes,
    }



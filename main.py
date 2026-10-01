"""Punto de entrada: `python main.py` corre todo el proyecto y deja tablas en docs/tables y figuras
en docs/figures. `python main.py --quick` usa pocas pruebas por ventana para comprobar que todo
corre; sus resultados no son los del reporte y no tocan docs/theta_frozen.json."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import plots
from src.data import (
    audit_prices,
    check_no_overlap,
    clean_prices,
    drop_overlap,
    load_prices,
    segment_ids,
    validate_prices,
)
from src.metrics import compare_groups, periodic_returns, summary, trade_returns
from src.optimize import (
    N_TRIALS,
    breakeven_fee,
    confirmation_effect,
    cost_curve,
    evaluate_frozen,
    freeze_theta,
    make_folds,
    sensitivity,
    walk_forward,
    walk_forward_regimes,
)
from src.regimes import NAMES, RuleRegimes, hold_labels, regime_features, regime_report

SEED = 42  # semilla de random, numpy y Optuna (src/optimize.py)
QUICK_TRIALS = 8
LAB_FEE_BPS = 12.5  # comisión por lado fijada por el lab
DATA, DOCS = Path("data"), Path("docs")
TABLES, FIGURES = DOCS / "tables", DOCS / "figures"
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f} s] {message}", flush=True)


def save_table(frame: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv")


def _default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def jsonable(obj):
    """Convierte tipos de numpy y pandas a tipos que JSON acepta."""
    return json.loads(json.dumps(obj, default=_default))


def to_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_default))


def metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({name: summary(r) for name, r in results.items()}).T


def save_periodic_returns(results: dict, dataset: str) -> None:
    for name, r in results.items():
        for freq, label in (("M", "mensual"), ("Q", "trimestral"), ("Y", "anual")):
            save_table(periodic_returns(r.equity, freq).to_frame("retorno"), f"retornos_{label}_{dataset}_{name}")


def prepare_data():
    train_raw = load_prices(DATA / "btc_project_train.csv")
    test_raw = load_prices(DATA / "btc_project_test.csv")
    audit = pd.DataFrame({"train": audit_prices(train_raw), "test": audit_prices(test_raw)}).astype(str)
    save_table(audit, "auditoria_datos")
    train = clean_prices(train_raw)
    test = drop_overlap(train, clean_prices(test_raw))
    for d in (train, test):
        validate_prices(d)
    check_no_overlap(train, test)
    return train, test


def train_walk_forward(train, feats, seg, n_trials):
    """Walk-forward semanal sobre train: un solo theta (con gate) y un theta por régimen."""
    folds = make_folds(seg)
    log(f"Walk-forward con un solo theta: {len(folds)} ventanas x {n_trials} pruebas")
    table_single, oos_single, bench = walk_forward(train, folds, n_trials, gate=0.0, verbose=True)
    log("Walk-forward con un theta por régimen")
    table_regimes, oos_regimes, _ = walk_forward_regimes(train, feats, seg, folds, n_trials, verbose=True)
    save_table(table_single, "walk_forward_theta_unico")
    save_table(table_regimes, "walk_forward_theta_por_regimen")
    results = {"theta_unico": oos_single, "theta_por_regimen": oos_regimes, "comprar_y_mantener": bench}
    save_table(metrics_table(results), "metricas_train")
    save_periodic_returns(results, "train")
    return table_single, table_regimes, results


def regime_differentiation(trades: pd.DataFrame) -> dict:
    """Pregunta 5: retorno por operación en cada régimen y prueba entre reversión y tendencia."""
    t = trade_returns(trades)
    t["regimen"] = t["regime"].map(NAMES)
    save_table(t.groupby("regimen")[["bruto_pct", "neto_pct"]].agg(["count", "mean"]), "desempeno_por_regimen_train")
    a = t.loc[t["regimen"] == "reversion", "bruto_pct"]
    b = t.loc[t["regimen"] == "tendencia", "bruto_pct"]
    return compare_groups(a, b) if min(len(a), len(b)) > 1 else {}


def freeze_and_evaluate(train, test, feats, seg, n_trials, quick):
    """Optimiza theta una sola vez sobre train y lo evalúa en test. La evaluación usa el theta
    congelado y commiteado (docs/theta_frozen.json); main.py solo recalcula para verificarlo."""
    log("Recalculando theta sobre todo el train")
    recomputed = freeze_theta(train, feats, seg, n_trials=n_trials)
    to_json(recomputed, DOCS / "theta_recomputed.json")
    frozen_file = DOCS / "theta_frozen.json"
    if frozen_file.exists() and not quick:
        theta = json.loads(frozen_file.read_text())
        same = jsonable(recomputed) == theta
        log("theta recalculado " + ("IGUAL" if same else "DISTINTO") + " al congelado (docs/theta_frozen.json)")
    else:
        theta = jsonable(recomputed)
    log("Evaluando en test con theta congelado")
    results = evaluate_frozen(train, test, theta)
    save_table(metrics_table(results), "metricas_test")
    save_periodic_returns(results, "test")
    return theta, results


def robustness(train, theta):
    """Preguntas 1, 3 y 4, con el theta único congelado sobre train."""
    p = theta["single"]["params"]
    if p is None:
        log("theta único sin parámetros (gate no cumplido): se omite el análisis de robustez")
        return {}
    log("Robustez: 2 de 3 contra un indicador, sensibilidad de ±20% y curva de costos")
    effect = confirmation_effect(train, p)
    sens = sensitivity(train, p)
    curve = cost_curve(train, p)
    equilibrio = breakeven_fee(curve)
    save_table(effect, "pregunta1_dos_de_tres")
    save_table(sens, "pregunta3_sensibilidad")
    save_table(curve, "pregunta4_curva_de_costos")
    plots.plot_sensitivity(sens, FIGURES / "fig4_sensibilidad.png")
    plots.plot_cost_curve(curve, equilibrio, LAB_FEE_BPS, FIGURES / "fig5_costos.png")
    return {"comision_equilibrio_bps_por_lado": equilibrio}


def regime_analysis(train, test, theta):
    """Estabilidad del régimen (train contra test, umbrales congelados) y etiquetas para las figuras."""
    full = pd.concat([train, test])
    seg = segment_ids(full)
    feats = regime_features(full, seg)
    rules = RuleRegimes(theta["rules"]["vol_crisis"], theta["rules"]["r2_trend"])
    labels = hold_labels(rules.classify(feats), seg)
    n = len(train)
    for name, part in (("train", slice(0, n)), ("test", slice(n, None))):
        table, extra = regime_report(feats.iloc[part], labels.iloc[part], seg.iloc[part])
        save_table(table, f"regimenes_{name}")
        save_table(pd.Series(extra, name="valor").to_frame(), f"regimenes_{name}_global")
    return full, feats, labels


def make_figures(train_results, test_results, full, feats, labels):
    log("Generando figuras")
    names = {"theta_unico": "θ único", "theta_por_regimen": "θ por régimen", "comprar_y_mantener": "Comprar y mantener"}
    curves = lambda res: {names[k]: v.equity for k, v in res.items()}
    plots.plot_equity(curves(train_results), curves(test_results), FIGURES / "fig1_valor_portafolio.png")
    plots.plot_drawdown(curves(train_results), curves(test_results), FIGURES / "fig2_drawdown.png")
    for dataset, res in (("entrenamiento", train_results), ("prueba", test_results)):
        plots.plot_returns_table(res["theta_por_regimen"].equity, f"θ por régimen, {dataset}",
                                 FIGURES / f"fig3_retornos_{dataset}.png")
    plots.plot_regime_timeline(full["Close"], labels, FIGURES / "fig6a_regimenes_precio.png")
    plots.plot_feature_distributions(feats, labels, FIGURES / "fig6b_distribuciones.png")
    for dataset, res in (("entrenamiento", train_results), ("prueba", test_results)):
        plots.plot_equity_with_regimes(res["theta_por_regimen"].equity, labels, f"θ por régimen, {dataset}",
                                       FIGURES / f"fig6c_portafolio_regimenes_{dataset}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="pocas pruebas por ventana (solo para verificar)")
    quick = parser.parse_args().quick
    random.seed(SEED)
    np.random.seed(SEED)
    n_trials = QUICK_TRIALS if quick else N_TRIALS

    log("Cargando, limpiando y auditando datos")
    train, test = prepare_data()
    seg = segment_ids(train)
    feats = regime_features(train, seg)

    table_single, table_regimes, train_results = train_walk_forward(train, feats, seg, n_trials)
    summary_numbers = {
        "pruebas_por_ventana": n_trials,
        "ventanas": len(table_single),
        "calmar_entrenamiento_mediano": float(table_single["is_calmar"].median()),
        "calmar_fuera_de_muestra": float(summary(train_results["theta_unico"])["calmar"]),
        "diferenciacion_reversion_vs_tendencia": regime_differentiation(train_results["theta_por_regimen"].trades),
    }
    theta, test_results = freeze_and_evaluate(train, test, feats, seg, n_trials, quick)
    regimes_fitted = table_regimes[["is_reversion", "is_tendencia", "is_crisis"]].notna().sum().sum()
    frozen_fits = 1 + sum(r["calmar"] is not None for r in theta["regimes"].values())
    summary_numbers.update(
        {
            "configuraciones_walk_forward_theta_unico": n_trials * len(table_single),
            "configuraciones_walk_forward_por_regimen": n_trials * int(regimes_fitted),
            "configuraciones_theta_congelado": n_trials * frozen_fits,
        }
    )
    summary_numbers.update(robustness(train, theta))
    full, full_feats, labels = regime_analysis(train, test, theta)
    make_figures(train_results, test_results, full, full_feats, labels)
    summary_numbers["configuraciones_totales"] = sum(v for k, v in summary_numbers.items() if k.startswith("configuraciones_"))
    summary_numbers["tiempo_total_s"] = round(time.time() - START)
    to_json(summary_numbers, TABLES / "resumen.json")
    log("Listo. Tablas en docs/tables y figuras en docs/figures")


if __name__ == "__main__":
    main()

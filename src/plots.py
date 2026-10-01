"""Figuras del reporte. Todas tienen título, ejes etiquetados y leyenda, y se guardan en disco."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src.metrics import drawdown_curve, periodic_returns
from src.regimes import NAMES, UNLABELED

REGIME_COLORS = {"reversion": "#4c78a8", "tendencia": "#54a24b", "crisis": "#e45756"}
GAP = pd.Timedelta("6h")  # mismo umbral que abre un tramo nuevo en src/data.py


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_dates(ax, money: bool = False) -> None:
    """Fechas legibles en el eje x y, si money, miles separados por comas en el eje y."""
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if money:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _break_gaps(s: pd.Series) -> pd.Series:
    """Inserta un NaN en cada hueco de datos para que la línea no una tramos distintos."""
    jumps = s.index.to_series().diff() > GAP
    if not jumps.any():
        return s
    breaks = pd.Series(np.nan, index=s.index[jumps] - pd.Timedelta("1ns"))
    return pd.concat([s, breaks]).sort_index()


def _regime_legend() -> list:
    return [Patch(facecolor=c, alpha=0.4, label=n) for n, c in REGIME_COLORS.items()]


def _shade_regimes(ax, labels: pd.Series) -> None:
    """Colorea el fondo según el régimen vigente."""
    run = (labels != labels.shift()).cumsum()
    for _, part in labels.groupby(run):
        code = part.iloc[0]
        if code != UNLABELED:
            ax.axvspan(part.index[0], part.index[-1], color=REGIME_COLORS[NAMES[code]], alpha=0.25, lw=0)


def plot_equity(train_curves: dict, test_curves: dict, path) -> None:
    """Figura 1: valor del portafolio en entrenamiento y en prueba, con su benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    titles = ("Entrenamiento (semanas fuera de muestra del walk-forward)", "Prueba (parámetros congelados)")
    for ax, curves, title in zip(axes, (train_curves, test_curves), titles):
        for name, s in curves.items():
            s = _break_gaps(s)
            ax.plot(s.index, s.to_numpy(), label=name, lw=1.4)
        ax.set_title(title)
        ax.set_xlabel("Fecha (UTC)")
        ax.set_ylabel("Valor del portafolio (USD)")
        ax.legend()
        _format_dates(ax, money=True)
    fig.suptitle("Figura 1. Valor del portafolio a lo largo del tiempo")
    _save(fig, path)


def plot_drawdown(train_curves: dict, test_curves: dict, path) -> None:
    """Figura 2: curva de drawdown de cada serie de la figura 1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, curves, title in zip(axes, (train_curves, test_curves), ("Entrenamiento", "Prueba")):
        for name, s in curves.items():
            d = _break_gaps(drawdown_curve(s) * 100)
            ax.plot(d.index, d.to_numpy(), label=name, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("Fecha (UTC)")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        _format_dates(ax)
    fig.suptitle("Figura 2. Curva de drawdown")
    _save(fig, path)


def plot_returns_table(equity: pd.Series, title: str, path) -> None:
    """Figura 3: retornos mensuales, trimestrales y anuales (%) de una serie de equity."""
    def pivot(freq, column):
        r = periodic_returns(equity, freq) * 100
        frame = r.to_frame("r")
        frame["year"] = r.index.year
        frame["col"] = getattr(r.index, column)
        return frame.pivot(index="year", columns="col", values="r")

    tables = [("Mensual (%)", pivot("M", "month")), ("Trimestral (%)", pivot("Q", "quarter"))]
    annual = periodic_returns(equity, "Y") * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [4, 2, 1.2]})
    for ax, (name, t) in zip(axes[:2], tables):
        lim = max(1.0, np.nanmax(np.abs(t.to_numpy())))
        ax.imshow(t.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(t.columns)), labels=t.columns)
        ax.set_yticks(range(len(t.index)), labels=t.index)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("Mes" if "Mensual" in name else "Trimestre")
        ax.set_ylabel("Año")
    bars = axes[2].bar(annual.index.year.astype(str), annual.to_numpy(), color="#4c78a8", label="Retorno anual")
    axes[2].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Anual (%)")
    axes[2].set_xlabel("Año")
    axes[2].set_ylabel("Retorno (%)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Figura 3. Retornos periódicos: {title}")
    _save(fig, path)


def plot_sensitivity(table: pd.DataFrame, path) -> None:
    """Figura 4: Calmar al variar cada parámetro óptimo en -20% y +20% (línea: valor base)."""
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, table["calmar_-20%"], width=0.4, label="Parámetro -20%", color="#f58518")
    ax.bar(x + 0.2, table["calmar_+20%"], width=0.4, label="Parámetro +20%", color="#4c78a8")
    ax.axhline(table["base_calmar"].iloc[0], color="black", ls="--", label="Calmar con el parámetro óptimo")
    ax.set_xticks(x, labels=table.index, rotation=30)
    ax.set_xlabel("Parámetro variado")
    ax.set_ylabel("Calmar")
    ax.set_title("Figura 4. Sensibilidad de los parámetros óptimos ante variaciones de ±20%")
    ax.legend()
    _save(fig, path)


def plot_cost_curve(curve: pd.DataFrame, breakeven_bps, lab_bps: float, path) -> None:
    """Figura 5: retorno total contra comisión por lado, con la comisión del lab y la de equilibrio."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curve.index, curve["retorno_total"] * 100, marker="o", label="Retorno total")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(lab_bps, color="#e45756", ls="--", label=f"Comisión del lab ({lab_bps:g} pb por lado)")
    if breakeven_bps is not None:
        ax.axvline(breakeven_bps, color="#54a24b", ls="--", label=f"Equilibrio ({breakeven_bps:.1f} pb por lado)")
    ax.set_xlabel("Comisión por lado (puntos base)")
    ax.set_ylabel("Retorno total (%)")
    ax.set_title("Figura 5. Retorno neto contra nivel de costo de transacción")
    ax.legend()
    _save(fig, path)


def plot_regime_timeline(price: pd.Series, labels: pd.Series, path) -> None:
    """Figura 6a: línea de tiempo de regímenes sobre el precio."""
    fig, ax = plt.subplots(figsize=(14, 4.8))
    p = _break_gaps(price)
    _shade_regimes(ax, labels.reindex(price.index).fillna(UNLABELED).astype(int))
    ax.plot(p.index, p.to_numpy(), color="black", lw=0.6, label="Precio de cierre")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Precio de cierre (USD)")
    ax.set_title("Figura 6a. Línea de tiempo de regímenes sobre el precio")
    _format_dates(ax, money=True)
    ax.legend(handles=[*_regime_legend(), plt.Line2D([], [], color="black", lw=0.8, label="Precio de cierre")])
    _save(fig, path)


def plot_feature_distributions(feats: pd.DataFrame, labels: pd.Series, path) -> None:
    """Figura 6b: distribución de cada variable de régimen dentro de cada régimen."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    descriptions = {"vol": "Volatilidad (desv. est. del retorno)", "trend_r2": "Fuerza de tendencia (R²)",
                    "autocorr_1h": "Autocorrelación horaria"}
    for ax, col in zip(axes, descriptions):
        data = [feats.loc[labels == code, col].dropna().to_numpy() for code in NAMES]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, name in zip(box["boxes"], NAMES.values()):
            patch.set_facecolor(REGIME_COLORS[name])
        ax.set_xticks(range(1, len(NAMES) + 1), labels=list(NAMES.values()))
        ax.set_xlabel("Régimen")
        ax.set_ylabel(descriptions[col])
        ax.set_title(col)
    axes[0].legend(handles=[Patch(facecolor=c, label=n) for n, c in REGIME_COLORS.items()], fontsize=8)
    fig.suptitle("Figura 6b. Distribución de las variables por régimen")
    _save(fig, path)


def plot_equity_with_regimes(equity: pd.Series, labels: pd.Series, title: str, path) -> None:
    """Figura 6c: valor del portafolio con los regímenes superpuestos."""
    fig, ax = plt.subplots(figsize=(14, 4.8))
    e = _break_gaps(equity)
    _shade_regimes(ax, labels.reindex(equity.index).fillna(UNLABELED).astype(int))
    ax.plot(e.index, e.to_numpy(), color="black", lw=1.2)
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Valor del portafolio (USD)")
    ax.set_title(f"Figura 6c. Valor del portafolio con regímenes superpuestos: {title}")
    _format_dates(ax, money=True)
    ax.legend(handles=[*_regime_legend(), plt.Line2D([], [], color="black", lw=1.2, label="Valor del portafolio")])
    _save(fig, path)

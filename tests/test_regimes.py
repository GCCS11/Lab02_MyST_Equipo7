"""Pruebas de las variables de régimen, las etiquetas y su actualización."""
import numpy as np
import pandas as pd

from src.data import segment_ids
from src.regimes import (
    CRISIS,
    REVERSION,
    TREND,
    HMMRegimes,
    KMeansRegimes,
    RuleRegimes,
    hold_labels,
    regime_features,
    run_lengths,
)
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


def test_hold_labels_updates_every_n_bars_per_segment():
    labels = pd.Series([2, 1, 0, 0, 1, 1, 2, 2], index=range(8))
    seg = pd.Series([0, 0, 0, 0, 0, 1, 1, 1], index=range(8))
    held = hold_labels(labels, seg, every=3)
    # tramo 0: se actualiza en las posiciones 0 y 3; tramo 1: en la posición 0
    assert held.tolist() == [2, 2, 2, 0, 0, 1, 1, 1]


def test_run_lengths_do_not_cross_segments():
    labels = pd.Series([1, 1, 1, 1, 0, 0])
    seg = pd.Series([0, 0, 1, 1, 1, 1])
    runs = run_lengths(labels, seg)
    assert runs["label"].tolist() == [1, 1, 0]
    assert runs["bars"].tolist() == [2, 2, 2]


def test_labels_are_causal():
    """Con umbrales fijos, la etiqueta en t no cambia al agregar datos posteriores a t."""
    df = make_prices()
    feats_full = regime_features(df, segment_ids(df), WINDOW)
    rules = RuleRegimes.fit(feats_full.iloc[:600])  # umbrales ajustados solo con el pasado
    full = hold_labels(rules.classify(feats_full), segment_ids(df))
    assert (full != -1).sum() > 500, "la prueba sería vacía sin etiquetas"
    for t in (400, 900, 1100, 1500, 1800):
        cut = df.iloc[: t + 1]
        seg_cut = segment_ids(cut)
        part = hold_labels(rules.classify(regime_features(cut, seg_cut, WINDOW)), seg_cut)
        assert part.iloc[-1] == full.iloc[t], f"etiqueta distinta en t={t}"


def test_kmeans_names_clusters_by_centroids():
    """Sin importar cómo K-means numere los clusters, los nombres salen de los centroides."""
    rng = np.random.default_rng(0)

    def blob(vol, r2, n=300):
        return pd.DataFrame({"vol": rng.normal(vol, vol * 0.05, n), "trend_r2": rng.normal(r2, 0.03, n)})

    feats = pd.concat([blob(0.0010, 0.80), blob(0.0010, 0.10), blob(0.0030, 0.50)], ignore_index=True)
    model = KMeansRegimes.fit(feats)
    labels = model.classify(feats)
    assert (labels.iloc[:300] == TREND).all()
    assert (labels.iloc[300:600] == REVERSION).all()
    assert (labels.iloc[600:] == CRISIS).all()


def test_kmeans_labels_are_causal():
    """Con centroides fijos ajustados en el pasado, la etiqueta en t no depende del futuro."""
    df = make_prices()
    feats_full = regime_features(df, segment_ids(df), WINDOW)
    model = KMeansRegimes.fit(feats_full.iloc[:600], cols=("vol", "trend_r2", "autocorr_1h"))
    full = hold_labels(model.classify(feats_full), segment_ids(df))
    assert (full != -1).sum() > 500, "la prueba sería vacía sin etiquetas"
    for t in (400, 900, 1100, 1500, 1800):
        cut = df.iloc[: t + 1]
        seg_cut = segment_ids(cut)
        part = hold_labels(model.classify(regime_features(cut, seg_cut, WINDOW)), seg_cut)
        assert part.iloc[-1] == full.iloc[t], f"etiqueta distinta en t={t}"


def regime_prices(n=14000, seed=0):
    """Precios sintéticos con bloques alternados de calma, tendencia y alta volatilidad."""
    rng = np.random.default_rng(seed)
    block = np.arange(n) // 1400
    vol = np.where(block % 3 == 2, 0.003, 0.0008)
    drift = np.where(block % 3 == 1, 0.0004, 0.0)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol + drift))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"Close": close}, index=idx)


def test_hmm_filtered_labels_are_causal():
    """Con parámetros fijos, la etiqueta filtrada en t no cambia al agregar datos posteriores."""
    df = regime_prices()
    seg = segment_ids(df)
    feats = regime_features(df, seg, 300)
    model = HMMRegimes.fit(feats, seg)
    full = model.classify(feats, seg)
    assert (full != -1).sum() > 5000, "la prueba sería vacía sin etiquetas"
    for t in (2500, 3900, 5300, 6700, 8100, 9500, 10900, 12300):
        cut = df.iloc[: t + 1]
        seg_cut = segment_ids(cut)
        part = model.classify(regime_features(cut, seg_cut, 300), seg_cut)
        assert part.iloc[-1] == full.iloc[t], f"etiqueta distinta en t={t}"


def test_hmm_filter_matches_hmmlearn_at_the_last_observation():
    """En el último instante, la probabilidad filtrada coincide con la posterior de hmmlearn."""
    df = regime_prices()
    seg = segment_ids(df)
    feats = regime_features(df, seg, 300)
    model = HMMRegimes.fit(feats, seg)
    cut_feats = feats.iloc[:9001]
    probs = model.filtered_probs(cut_feats, seg.iloc[:9001])
    z = (cut_feats.loc[probs.index, list(model.cols)].to_numpy() - model.mean) / model.std
    reference = model._model().predict_proba(z)[-1]
    np.testing.assert_allclose(probs.iloc[-1].to_numpy(), reference, atol=1e-6)

    
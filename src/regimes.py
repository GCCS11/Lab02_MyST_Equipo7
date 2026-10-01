"""Detección de régimen de mercado: variables sobre ventana móvil de 1 semana."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

WEEK = 7 * 288  # 1 semana en barras de 5 minutos
HOUR = 12  # 1 hora en barras de 5 minutos


def regime_features(df: pd.DataFrame, seg: pd.Series, window: int = WEEK, lag: int = HOUR) -> pd.DataFrame:
    """Tres variables de régimen, causales y calculadas dentro de cada tramo continuo:

    vol          volatilidad realizada: desviación estándar de los retornos logarítmicos.
    trend_r2     fuerza de tendencia: R^2 de la regresión lineal del log-precio contra el
                 tiempo (cercano a 1 si el precio avanza en línea recta).
    autocorr_1h  fuerza de reversión: autocorrelación entre el retorno de la última hora y
                 el de la hora anterior (negativa si los movimientos tienden a revertirse).
                 Se usa retorno horario y no de 5 minutos porque en estos datos la
                 autocorrelación de 5 minutos es positiva solo en el rezago 1, un efecto de
                 cómo se construyen las barras y no de reversión o tendencia.

    Son NaN hasta completar la primera ventana de cada tramo."""
    parts = []
    for _, d in df.groupby(seg, sort=False):
        logp = np.log(d["Close"])
        ret = logp.diff()
        ret_h = logp.diff(lag)
        t = pd.Series(np.arange(len(d), dtype=float), index=d.index)
        parts.append(
            pd.DataFrame(
                {
                    "vol": ret.rolling(window).std(),
                    "trend_r2": logp.rolling(window).corr(t) ** 2,
                    "autocorr_1h": ret_h.rolling(window).corr(ret_h.shift(lag)),
                }
            )
        )
    return pd.concat(parts)


# ---------------------------------------------------------------------------
# Clasificadores, actualización de etiquetas y validación
# ---------------------------------------------------------------------------
REVERSION, TREND, CRISIS, UNLABELED = 0, 1, 2, -1
NAMES = {REVERSION: "reversion", TREND: "tendencia", CRISIS: "crisis"}
UPDATE_EVERY = 12  # la etiqueta se actualiza cada hora (12 barras de 5 min)
BAR_HOURS = 5 / 60


def _codes_from_centers(raw: np.ndarray, cols) -> np.ndarray:
    """Nombres por centroide (en unidades originales): el de mayor volatilidad es crisis;
    de los otros dos, el de mayor trend_r2 es tendencia y el de menor es reversión (rango)."""
    crisis = int(np.argmax(raw[:, cols.index("vol")]))
    rest = sorted((c for c in range(len(raw)) if c != crisis), key=lambda c: raw[c, cols.index("trend_r2")])
    codes = np.empty(len(raw), dtype=int)
    codes[crisis], codes[rest[0]], codes[rest[1]] = CRISIS, REVERSION, TREND
    return codes


@dataclass(frozen=True)
class RuleRegimes:
    """Régimen por reglas con umbrales fijos, ajustados una sola vez en train:

        crisis     si vol >= vol_crisis (cuantil alto de la volatilidad de train)
        tendencia  si no es crisis y trend_r2 >= r2_trend
        reversion  en otro caso (comportamiento de rango)

    Solo usa vol y trend_r2; autocorr_1h se evalúa aparte en la comparación de variables."""

    vol_crisis: float
    r2_trend: float = 0.5

    @classmethod
    def fit(cls, feats: pd.DataFrame, q_crisis: float = 0.90, r2_trend: float = 0.5) -> "RuleRegimes":
        return cls(float(feats["vol"].quantile(q_crisis)), r2_trend)

    def classify(self, feats: pd.DataFrame, seg: pd.Series | None = None) -> pd.Series:
        label = np.where(
            feats["vol"] >= self.vol_crisis,
            CRISIS,
            np.where(feats["trend_r2"] >= self.r2_trend, TREND, REVERSION),
        )
        missing = feats[["vol", "trend_r2"]].isna().any(axis=1)
        return pd.Series(np.where(missing, UNLABELED, label), index=feats.index)


@dataclass(frozen=True)
class KMeansRegimes:
    """Régimen por K-means con K=3, ajustado una sola vez en train sobre las variables
    estandarizadas con la media y desviación de train. Después cada barra se asigna al
    centroide más cercano (centroides fijos, así que la etiqueta es causal).

    Como K-means numera los clusters al azar, los nombres salen de los centroides:
    el de mayor volatilidad es crisis; de los otros dos, el de mayor trend_r2 es
    tendencia y el de menor es reversión (rango)."""

    cols: tuple
    mean: np.ndarray
    std: np.ndarray
    centers: np.ndarray  # estandarizados, un renglón por cluster
    codes: np.ndarray  # régimen asignado a cada cluster

    @classmethod
    def fit(cls, feats: pd.DataFrame, cols=("vol", "trend_r2"), seed: int = 42) -> "KMeansRegimes":
        cols = tuple(cols)
        x = feats[list(cols)].dropna()
        mean, std = x.mean().to_numpy(), x.std().to_numpy()
        km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit((x.to_numpy() - mean) / std)
        raw = km.cluster_centers_ * std + mean
        return cls(cols, mean, std, km.cluster_centers_, _codes_from_centers(raw, cols))

    def classify(self, feats: pd.DataFrame, seg: pd.Series | None = None) -> pd.Series:
        x = feats[list(self.cols)]
        ok = x.notna().all(axis=1).to_numpy()
        z = (x.to_numpy()[ok] - self.mean) / self.std
        nearest = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        out = np.full(len(feats), UNLABELED)
        out[ok] = self.codes[nearest]
        return pd.Series(out, index=feats.index)


def _hourly_sample(feats: pd.DataFrame, seg: pd.Series, cols, every: int = UPDATE_EVERY):
    """Observaciones horarias (posición múltiplo de `every` dentro de cada tramo) con las
    variables completas. Devuelve (variables muestreadas, tramo de cada una)."""
    pos = pd.Series(0, index=feats.index).groupby(seg).cumcount()
    sel = (pos % every == 0) & feats[list(cols)].notna().all(axis=1)
    return feats.loc[sel, list(cols)], seg[sel]


@dataclass(frozen=True)
class HMMRegimes:
    """HMM gaussiano de 3 estados sobre las variables estandarizadas, muestreadas cada hora.

    Se ajusta una sola vez en train (el mejor de varios reinicios de EM). La etiqueta de cada
    hora es el estado más probable dado SOLO lo observado hasta esa hora (probabilidades
    filtradas, recursión hacia adelante), nunca la ruta de Viterbi, que reetiqueta el pasado
    usando el futuro. Los estados se nombran por sus medias, igual que en K-means."""

    cols: tuple
    mean: np.ndarray
    std: np.ndarray
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray  # estandarizadas
    covars: np.ndarray  # estandarizadas, matriz completa por estado
    codes: np.ndarray

    @classmethod
    def fit(cls, feats: pd.DataFrame, seg: pd.Series, cols=("vol", "trend_r2"), seed: int = 42,
            n_init: int = 5) -> "HMMRegimes":
        cols = tuple(cols)
        x, seg_x = _hourly_sample(feats, seg, cols)
        mean, std = x.mean().to_numpy(), x.std().to_numpy()
        z = (x.to_numpy() - mean) / std
        lengths = seg_x.groupby(seg_x, sort=False).size().tolist()
        best_score, best = -np.inf, None
        for i in range(n_init):
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=seed + i)
            model.fit(z, lengths)
            score = model.score(z, lengths)
            if score > best_score:
                best_score, best = score, model
        raw = best.means_ * std + mean
        return cls(cols, mean, std, best.startprob_, best.transmat_, best.means_, best.covars_,
                   _codes_from_centers(raw, cols))

    @property
    def expected_duration_h(self) -> np.ndarray:
        """Duración esperada de cada estado en horas: 1 / (1 - A_jj), un paso es una hora."""
        return 1.0 / (1.0 - np.diag(self.transmat))

    @property
    def prior(self) -> np.ndarray:
        """Distribución inicial de cada tramo: la estacionaria de la matriz de transición
        (no sabemos en qué régimen arranca un tramo, y el startprob estimado en train
        refleja solo cómo empezó la primera serie)."""
        vals, vecs = np.linalg.eig(self.transmat.T)
        pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return pi / pi.sum()

    def _z(self, x: pd.DataFrame) -> np.ndarray:
        return (x.to_numpy() - self.mean) / self.std

    def _model(self) -> GaussianHMM:
        """Modelo de hmmlearn con los parámetros ajustados (para Viterbi y para verificar el filtro)."""
        model = GaussianHMM(n_components=3, covariance_type="full")
        model.startprob_, model.transmat_ = self.prior, self.transmat
        model.means_, model.covars_ = self.means, self.covars
        return model

    def filtered_probs(self, feats: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
        """P(estado_t | observaciones hasta t) en cada hora, con el filtro reiniciado en cada tramo."""
        x, seg_x = _hourly_sample(feats, seg, self.cols)
        log_a = np.log(np.clip(self.transmat, 1e-300, None))
        log_prior = np.log(np.clip(self.prior, 1e-300, None))
        parts = []
        for s in pd.unique(seg_x):
            part = x[(seg_x == s).to_numpy()]
            z = self._z(part)
            log_b = np.column_stack(
                [multivariate_normal.logpdf(z, mean=self.means[j], cov=self.covars[j]) for j in range(3)]
            )
            alpha = np.empty_like(log_b)
            a = log_prior + log_b[0]
            alpha[0] = a - logsumexp(a)
            for t in range(1, len(z)):
                a = logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
                alpha[t] = a - logsumexp(a)
            parts.append(pd.DataFrame(np.exp(alpha), index=part.index))
        return pd.concat(parts)

    def classify(self, feats: pd.DataFrame, seg: pd.Series) -> pd.Series:
        """Etiqueta filtrada de la última actualización horaria, mantenida entre actualizaciones."""
        probs = self.filtered_probs(feats, seg)
        hourly = pd.Series(self.codes[probs.to_numpy().argmax(axis=1)], index=probs.index)
        out = hourly.reindex(feats.index).groupby(seg).ffill()
        return out.fillna(UNLABELED).astype(int)

    def viterbi_labels(self, feats: pd.DataFrame, seg: pd.Series) -> pd.Series:
        """Etiquetas con Viterbi sobre toda la secuencia de cada tramo. NO son causales (usan
        el futuro para reetiquetar el pasado): solo sirven para compararlas con las filtradas."""
        x, seg_x = _hourly_sample(feats, seg, self.cols)
        model = self._model()
        parts = []
        for s in pd.unique(seg_x):
            part = x[(seg_x == s).to_numpy()]
            parts.append(pd.Series(self.codes[model.predict(self._z(part))], index=part.index))
        hourly = pd.concat(parts)
        out = hourly.reindex(feats.index).groupby(seg).ffill()
        return out.fillna(UNLABELED).astype(int)


def hold_labels(labels: pd.Series, seg: pd.Series, every: int = UPDATE_EVERY) -> pd.Series:
    """Actualiza la etiqueta cada `every` barras (contadas desde el inicio de cada tramo)
    y la mantiene constante entre actualizaciones."""
    pos = labels.groupby(seg).cumcount()
    held = labels.where(pos % every == 0).groupby(seg).ffill()
    return held.fillna(UNLABELED).astype(int)


def run_lengths(labels: pd.Series, seg: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de una misma etiqueta (sin cruzar tramos): etiqueta y duración en barras."""
    new_run = (labels != labels.shift()) | (seg != seg.shift())
    runs = labels.groupby(new_run.cumsum()).agg(["first", "size"])
    return runs.rename(columns={"first": "label", "size": "bars"})


def regime_report(feats: pd.DataFrame, labels: pd.Series, seg: pd.Series, cols=("vol", "trend_r2"),
                  sample: int = 10_000, seed: int = 42):
    """Validación del régimen: participación y duración por régimen, transiciones por mes
    y silhouette (sobre las variables estandarizadas, con submuestra por costo)."""
    valid = labels != UNLABELED
    runs = run_lengths(labels, seg)
    runs = runs[runs["label"] != UNLABELED]
    rows = {}
    for code, name in NAMES.items():
        r = runs[runs["label"] == code]["bars"] * BAR_HOURS
        rows[name] = {
            "participacion_%": 100 * (labels[valid] == code).mean(),
            "duracion_media_h": r.mean(),
            "duracion_mediana_h": r.median(),
            "rachas": len(r),
        }
    prev = labels.shift()
    change = (labels != prev) & (seg == seg.shift()) & valid & (prev != UNLABELED)
    months = valid.sum() / (288 * 30)
    x = feats.loc[valid, list(cols)]
    x = (x - x.mean()) / x.std()
    sil = silhouette_score(x, labels[valid], sample_size=min(sample, len(x)), random_state=seed)
    return pd.DataFrame(rows).T, {"transiciones_por_mes": change.sum() / months, "silhouette": sil}



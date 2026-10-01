"""Carga, validación, limpieza y auditoría de datos de BTCUSDT a 5 minutos."""
from pathlib import Path

import pandas as pd

BAR = pd.Timedelta("5min")
BREAK_GAP = pd.Timedelta("6h")  # un hueco mayor abre un tramo nuevo
OHLC = ["Open", "High", "Low", "Close"]


def load_prices(path: str | Path) -> pd.DataFrame:
    """Carga un CSV de barras de 5 min indexado por tiempo UTC (sin limpiar)."""
    df = pd.read_csv(path)
    if (df["Gmtoffset"] != 0).any():
        raise ValueError("Se esperaba Gmtoffset = 0 (UTC)")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df["Timestamp"], unit="s", utc=True), name="time"
    )
    return df[OHLC + ["Volume"]]


def _on_grid(df: pd.DataFrame):
    """Máscara de barras alineadas a la rejilla de 5 minutos."""
    return (df.index.minute % 5 == 0) & (df.index.second == 0)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Quita barras fuera de la rejilla y barras vacías. No rellena nada."""
    return df[_on_grid(df)].dropna(subset=OHLC)


def drop_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Quita del test las barras que ya están en el train, verificando que sean iguales."""
    shared = test.index.intersection(train.index)
    if len(shared) and not train.loc[shared, OHLC].equals(test.loc[shared, OHLC]):
        raise ValueError("Las barras compartidas entre train y test difieren")
    return test[test.index > train.index[-1]]


def segment_ids(df: pd.DataFrame, max_gap: pd.Timedelta = BREAK_GAP) -> pd.Series:
    """Etiqueta tramos continuos: un hueco mayor a max_gap abre un tramo nuevo."""
    new_segment = df.index.to_series().diff() > max_gap
    return new_segment.cumsum().rename("segment")


def validate_prices(df: pd.DataFrame) -> None:
    """Falla si los datos limpios tienen problemas que invalidarían el backtest."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("El índice de tiempo no está ordenado")
    if df.index.has_duplicates:
        raise ValueError("Hay timestamps duplicados")
    if not _on_grid(df).all():
        raise ValueError("Hay timestamps fuera de la rejilla de 5 min")
    if df[OHLC].isna().any().any():
        raise ValueError("Hay NaN en precios OHLC")
    if (df[OHLC] <= 0).any().any():
        raise ValueError("Hay precios no positivos")


def check_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Verifica que el test empiece después de que termine el train."""
    if train.index[-1] >= test.index[0]:
        raise ValueError("Train y test se traslapan")


def audit_prices(raw: pd.DataFrame) -> dict:
    """Mide la calidad del CSV crudo y lo que la limpieza va a quitar."""
    off = raw[~_on_grid(raw)]
    clean = clean_prices(raw)
    gaps = clean.index.to_series().diff().dropna()
    bad_ohlc = (clean["High"] < clean[["Open", "Close", "Low"]].max(axis=1)) | (
        clean["Low"] > clean[["Open", "Close", "High"]].min(axis=1)
    )
    return {
        "n_raw_rows": len(raw),
        "n_nan_ohlc": int(raw[OHLC].isna().any(axis=1).sum()),
        "n_off_grid": len(off),
        "n_off_grid_flat_no_volume": int(
            ((off["High"] == off["Low"]) & off["Volume"].isna()).sum()
        ),
        "n_clean_bars": len(clean),
        "start": clean.index[0],
        "end": clean.index[-1],
        "n_short_gaps": int(((gaps > BAR) & (gaps <= BREAK_GAP)).sum()),
        "n_breaks": int((gaps > BREAK_GAP).sum()),
        "max_gap": gaps.max(),
        "n_segments": int(segment_ids(clean).nunique()),
        "n_inconsistent_ohlc": int(bad_ohlc.sum()),
        "n_flat_bars": int((clean["High"] == clean["Low"]).sum()),
    }
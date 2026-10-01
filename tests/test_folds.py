"""Pruebas de las ventanas del walk-forward."""
import numpy as np
import pandas as pd

from src.optimize import make_folds


def make_seg(sizes):
    return pd.Series(np.repeat(np.arange(len(sizes)), sizes))


def test_folds_never_cross_a_segment_cut():
    seg = make_seg([500, 90, 300])
    folds = make_folds(seg, train_bars=100, test_bars=20, step_bars=20)
    assert folds, "debería haber ventanas"
    for f in folds:
        assert seg.iloc[f.train_start] == seg.iloc[f.test_end - 1]


def test_folds_test_follows_train_and_tests_do_not_overlap():
    seg = make_seg([1000])
    folds = make_folds(seg, train_bars=100, test_bars=20, step_bars=20)
    for f in folds:
        assert f.train_end - f.train_start == 100
        assert f.test_end - f.train_end == 20
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train_end >= prev.test_end  # las semanas de test no se solapan
        assert nxt.train_end - prev.train_end == 20


def test_short_segments_produce_no_folds():
    seg = make_seg([100, 5000])
    folds = make_folds(seg, train_bars=200, test_bars=50, step_bars=50)
    assert all(f.train_start >= 100 for f in folds)
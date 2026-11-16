import numpy as np

from physiodyn.data.beats import AAMI_CLASS_MAP, _class_from_symbol
from physiodyn.data.rr import compute_rr_series, split_records


def test_class_mapping():
    assert _class_from_symbol("N") == 0
    assert _class_from_symbol("S") == 1
    assert _class_from_symbol("V") == 2
    assert _class_from_symbol("F") == 3
    assert _class_from_symbol("Q") == 4
    assert set(AAMI_CLASS_MAP[0]) == {"N", "L", "R", "e", "j"}


def test_rr_in_plausible_range():
    rr = compute_rr_series(np.array([0, 180, 360, 540, 900], dtype=int), fs=360)
    assert rr.shape[0] == 4
    assert np.all((rr >= 0.2) & (rr <= 3.0))


def test_split_records_do_not_overlap():
    records = [f"rec_{i}" for i in range(10)]
    train, val, test = split_records(records, seed=7)
    all_splits = train + val + test
    assert len(all_splits) == len(records)
    assert len(set(all_splits)) == len(records)


def test_output_shapes():
    signal = np.zeros(1000, dtype=np.float64)
    peaks = np.array([120, 220, 420, 700], dtype=int)
    beats, labels = __import__("physiodyn.data.beats", fromlist=["extract_beats"]).extract_beats(signal, peaks, fs=250, target_length=200, annotation_symbols=["N", "N", "V", "F"])
    assert beats.shape == (4, 200)
    assert labels.shape == (4,)

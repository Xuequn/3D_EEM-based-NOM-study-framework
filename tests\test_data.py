import numpy as np
import pandas as pd

from eem_rc.data import align_eem, source_from_filename


def test_align_eem_keeps_unmeasured_cells_as_nan(tmp_path):
    frame = pd.DataFrame([[0, 200, 205], [200, 1.0, 2.0], [205, 3.0, 4.0]])
    path = tmp_path / "PPHA_example.xlsx"
    frame.to_excel(path, header=False, index=False)
    aligned = align_eem(path)
    assert aligned.shape == (81, 81)
    assert np.isclose(aligned[0, 0], 1.0)
    assert np.isclose(aligned[1, 1], 4.0)
    assert np.isnan(aligned[-1, -1])


def test_source_from_filename():
    assert source_from_filename("PPHA_001.xlsx") == "PPHA"
    assert source_from_filename("esha_001.xlsx") == "ESHA"


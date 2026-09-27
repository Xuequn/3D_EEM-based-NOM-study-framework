import numpy as np

from eem_rc.twodcos import generalized_2dcos


def test_generalized_2dcos_shapes_and_sync_symmetry():
    spectra = np.arange(30, dtype=float).reshape(5, 6)
    synchronous, asynchronous = generalized_2dcos(spectra)
    assert synchronous.shape == (6, 6)
    assert asynchronous.shape == (6, 6)
    assert np.allclose(synchronous, synchronous.T)
    assert np.allclose(asynchronous, -asynchronous.T)


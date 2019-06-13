import pytest
import numpy as np

from grigora.keras.preprocessing import sequence


def test_skipgrams():
    # test with no window size and binary labels
    couples, labels = sequence.skipgrams(np.arange(3), vocabulary_size=3)
    for couple in couples:
        assert couple[0] in [0, 1, 2] and couple[1] in [0, 1, 2]

    # test window size
    couples, labels = sequence.skipgrams(np.arange(5),
                                         vocabulary_size=5,
                                         window_size=1)
    for couple in couples:
        assert couple[0] - couple[1] <= 3


if __name__ == '__main__':
    pytest.main([__file__])

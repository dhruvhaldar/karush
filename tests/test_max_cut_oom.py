import numpy as np
import pytest
from karush.convex.relaxations import max_cut_sdp_relaxation

def test_max_cut_oom():
    # If n=1000, 1000^3 = 1 billion float64 = 8GB.
    # W is 1000x1000 = 1 million = 8MB (very small input)
    np.ones((1000, 1000))
    # Let's try n=1500 -> 3.375 billion = 27GB
    # It should ideally throw ValueError before trying to allocate A_stack
    with pytest.raises(ValueError):
        max_cut_sdp_relaxation(np.ones((1500, 1500)))

if __name__ == "__main__":
    test_max_cut_oom()

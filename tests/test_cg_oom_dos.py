import pytest

from karush.unconstrained.conjugate_gradient import conjugate_gradient


def test_cg_late_validation_dos():
    class MassiveList(list):
        def __len__(self): return 50000
    x0 = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        conjugate_gradient(lambda x: 0, lambda x: x, x0)

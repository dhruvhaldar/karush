import pytest
from karush.semidefinite.interior_point import solve_sdp_barrier
from karush.convex.relaxations import max_cut_sdp_relaxation
from karush.constrained.barrier import barrier_method
from karush.unconstrained.newton import newton_method
from karush.unconstrained.quasi_newton import bfgs_method
from karush.constrained.sqp import sqp_equality_constrained

def test_late_validation_interior_point():
    class MassiveList(list):
        def __len__(self): return 50000
    C = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        solve_sdp_barrier(C, [], [], MassiveList())

def test_late_validation_max_cut():
    class MassiveList(list):
        def __len__(self): return 50000
    W = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        max_cut_sdp_relaxation(W)

def test_late_validation_barrier():
    class MassiveList(list):
        def __len__(self): return 50000
    x0 = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        barrier_method(lambda x: 0, lambda x: x, lambda x: x, lambda x: x, lambda x: x, x0)

def test_late_validation_newton():
    class MassiveList(list):
        def __len__(self): return 50000
    x0 = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        newton_method(lambda x: 0, lambda x: x, lambda x: x, x0)

def test_late_validation_bfgs():
    class MassiveList(list):
        def __len__(self): return 50000
    x0 = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        bfgs_method(lambda x: 0, lambda x: x, x0)

def test_late_validation_sqp():
    class MassiveList(list):
        def __len__(self): return 50000
    x0 = MassiveList()
    with pytest.raises(ValueError, match="System dimensions exceed"):
        sqp_equality_constrained(lambda x: 0, lambda x: x, lambda x: x, lambda x: x, lambda x: x, x0)

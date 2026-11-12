import importlib.util
import io
import pickle

import numpy as np
import pytest

from hand_model.mano import (_MANOUnpickler, fix_left_data, left_shapedirs_mismatch, mano_dir, mirror_axis_angle,
                             needs_left_fix)

HAVE_MANO = all((mano_dir() / f"MANO_{s}.pkl").is_file() for s in ("LEFT", "RIGHT"))
HAVE_SMPLX = importlib.util.find_spec("smplx") is not None and importlib.util.find_spec("torch") is not None


def _shapedirs(seed):
    return np.random.default_rng(seed).normal(size=(778, 3, 10))


def test_detects_and_fixes_v12_left_bug():
    right = {"shapedirs": _shapedirs(0)}
    buggy_left = {"shapedirs": right["shapedirs"].copy()}           # v1.2: same sign as right
    assert needs_left_fix(buggy_left, right)
    fixed = fix_left_data(buggy_left)
    assert not needs_left_fix(fixed, right)                          # idempotent: not flipped back
    assert left_shapedirs_mismatch(fixed, right)[1] == 0.0
    np.testing.assert_array_equal(fixed["shapedirs"][:, 1:], buggy_left["shapedirs"][:, 1:])
    assert buggy_left["shapedirs"] is not fixed["shapedirs"]          # input left untouched


def test_mirror_axis_angle():
    pose = np.arange(6.0)
    np.testing.assert_array_equal(mirror_axis_angle(pose), [0, -1, -2, 3, -4, -5])


def test_unpickler_refuses_unknown_classes():
    payload = pickle.dumps({"x": io.BytesIO()}, protocol=2)
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        _MANOUnpickler(io.BytesIO(payload), encoding="latin1").load()


@pytest.mark.skipif(not (HAVE_MANO and HAVE_SMPLX), reason="needs model/mano/*.pkl and smplx")
def test_fixed_left_hand_mirrors_right():
    from hand_model.cli import mirror_error
    from hand_model.mano import create_mano

    right, left = create_mano("right"), create_mano("left")
    assert left.left_shapedirs_fixed
    assert mirror_error(left, right, samples=8, beta_std=2.0, pose_std=0.5) < 1e-4

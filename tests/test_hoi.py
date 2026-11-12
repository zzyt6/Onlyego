"""Static checks of the model wrappers kept in src/HOI (no heavy dependencies are imported).

Every file must parse, and every `HOI.*` import must point at a module that exists in the tree, so nothing we
still call depends on a removed V2D module. The entry points our parts start in WSL must exist.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
HOI = SRC / "HOI"
FILES = sorted(HOI.rglob("*.py"))
FILES = [f for f in FILES if "FoundationPose" not in f.parts]  # NVIDIA's code, kept as is
ENTRY_POINTS = ["HOI.wilor.video_to_hands", "HOI.grounding_dino.image_to_object_bboxes", "HOI.sam2.video_to_masks",
                "HOI.sam3d.image_to_mesh", "HOI.foundation_pose.foundation_pose_tracker"]


def _module_exists(name: str) -> bool:
    p = SRC.joinpath(*name.split("."))
    return (p / "__init__.py").is_file() or p.with_suffix(".py").is_file()


def test_entry_points_and_licence():
    for m in ENTRY_POINTS:
        assert _module_exists(m), m
    assert (HOI / "foundation_pose" / "FoundationPose" / "LICENSE").is_file()  # NVIDIA source code license kept


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(HOI)))
def test_parses_and_imports_resolve(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module.startswith("HOI") and not _module_exists(node.module):
                if not all(_module_exists(f"{node.module}.{a.name}") for a in node.names):
                    missing.append(node.module)
        elif isinstance(node, ast.Import):
            missing += [a.name for a in node.names if a.name.startswith("HOI") and not _module_exists(a.name)]
    assert not missing, f"unresolved HOI imports: {missing}"


def test_our_wsl_calls_name_existing_modules():
    for f in ["object_model/segment.py", "object_model/model.py", "object_pose/fp_track.py"]:
        text = (SRC / f).read_text(encoding="utf-8")
        for name in {w.strip("\"'(),") for w in text.split() if w.strip("\"'(),").startswith("HOI.")}:
            mod = name.split(" ")[0]
            assert _module_exists(mod) or _module_exists(mod.rsplit(".", 1)[0]), (f, mod)

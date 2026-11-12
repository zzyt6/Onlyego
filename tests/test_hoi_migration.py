"""Static checks of the V2D hand + object code copied into src/HOI (no heavy dependencies are imported).

Every file must parse, and every `HOI.*` import must point at a module that exists in the tree, so the
rename from V2D's `v2d.<module>.lib` layout is complete. References to V2D modules that live in our other
parts (MoGe, AnyCalib, GeoCalib, DROID-SLAM) are only allowed in the pipeline script, which is not wired yet.
"""

import ast
import json
from pathlib import Path

import pytest

HOI = Path(__file__).resolve().parents[1] / "src" / "HOI"
FILES = sorted(HOI.rglob("*.py"))


def _module_exists(name: str) -> bool:
    parts = name.split(".")[1:]  # drop "HOI"
    p = HOI.joinpath(*parts) if parts else HOI
    return p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()


def test_tree_was_migrated():
    assert len(FILES) > 200
    for m in ("wilor", "hamer", "hawor", "hand_alignment", "sam2", "grounding_dino", "sam3d", "foundation_pose",
              "gsplat_refinement", "mesh", "mediapipe", "common", "pipelines"):
        assert (HOI / m / "__init__.py").is_file(), m
    assert (HOI / "foundation_pose" / "FoundationPose" / "LICENSE").is_file()  # NVIDIA source code license kept


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(HOI)))
def test_parses_and_hoi_imports_resolve(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            if node.module.startswith("HOI") and not _module_exists(node.module):
                missing.append(node.module)
            continue
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        missing += [n for n in names if n.startswith("HOI") and not _module_exists(n)]
    assert not missing, f"unresolved HOI imports: {missing}"


def test_only_pipeline_refers_to_unmigrated_v2d_modules():
    refs = json.loads((HOI / "MIGRATION.json").read_text(encoding="utf-8"))["external_refs"]
    for mod, files in refs.items():
        if mod == "v2d.mv":  # multi-view code was left out; only a docstring mentions it
            assert files == ["common/video.py (docstring only)"]
            continue
        assert all(f.startswith("pipelines") for f in files), (mod, files)
    assert not list(HOI.rglob("mv_*.py")) and not list(HOI.rglob("run_mv_*.py"))

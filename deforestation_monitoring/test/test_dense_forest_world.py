#!/usr/bin/env python3
"""Static checks of the dense forest world and the local models.

These tests do not need a running Gazebo. They check that the worlds only
use local model materials that exist, and that dense_forest has the
expected grass ground tiles and oak and pine trees.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLDS_DIR = PROJECT_ROOT / "41068_ignition_bringup" / "worlds"
MODELS_DIR = PROJECT_ROOT / "41068_ignition_bringup" / "models"


def _includes(world_path):
    tree = ET.parse(world_path)
    return [
        elem
        for elem in tree.getroot().iter("include")
    ]


def _uri(elem):
    uri = elem.findtext("uri", default="")
    return uri.strip()


def _resolve_local_model_uri(uri):
    if not uri.startswith("model://"):
        return None
    model_name = uri[len("model://"):].strip()
    return MODELS_DIR / model_name


def _referenced_texture_paths(model_sdf):
    tree = ET.parse(model_sdf)
    return [
        Path(text.strip())
        for text in tree.getroot().itertext()
        if text.strip().startswith("model://")
        and "/materials/textures/" in text
    ]


def test_dense_forest_world_structure():
    dense = WORLDS_DIR / "dense_forest.sdf"
    assert dense.exists()

    root = ET.parse(dense).getroot()
    world = root.find("world")
    assert world is not None and world.get("name") == "dense_forest"

    includes = _includes(dense)
    grass_tiles = [e for e in includes if _uri(e) == "model://grass_plane"]
    oak_trees = [e for e in includes if "OpenRobotics/models/Oak%20tree" in _uri(e)]
    pine_trees = [e for e in includes if "OpenRobotics/models/Pine%20tree" in _uri(e)]

    assert len(grass_tiles) == 25
    assert len(oak_trees) + len(pine_trees) == 214


def test_all_world_local_model_materials_resolve():
    """Every local model:// URI in every world must point to a model whose
    model.sdf and texture files exist."""
    world_paths = sorted(WORLDS_DIR.glob("*.sdf"))
    assert world_paths, "no worlds found"

    checked_models = set()
    for world_path in world_paths:
        for elem in _includes(world_path):
            uri = _uri(elem)
            model_dir = _resolve_local_model_uri(uri)
            if model_dir is None:
                continue

            model_sdf = model_dir / "model.sdf"
            model_config = model_dir / "model.config"
            assert model_sdf.exists(), f"missing model.sdf for {uri}"
            assert model_config.exists(), f"missing model.config for {uri}"

            for texture_path in _referenced_texture_paths(model_sdf):
                relative = texture_path.relative_to(f"model://{model_dir.name}/")
                resolved = model_dir / relative
                assert resolved.exists(), f"missing material texture: {resolved}"
                checked_models.add(model_dir.name)

    # The project has three local material models and the worlds use all
    # three: grass_plane, forest_plane and forest_wall.
    assert {"grass_plane", "forest_plane", "forest_wall"} <= checked_models


def test_showcase_forest_every_tree_visible_from_above():
    """Showcase world: oaks at least 7 m apart, pines at least 7 m from every
    oak and 3.5 m from each other (see make_showcase_world.py)."""
    import math
    text = (WORLDS_DIR / "showcase_forest.sdf").read_text()
    root = ET.fromstring(text)
    assert root.find("world").get("name") == "showcase_forest"
    trees = [(e.findtext("name"), *map(float, e.findtext("pose").split()[:2]))
             for e in root.iter("include") if "tree" in _uri(e).lower()]
    oaks = [t for t in trees if t[0].startswith("oak")]
    pines = [t for t in trees if t[0].startswith("pine")]
    d = lambda a, b: math.hypot(a[1] - b[1], a[2] - b[2])  # noqa: E731
    assert min(d(a, b) for a in oaks for b in oaks if a is not b) >= 7.0 - 1e-6
    assert min(d(p, o) for p in pines for o in oaks) >= 7.0 - 1e-6
    assert min(d(a, b) for a in pines for b in pines if a is not b) >= 3.5 - 1e-6
    assert len({t[0] for t in trees}) == len(trees)

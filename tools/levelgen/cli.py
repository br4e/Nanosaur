"""CLI entrypoint orchestrating terrain generation, serialization, and validation."""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter

from .constants import (
    BUNDLE,
    ENEMY_TYPES,
    OUT_IMAGES,
    OUT_TERRAIN,
    ORIG_DIR,
    PREVIEW_PATH,
    SEED,
    ZONE_NAMES,
)
from .heightmap import build_heightmaps
from .io import write_map_tga, write_preview, write_ter
from .items import build_items
from .original import Original
from .textures import paint_textures
from .validate import validate_ter, validate_tga
from .world import build_world


def main():
    rng = random.Random(SEED)
    orig = Original()

    print("building world ...")
    world = build_world(rng)

    print("synthesizing exact-corner heightmap tiles ...")
    hm_grid, path_layer, tile_bank = build_heightmaps(world, orig)

    print("painting biome textures ...")
    tex = paint_textures(world, hm_grid, orig, rng)

    print("placing items ...")
    P, bfs, _start, _bridge_points = build_items(rng, world, hm_grid, path_layer)

    items_sorted = sorted(P.items, key=lambda it: (it[0], it[1]))
    normal_items = [it for it in items_sorted if not it[4]]
    pro_items = items_sorted

    OUT_TERRAIN.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    write_ter(OUT_TERRAIN / "Level1.ter", tex, hm_grid, path_layer, normal_items, tile_bank, orig)
    write_ter(OUT_TERRAIN / "Level1Pro.ter", tex, hm_grid, path_layer, pro_items, tile_bank, orig)
    shutil.copyfile(ORIG_DIR / "Level1.trt", OUT_TERRAIN / "Level1.trt")
    write_map_tga(OUT_IMAGES / "Map.tga", tex, orig)
    write_preview(PREVIEW_PATH, tex, world["tmean"], P.items, orig)

    print("\n=== VALIDATION REPORT ===")
    all_ok = True
    for label, f in (
        ("Level1.ter", OUT_TERRAIN / "Level1.ter"),
        ("Level1Pro.ter", OUT_TERRAIN / "Level1Pro.ter"),
    ):
        checks, st, items = validate_ter(f, orig, label)
        print(f"\n--- {label} ---")
        for name, ok, detail in checks:
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(
            f"  stats: floor {st['floor_pct']:.1f}%  cliff255 {st['cliff_pct']:.1f}%  "
            f"items {st['items']}  hm tiles {st['hm_tiles']}  BFS cells {st['bfs_cells']}"
        )
        tc = Counter(it[2] for it in items)
        print(f"  item counts by type: {dict(sorted(tc.items()))}")

    ok9, det9 = validate_tga(OUT_IMAGES / "Map.tga")
    all_ok &= ok9
    print(f"\n  [{'PASS' if ok9 else 'FAIL'}] 9 Map.tga format: {det9}")

    zone = world["zone"]
    print("\n  zone breakdown (BFS tiles & normal enemies):")
    for zi, zname in enumerate(ZONE_NAMES):
        ztiles = sum(1 for r, c in bfs if zone[r, c] == zi)
        zen = sum(
            1 for x, z, t, _p, pro in P.items
            if not pro and t in ENEMY_TYPES and zone[z // 32, x // 32] == zi
        )
        print(
            f"    {zname:22s}: {ztiles:4d} BFS tiles, {zen:3d} enemies "
            f"({1000 * zen / max(1, ztiles):.1f}/1k)"
        )

    if not all_ok:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")

    if BUNDLE.is_dir():
        (BUNDLE / "Terrain").mkdir(parents=True, exist_ok=True)
        (BUNDLE / "Images").mkdir(parents=True, exist_ok=True)
        for f in ("Level1.ter", "Level1Pro.ter", "Level1.trt"):
            shutil.copyfile(OUT_TERRAIN / f, BUNDLE / "Terrain" / f)
        shutil.copyfile(OUT_IMAGES / "Map.tga", BUNDLE / "Images" / "Map.tga")
        print(f"installed into {BUNDLE}")

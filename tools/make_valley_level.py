#!/usr/bin/env python3
"""
Build a Nanosaur 1 playfield by cropping a real region out of original Level1.

This keeps authentic grass tile sequencing, cliff face textures (with flip bits),
heightmap tiles, and path solids — instead of inventing a fake layout.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Data" / "Terrain"
ORIG_DIR = ROOT / "tools" / "original_terrain"

# Crop origin/size in original Level1 tile coords (must be multiples of 5 for size)
# Tuned around the player start (tile 34,107) to include nearby cliff ledges.
CROP_COL = 14
CROP_ROW = 87
MAP_W = 40
MAP_D = 40
TILE_PX = 32
SUPERTILE = 5

ITEM_START = 0
ITEM_POWERUP = 1
ITEM_REX = 3
ITEM_EGG = 5
ITEM_PORTAL = 9
ITEM_TREE = 10
ITEM_BUSH = 13

POW_HEATSEEK = 0
POW_LASER = 1
POW_HEALTH = 3
POW_SHIELD = 4


def pack_item(x: int, y: int, typ: int, parm=(0, 0, 0, 0), flags=0) -> bytes:
    return struct.pack(
        ">hhh4bHii",
        x,
        y,
        typ,
        parm[0],
        parm[1],
        parm[2],
        parm[3],
        flags,
        0,
        0,
    )


def tile_center(col: int, row: int) -> tuple[int, int]:
    return col * TILE_PX + TILE_PX // 2, row * TILE_PX + TILE_PX // 2


def build_items(floor_cells: list[tuple[int, int]]) -> list[bytes]:
    """Place gameplay items on interior floor cells, sorted by X."""
    # Prefer cells away from the rim
    interior = [
        (c, r)
        for c, r in floor_cells
        if 6 <= c < MAP_W - 6 and 6 <= r < MAP_D - 6
    ]
    if len(interior) < 20:
        interior = floor_cells

    def pick(frac_x: float, frac_z: float) -> tuple[int, int]:
        tx = int(frac_x * (MAP_W - 1))
        tz = int(frac_z * (MAP_D - 1))
        return min(interior, key=lambda p: (p[0] - tx) ** 2 + (p[1] - tz) ** 2)

    raw: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    def add(col: int, row: int, typ: int, parm=(0, 0, 0, 0)) -> None:
        x, y = tile_center(col, row)
        raw.append((x, y, typ, parm))

    sc, sr = pick(0.50, 0.50)
    add(sc, sr, ITEM_START)
    add(*pick(0.50, 0.35), ITEM_PORTAL, (0, 0, 0, 0))

    for fx, fz, species in [
        (0.35, 0.45, 0),
        (0.65, 0.40, 1),
        (0.40, 0.65, 2),
        (0.60, 0.70, 3),
        (0.30, 0.60, 4),
    ]:
        add(*pick(fx, fz), ITEM_EGG, (species, 0, 0, 1))

    for fx, fz in [(0.30, 0.30), (0.70, 0.35), (0.55, 0.70), (0.35, 0.75)]:
        add(*pick(fx, fz), ITEM_REX, (0, 0, 0, 1))

    for fx, fz, kind in [
        (0.55, 0.45, POW_HEALTH),
        (0.42, 0.55, POW_LASER),
        (0.62, 0.58, POW_SHIELD),
        (0.33, 0.48, POW_HEATSEEK),
        (0.52, 0.62, POW_HEALTH),
    ]:
        add(*pick(fx, fz), ITEM_POWERUP, (kind, 0, 0, 0))

    for fx, fz, tree_type in [
        (0.22, 0.40, 4),
        (0.20, 0.55, 0),
        (0.25, 0.70, 5),
        (0.78, 0.40, 4),
        (0.80, 0.55, 1),
        (0.75, 0.70, 5),
        (0.40, 0.22, 4),
        (0.60, 0.20, 0),
        (0.40, 0.80, 5),
        (0.60, 0.78, 4),
        (0.25, 0.25, 2),
        (0.75, 0.75, 2),
    ]:
        add(*pick(fx, fz), ITEM_TREE, (tree_type, 0, 0, 0))

    for fx, fz in [
        (0.38, 0.32),
        (0.62, 0.30),
        (0.32, 0.62),
        (0.68, 0.65),
        (0.50, 0.72),
        (0.28, 0.50),
    ]:
        add(*pick(fx, fz), ITEM_BUSH, (0, 0, 0, 0))

    raw.sort(key=lambda it: (it[0], it[1]))
    return [pack_item(x, y, typ, parm) for x, y, typ, parm in raw]


def write_trt(path: Path) -> None:
    shutil.copyfile(ORIG_DIR / "Level1.trt", path)
    n = struct.unpack_from(">i", path.read_bytes(), 0)[0]
    print(f"wrote {path} (original tileset, {n} tiles)")


def write_ter(path: Path) -> None:
    assert MAP_W % SUPERTILE == 0 and MAP_D % SUPERTILE == 0
    orig = (ORIG_DIR / "Level1.ter").read_bytes()
    offs = struct.unpack_from(">7i2h2i", orig, 0)
    ow, od = offs[7], offs[8]
    assert CROP_COL + MAP_W <= ow and CROP_ROW + MAP_D <= od

    tex_all = struct.unpack_from(f">{ow * od}H", orig, offs[0])
    hm_all = struct.unpack_from(f">{ow * od}H", orig, offs[1])
    path_all = struct.unpack_from(f">{ow * od}H", orig, offs[2])
    hmt_off = offs[5]
    tex_attrs = orig[offs[9] : offs[10]]

    tex_layer: list[int] = []
    hm_layer: list[int] = []
    path_layer: list[int] = []
    hm_tiles: list[bytes] = []
    hm_remap: dict[int, int] = {}
    floor_cells: list[tuple[int, int]] = []

    for row in range(MAP_D):
        for col in range(MAP_W):
            src = (CROP_ROW + row) * ow + (CROP_COL + col)
            tval = tex_all[src]  # keep flip/rot bits
            hval = hm_all[src]
            pval = path_all[src]
            hid = hval & 0x0FFF
            flags = hval & 0xF000

            if hid not in hm_remap:
                assert len(hm_tiles) < 300
                hm_remap[hid] = len(hm_tiles)
                blob = orig[hmt_off + hid * 1024 : hmt_off + (hid + 1) * 1024]
                assert len(blob) == 1024
                hm_tiles.append(blob)

            tex_layer.append(tval)
            hm_layer.append(hm_remap[hid] | flags)
            path_layer.append(pval)

            # Floor = average height under ~120 (original flat floor tile is 85)
            mean_h = sum(hm_tiles[hm_remap[hid]]) / 1024.0
            if mean_h < 120 and pval == 0:
                floor_cells.append((col, row))

    items = build_items(floor_cells)
    n_cells = MAP_W * MAP_D

    header_size = 40
    tex_off = header_size
    hm_off = tex_off + n_cells * 2
    path_off = hm_off + n_cells * 2
    obj_off = path_off + n_cells * 2
    obj_blob = struct.pack(">i", len(items)) + b"".join(items)
    hmtiles_off = obj_off + len(obj_blob)
    hmtiles_blob = b"".join(hm_tiles)
    texattr_off = hmtiles_off + len(hmtiles_blob)
    tilanim_off = texattr_off + len(tex_attrs)

    header = struct.pack(
        ">7i2h2i",
        tex_off,
        hm_off,
        path_off,
        obj_off,
        0,
        hmtiles_off,
        0,
        MAP_W,
        MAP_D,
        texattr_off,
        tilanim_off,
    )
    body = b"".join(
        [
            struct.pack(f">{n_cells}H", *tex_layer),
            struct.pack(f">{n_cells}H", *hm_layer),
            struct.pack(f">{n_cells}H", *path_layer),
            obj_blob,
            hmtiles_blob,
            tex_attrs,
        ]
    )
    path.write_bytes(header + body)
    print(
        f"wrote {path} ({len(header)+len(body)} bytes) "
        f"crop=({CROP_COL},{CROP_ROW}) {MAP_W}x{MAP_D} "
        f"hm_tiles={len(hm_tiles)} floor_cells={len(floor_cells)} items={len(items)}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_trt(OUT_DIR / "Level1.trt")
    write_ter(OUT_DIR / "Level1.ter")
    write_ter(OUT_DIR / "Level1Pro.ter")


if __name__ == "__main__":
    main()

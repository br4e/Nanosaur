#!/usr/bin/env python3
"""
Generate a Nanosaur 1 playfield: grassy clearing with T-Rexes,
powerups, eggs, and a time portal.

Uses original Level1.trt grass/cliff tiles. Cliff rim uses high
heightmap values + PATH_TILE_SOLID_ALL so you can't jump out.
"""

from __future__ import annotations

import math
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Data" / "Terrain"
ORIG_DIR = ROOT / "tools" / "original_terrain"

# Multiples of SUPERTILE_SIZE (5)
MAP_W = 40
MAP_D = 40
TILE_PX = 32
SUPERTILE = 5
MAX_HM_TILES = 300
CLIFF_DEPTH = 3  # outer tiles that are impassable cliffs

# Original Level1 texture indices (grassy floor near start + cliff faces)
TEX_GRASS = (
    17,
    36,
    20,
    1,
    38,
    39,
    2,
    21,
    40,
    19,
    97,
    96,
    83,
    98,
    45,
    62,
    18,
)
TEX_CLIFF = (189, 200, 201, 190)

PATH_SOLID_ALL = 9  # PATH_TILE_SOLID_ALL

FLOOR_H = 85  # matches original flat height tile near start
CLIFF_H = 255  # matches original high plateau / cliff tops

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


def load_original_tex_attrs() -> bytes:
    data = (ORIG_DIR / "Level1.ter").read_bytes()
    offs = struct.unpack_from(">7i2h2i", data, 0)
    ta, tilanim = offs[9], offs[10]
    return data[ta:tilanim]


def write_trt(path: Path) -> None:
    shutil.copyfile(ORIG_DIR / "Level1.trt", path)
    n = struct.unpack_from(">i", path.read_bytes(), 0)[0]
    print(f"wrote {path} (copied original tileset, {n} tiles)")


def rim_t(col: float, row: float) -> float:
    """0 at center/interior, 1 at outer edge."""
    # Distance to nearest map edge, in tiles
    dist = min(col, row, MAP_W - 1 - col, MAP_D - 1 - row)
    if dist >= CLIFF_DEPTH + 1.5:
        return 0.0
    # 0 just inside cliff band, 1 at outer edge
    return max(0.0, min(1.0, 1.0 - dist / (CLIFF_DEPTH + 1.5)))


def field_height(col: float, row: float) -> int:
    t = rim_t(col, row)
    if t <= 0:
        # Gentle interior rolls
        roll = 4 * math.sin(col * 0.2) * math.cos(row * 0.18)
        return int(max(0, min(255, round(FLOOR_H + roll))))
    # Rise sharply into cliff tops
    h = FLOOR_H + t * t * (CLIFF_H - FLOOR_H)
    return int(max(0, min(255, round(h))))


def qh(h: int) -> int:
    return int(round(h / 8.0) * 8)


def make_height_tile(h00: int, h10: int, h01: int, h11: int) -> bytes:
    out = bytearray(TILE_PX * TILE_PX)
    for y in range(TILE_PX):
        v = y / (TILE_PX - 1)
        for x in range(TILE_PX):
            u = x / (TILE_PX - 1)
            h = (
                h00 * (1 - u) * (1 - v)
                + h10 * u * (1 - v)
                + h01 * (1 - u) * v
                + h11 * u * v
            )
            out[y * TILE_PX + x] = max(0, min(255, int(round(h))))
    return bytes(out)


def pick_texture(col: int, row: int) -> int:
    t = rim_t(col + 0.5, row + 0.5)
    if t > 0.35:
        return TEX_CLIFF[(col * 3 + row * 5) % len(TEX_CLIFF)]
    # Mix many original grass tiles with a stable hash so it looks natural, not striped
    h = (col * 73856093) ^ (row * 19349663)
    return TEX_GRASS[h % len(TEX_GRASS)]


def pick_path(col: int, row: int) -> int:
    # Solid collision on the outer cliff band
    dist = min(col, row, MAP_W - 1 - col, MAP_D - 1 - row)
    if dist < CLIFF_DEPTH:
        return PATH_SOLID_ALL
    return 0


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


def build_items() -> list[bytes]:
    cx, cz = MAP_W // 2, MAP_D // 2
    raw: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    def add(col: int, row: int, typ: int, parm=(0, 0, 0, 0)) -> None:
        x, y = tile_center(col, row)
        raw.append((x, y, typ, parm))

    add(cx, cz, ITEM_START)
    add(cx, cz - 5, ITEM_PORTAL, (0, 0, 0, 0))

    for col, row, species in [
        (cx - 6, cz - 2, 0),
        (cx + 7, cz - 3, 1),
        (cx - 4, cz + 6, 2),
        (cx + 5, cz + 7, 3),
        (cx - 8, cz + 3, 4),
    ]:
        add(col, row, ITEM_EGG, (species, 0, 0, 1))

    for col, row in [
        (cx - 8, cz - 6),
        (cx + 8, cz - 5),
        (cx + 2, cz + 8),
        (cx - 6, cz + 9),
    ]:
        add(col, row, ITEM_REX, (0, 0, 0, 1))

    for col, row, kind in [
        (cx + 3, cz - 2, POW_HEALTH),
        (cx - 3, cz + 2, POW_LASER),
        (cx + 6, cz + 4, POW_SHIELD),
        (cx - 7, cz - 1, POW_HEATSEEK),
        (cx + 1, cz + 5, POW_HEALTH),
    ]:
        add(col, row, ITEM_POWERUP, (kind, 0, 0, 0))

    # Trees just inside the cliff rim
    margin = CLIFF_DEPTH + 2
    for col, row, tree_type in [
        (margin + 1, cz - 4, 4),
        (margin + 1, cz + 2, 0),
        (margin + 2, cz + 6, 5),
        (MAP_W - margin - 2, cz - 3, 4),
        (MAP_W - margin - 2, cz + 1, 1),
        (MAP_W - margin - 3, cz + 6, 5),
        (cx - 3, margin + 1, 4),
        (cx + 3, margin + 1, 0),
        (cx - 2, MAP_D - margin - 2, 5),
        (cx + 4, MAP_D - margin - 2, 4),
        (margin + 3, margin + 3, 2),
        (MAP_W - margin - 4, MAP_D - margin - 4, 2),
    ]:
        add(col, row, ITEM_TREE, (tree_type, 0, 0, 0))

    for col, row in [
        (cx - 5, cz - 8),
        (cx + 5, cz - 7),
        (cx - 8, cz + 5),
        (cx + 8, cz + 6),
        (cx, cz + 10),
        (cx - 10, cz),
    ]:
        add(col, row, ITEM_BUSH, (0, 0, 0, 0))

    raw.sort(key=lambda it: (it[0], it[1]))
    return [pack_item(x, y, typ, parm) for x, y, typ, parm in raw]


def write_ter(path: Path) -> None:
    assert MAP_W % SUPERTILE == 0 and MAP_D % SUPERTILE == 0
    n_cells = MAP_W * MAP_D
    tex_attrs = load_original_tex_attrs()
    num_tex = len(tex_attrs) // 8

    verts = [
        [qh(field_height(c, r)) for c in range(MAP_W + 1)]
        for r in range(MAP_D + 1)
    ]

    tex_layer = []
    hm_layer = []
    path_layer = []
    hm_tiles: list[bytes] = []
    hm_cache: dict[tuple[int, int, int, int], int] = {}

    for row in range(MAP_D):
        for col in range(MAP_W):
            tid = pick_texture(col, row) & 0x0FFF
            assert tid < num_tex
            tex_layer.append(tid)

            key = (
                verts[row][col],
                verts[row][col + 1],
                verts[row + 1][col],
                verts[row + 1][col + 1],
            )
            if key not in hm_cache:
                assert len(hm_tiles) < MAX_HM_TILES, "heightmap tile budget exceeded"
                hm_cache[key] = len(hm_tiles)
                hm_tiles.append(make_height_tile(*key))
            hm_layer.append(hm_cache[key] & 0x0FFF)
            path_layer.append(pick_path(col, row))

    items = build_items()

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
        f"wrote {path} ({len(header)+len(body)} bytes) map={MAP_W}x{MAP_D} "
        f"hm_tiles={len(hm_tiles)} items={len(items)} cliff={CLIFF_DEPTH}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_trt(OUT_DIR / "Level1.trt")
    write_ter(OUT_DIR / "Level1.ter")
    write_ter(OUT_DIR / "Level1Pro.ter")


if __name__ == "__main__":
    main()

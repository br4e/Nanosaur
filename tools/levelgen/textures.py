"""Biome texture painting using seamless multi-tile sheet parity and portal pads."""

from __future__ import annotations

import numpy as np

from .constants import (
    CANYON_SKIRT_TILE,
    CLIFF_TEX,
    D,
    KIND_SPIRE_178,
    LAVA_RIM_TILES,
    PAD_P1_2X2,
    PAD_SAND_4X4,
    SHEET_BASALT_2X2,
    SHEET_CANYON_2X2,
    SHEET_CLIFF_2X2,
    SHEET_GRASS_3X3,
    SHEET_LAVA_2X2,
    SHEET_MUD_2X2,
    SHEET_SAND_2X2,
    SHEET_WATER_LAGOON_2X2,
    SHEET_WATER_MARSH_2X2,
    W,
    WALL_FALLBACK,
    ZONE_CANYON,
    ZONE_LAGOON,
    ZONE_LAVA,
    ZONE_MARSH,
)
from .geometry import dilate8


def _sheet_tile(sheet, r, c):
    h = len(sheet)
    w = len(sheet[0])
    return sheet[r % h][c % w]


def _stamp_portal_pads(tex, portal_coords):
    """Stamp clean landmark pads centered on P1, P2, P3 (P0 uses 3x3 Marsh Grass)."""
    # P1: 2x2 Volcanic Portal Pad [[169, 170], [185, 186]] centered on lava_portal_island
    p1c, p1r = portal_coords[1]
    for dr in range(-1, 1):
        for dc in range(-1, 1):
            tex[p1r + dr, p1c + dc] = PAD_P1_2X2[(dr + 1) % 2][(dc + 1) % 2]

    # P2 & P3: 4x4 Framed Sand Pad centered on their raised overlook plateaus
    for pid in (2, 3):
        pc, pr = portal_coords[pid]
        r0, c0 = pr - 2, pc - 2
        for dr in range(4):
            for dc in range(4):
                tex[r0 + dr, c0 + dc] = PAD_SAND_4X4[dr][dc]


def paint_textures(world, hm_grid, orig, rng):
    """Paint seamless multi-tile sheet textures and clean portal pads."""
    del orig  # signature kept clean for pipeline symmetry
    kind = world["kind"]
    zone = world["zone"]
    walk = world["walk"]
    is_water_bed = world["is_water_bed"]
    is_water_47 = world["is_water_47"]
    is_water_shore = world["is_water_shore"]
    is_lava_tile = world["is_lava_tile"]
    is_lava_rim = world["is_lava_rim"]
    is_ford_tile = world["is_ford_tile"]
    crystal_bowl = world["crystal_bowl"]
    lagoon_meadow = world["lagoon_meadow"]
    portal_coords = world["portal_coords"]

    tex = np.zeros((D, W), dtype=np.uint16)

    # Cliff skirt: walkable tiles orthogonally/diagonally adjacent to non-walkable cliffs
    cliff_adj = walk & dilate8(~walk)

    for r in range(D):
        for c in range(W):
            v = int(hm_grid[r, c])
            key = (v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1)
            k = kind[r, c]
            z = zone[r, c]

            if not walk[r, c]:
                if key in CLIFF_TEX:
                    tex[r, c] = CLIFF_TEX[key]
                elif key[0] in (166, 167, 180, 181, 217, 235):
                    tex[r, c] = WALL_FALLBACK[(r + c) % len(WALL_FALLBACK)]
                else:
                    # High cliff plateaus: pure 2x2 sheets (no random mosaics)
                    if z == ZONE_LAVA:
                        tex[r, c] = _sheet_tile(SHEET_BASALT_2X2, r, c)
                    else:
                        tex[r, c] = _sheet_tile(SHEET_CLIFF_2X2, r, c)
                continue

            # Walkable tiles: STRICT hazard containment + pure sheet parity!
            # 1. Underwater (tmax <= 47): TILE_ATTRIB_WATER
            if is_water_bed[r, c] or is_water_47[r, c]:
                sheet = SHEET_WATER_LAGOON_2X2 if z == ZONE_LAGOON else SHEET_WATER_MARSH_2X2
                tex[r, c] = _sheet_tile(sheet, r, c)

            # 2. Lava bed (flat hm=67 in Zone 2): TILE_ATTRIB_LAVA
            elif is_lava_tile[r, c]:
                tex[r, c] = _sheet_tile(SHEET_LAVA_2X2, r, c)

            # 3. Water bank / shore ramp / ford ONLY (never speckled into grass)
            elif is_water_shore[r, c] or is_ford_tile[r, c]:
                tex[r, c] = _sheet_tile(SHEET_MUD_2X2, r, c)

            # 4. Lava Zone: pure 2x2 basalt, with 207 ONLY on direct lava rim tiles
            elif z == ZONE_LAVA:
                if is_lava_rim[r, c]:
                    tex[r, c] = LAVA_RIM_TILES[1]
                else:
                    tex[r, c] = _sheet_tile(SHEET_BASALT_2X2, r, c)

            # 5. Zone 1 Emerald Marsh: pure 3x3 grass sheet (ZERO random mud speckling)
            elif z == ZONE_MARSH:
                tex[r, c] = _sheet_tile(SHEET_GRASS_3X3, r, c)

            # 6. Zone 3 Crystal Canyon & Oasis Crossroads: pure 2x2 canyon (sand in bowl, 132 on cliff skirt)
            elif z == ZONE_CANYON:
                if crystal_bowl[r, c] and not cliff_adj[r, c]:
                    tex[r, c] = _sheet_tile(SHEET_SAND_2X2, r, c)
                elif cliff_adj[r, c]:
                    tex[r, c] = CANYON_SKIRT_TILE
                else:
                    tex[r, c] = _sheet_tile(SHEET_CANYON_2X2, r, c)

            # 7. Zone 4 Ancient Nest Lagoon: contiguous 3x3 grass meadows/islands or 2x2 canyon/basalt spires
            else:
                if k == KIND_SPIRE_178:
                    tex[r, c] = _sheet_tile(SHEET_BASALT_2X2, r, c)
                elif lagoon_meadow[r, c]:
                    tex[r, c] = _sheet_tile(SHEET_GRASS_3X3, r, c)
                elif cliff_adj[r, c]:
                    tex[r, c] = CANYON_SKIRT_TILE
                else:
                    tex[r, c] = _sheet_tile(SHEET_CANYON_2X2, r, c)

    _stamp_portal_pads(tex, portal_coords)
    return tex

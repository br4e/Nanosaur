"""Constants, item IDs, terrain kinds, zone IDs, and tile sheets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIG_DIR = ROOT / "tools" / "original_terrain"
OUT_TERRAIN = ROOT / "Data" / "Terrain"
OUT_IMAGES = ROOT / "Data" / "Images"
PREVIEW_PATH = ROOT / "tools" / "level_preview.png"
BUNDLE = ROOT / "build" / "RelWithDebInfo" / "Nanosaur.app" / "Contents" / "Resources"

W, D = 270, 355           # map size in tiles (cols, rows)
TILE_PX = 32              # Oreo map pixels per tile
SEED = 20261120

# ---------------------------------------------------------------- item types
IT_START, IT_POWERUP, IT_TRICER, IT_REX, IT_LAVA = 0, 1, 2, 3, 4
IT_EGG, IT_GASVENT, IT_PTERA, IT_STEGO, IT_PORTAL = 5, 6, 7, 8, 9
IT_TREE, IT_BOULDER, IT_MUSHROOM, IT_BUSH, IT_WATER = 10, 11, 12, 13, 14
IT_CRYSTAL, IT_SPITTER, IT_STEPSTONE, IT_ROLLBOULDER, IT_SPOREPOD = 15, 16, 17, 18, 19
ENEMY_TYPES = (IT_TRICER, IT_REX, IT_PTERA, IT_STEGO, IT_SPITTER)

# tree parm0 (Items.c AddTree)
TREE_FERN, TREE_STICKPALM, TREE_BAMBOO, TREE_CYPRESS, TREE_MAINPALM, TREE_PINEPALM = range(6)

# powerup parm0 (Triggers.c AddPowerUp)
POW_HEAT, POW_LASER, POW_TRI, POW_HEALTH, POW_SHIELD, POW_NUKE, POW_SONIC = range(7)

# ------------------------------------------------------------- terrain kinds
KIND_CLIFF = 0
KIND_CRAMP_U = 1
KIND_CRAMP_L = 2
KIND_FLOOR = 3
KIND_WATER_BED = 4
KIND_WATER_SHELF = 5
KIND_FORD = 6
KIND_LAVA_BED = 7
KIND_TERRACE_109 = 8
KIND_TERRACE_128 = 9
KIND_SPIRE_178 = 10
KIND_WALK_RAMP = 11

# Zone IDs
ZONE_MARSH = 0
ZONE_LAVA = 1
ZONE_CANYON = 2
ZONE_LAGOON = 3

ZONE_NAMES = [
    "Emerald Marsh",
    "Obsidian Lava Gorge",
    "Crystal Canyon",
    "Ancient Nest Lagoon",
]

# --------------------------------------------------- multi-tile texture sheets
# Indexed strictly by (r % H, c % W) or pad-local (dr, dc) for seamless tiling.
SHEET_GRASS_3X3 = [
    [0, 1, 2],
    [19, 20, 21],
    [38, 39, 40],
]
SHEET_MUD_2X2 = [
    [74, 75],
    [89, 90],
]
SHEET_WATER_MARSH_2X2 = [
    [23, 24],
    [42, 43],
]
SHEET_WATER_LAGOON_2X2 = [
    [223, 224],
    [231, 232],
]
SHEET_BASALT_2X2 = [
    [148, 149],
    [164, 165],
]
SHEET_LAVA_2X2 = [
    [137, 138],
    [151, 152],
]
SHEET_CANYON_2X2 = [
    [189, 190],
    [200, 201],
]
SHEET_SAND_2X2 = [
    [144, 145],
    [160, 161],
]
SHEET_CLIFF_2X2 = [
    [17, 18],
    [36, 37],
]

# Edge/skirt accents (never scattered into open floor)
LAVA_RIM_TILES = (195, 207)
CANYON_SKIRT_TILE = 132

# Portal pads
PAD_P1_2X2 = [
    [169, 170],
    [185, 186],
]
PAD_SAND_4X4 = [
    [127, 128, 129, 130],
    [143, 144, 145, 146],
    [159, 160, 161, 162],
    [175, 176, 177, 178],
]

# Original Level1 cliff kit (orth stacks + diagonal corners)
CLIFF_ORTH = {  # facing -> (lower_tid, upper_tid, fx, fy)
    "E": (167, 166, 0, 0),
    "W": (167, 166, 1, 0),
    "S": (235, 217, 0, 0),
    "N": (235, 217, 0, 1),
}
CLIFF_DIAG = {"SW": (0, 0), "SE": (1, 0), "NW": (0, 1), "NE": (1, 1)}
CLIFF_TEX = {  # (tid, fx, fy) -> full texture value from original Level1
    (167, 0, 0): 20561, (167, 0, 1): 20560, (167, 1, 0): 4176, (167, 1, 1): 4177,
    (166, 0, 0): 20545, (166, 0, 1): 20544, (166, 1, 0): 4160, (166, 1, 1): 4161,
    (235, 0, 0): 80, (235, 0, 1): 16464, (235, 1, 0): 32848, (235, 1, 1): 49232,
    (217, 0, 0): 64, (217, 0, 1): 16448, (217, 1, 0): 32832, (217, 1, 1): 49216,
    (181, 0, 0): 16506, (181, 0, 1): 66, (181, 1, 0): 49218, (181, 1, 1): 32890,
    (180, 0, 0): 16496, (180, 0, 1): 50, (180, 1, 0): 49264, (180, 1, 1): 32880,
    (182, 0, 0): 16516, (182, 0, 1): 132, (182, 1, 0): 49355, (182, 1, 1): 32971,
}
WALL_FALLBACK = [64, 65, 66, 122, 80, 81]

TILE_ATTRIB_MAKEDUST = 1 << 4
TILE_ATTRIB_LAVA = 1 << 5
TILE_ATTRIB_WATER = 1 << 7

PATH_OPEN = 0
PATH_SOLID_ALL = 9
PATH_SLOPEFORCE = 43

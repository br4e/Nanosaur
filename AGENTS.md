# AGENTS.md — Nanosaur (`custom-grassy-level`)

Guidance for AI coding agents working in this repository.

---

## 1. Repository & Branch Context
- **Upstream:** Fork of `jorio/Nanosaur` (modern SDL3/C/C++20 port of Pangea Software's 1998 *Nanosaur*).
- **Active custom-level branch:** `custom-grassy-level` (rebased onto `core-game-mods` / `game_balance`, which include the lower-left minimap overlay, extended draw distance, skipped boot intros, and gameplay balance tweaks).
- **Git workflow:** Commit and push completed changes directly to `origin/custom-grassy-level`. Keep the working tree clean (no `__pycache__/`, `*.pyc`, or temporary scripts).
- **Submodule:** `extern/Pomme` is required for building C/C++ sources (`git submodule update --init --recursive`).

---

## 2. Build, Run & Validation Commands
- **Regenerate & validate the custom level (fast, Python 3 + `numpy` + `Pillow`):**
  ```bash
  PYTHONDONTWRITEBYTECODE=1 python3 tools/generate_level.py
  ```
  Writes `Data/Terrain/Level1.ter`, `Data/Terrain/Level1Pro.ter`, `Data/Images/Map.tga`, and `tools/level_preview.png`, then runs all 11 automated validation gates (and copies assets into `build/RelWithDebInfo/Nanosaur.app/Contents/Resources/` if present).
- **Build & launch on macOS (local dev machine):**
  ```bash
  ./build-and-run.sh
  ```
- **Cross-platform build (macOS / Windows / Linux):**
  ```bash
  python3 build.py -1 -2 -3
  ```
  *(Note: Cloud Linux VMs generally do not have macOS Xcode or system SDL3 installed; validate level changes via `tools/generate_level.py` and clearly mark C/runtime playtesting as unverified when not run.)*

---

## 3. Custom Level Design Direction
- **Goal:** A cohesive, organic, human-feeling *Nanosaur* level at **~1/4 the playable area of original `Level1.ter`** (`~6,000–6,800` BFS-playable tiles vs. `25,420` original) on the standard `270×355` tile grid.
- **Layout:** Asymmetric S-curve valley & canyon world (never a symmetric ring/donut) with 4 themed zones:
  1. **Emerald Marsh (NW):** Safe grassy start (`S`), elevated `P0` overlook, curved marsh lakes + fordable island, `Species 0` eggs, ferns/bamboo/cypress/mushrooms, `Rex` + `Stego`.
  2. **Obsidian Lava Gorge (NE):** Flooded basalt lava caldera, central `P1` island, 2 reincarnating `StepStone` bridges, `GasVent`s, `Species 1` eggs, ambush `Bush`es, `Ptera` + `Tricer` + `Rex`.
  3. **Crystal Canyon & Oasis (Central/W):** Winding ochre canyon, dusty `Crystal Bowl` (`MAKEDUST` sand), `Palm Oasis`, elevated `P2` overlook, rim-mounted `RollBoulder`s, a hidden side alcove, western return loop, `Species 2` eggs, `Spitter` + `Rex` + `Tricer` + `Ptera`.
  4. **Ancient Nest Lagoon (South):** Endgame lagoon entered via a narrow throat onto the elevated `P3` overlook, smooth walkable nest terraces + `hm=178` jetpack spires paired with `GasVent`s, `Species 3` + `Species 4` eggs, rich `PowerUp`s, mixed endgame enemies.
- **Pacing:** Alternate open basins with narrow winding canyon throats (`3–5` tiles wide—never 1-tile pinches) and small side alcoves, inspired by `tools/original_terrain/Level1.ter`.

---

## 4. Level Generator Package (`tools/levelgen/`)
`tools/generate_level.py` is a thin CLI entrypoint calling `tools/levelgen/cli.py`:
- `constants.py` — Grid dimensions (`270×355`), item/terrain/zone IDs, cliff kit LUTs, and multi-tile texture sheets.
- `original.py` — Loads stock reference data from `tools/original_terrain/Level1.ter` and `Level1.trt`.
- `geometry.py` — Morphological mask helpers (`dilate8`, `erode8`, `dilate_n`), blobs/capsules, noise fields, and walkable heightfield smoothing.
- `world.py` — Zone masks, narrow connectors, lakes/lava/terraces/spires, vertex heightfield `V`, and tile classification (`build_world`).
- `heightmap.py` — Exact-corner bilinear heightmap tile synthesis, cliff kit stamping, and `path_layer` generation (`build_heightmaps`).
- `textures.py` — Multi-tile parity biome texturing and landmark portal pads (`paint_textures`).
- `items.py` — `Placer`, `compute_flat_3x3_mask`, water/lava patch tiling, `StepStone` bridges, portals, eggs, enemies (Normal & Pro), and scenery (`build_items`).
- `io.py` — Big-endian `.ter` writer, 8-bit RLE colormapped `Map.tga` writer, and `tools/level_preview.png` renderer.
- `validate.py` — 11-gate validation suite (`validate_ter`, `validate_tga`).
- `cli.py` — Orchestrates generation, writing, validation, and optional `.app` bundle copy.

---

## 5. Critical Engine & Terrain Invariants (Do Not Break)

### Grid, Scale & Limits
- Map grid: `270×355` tiles (`54×71` supertiles of `5×5`), `TERRAIN_POLYGON_SIZE = 140` world units/tile, `HEIGHT_EXTRUDE_FACTOR = 4.0f` (`worldY = hm * 4`).
- Hard engine limits (`src/System/File.c`, `src/Headers/terrain.h`): `MAX_HEIGHTMAP_TILES = 300`, `258` tiles in `Level1.trt`, items sorted by ascending `x` in `gMasterItemList`.

### Vertex Height Continuity (`src/Terrain/Terrain.c`)
- `GetTerrainHeightAtRowCol(r, c)` samples **only pixel `(0, 0)` (post-flip)** of `hm_tile[r, c]` for vertex `(r, c)`.
- Every walkable tile and cliff-perimeter corner touching `walk` must match `V[r, c]` exactly (`corner_mismatch == 0`, `perimeter_spikes == 0`). Never stamp flipped cliff tile `182` onto `path == 0` tiles.
- Keep walkable slope steps `<= 20` heightmap units/tile (`normal.y > 0.85`) so `Player_Control.c` does not slide the player downhill on ramps.
- Cliffs (`hm = 255` and `CRAMP_U`) use `path = 9` (`PATH_TILE_SOLID_ALL`); lower cliff skirts (`CRAMP_L`) use `path = 43` (`PATH_TILE_SLOPEFORCE`).

### Water & Lava Basins (`src/Items/Items.c`)
- `IT_WATER` (`parm3 = 0`) renders an `8×8`-tile mesh at `FIXED_WATER_Y = 210` (`hm = 52.5`). Water beds sit at `hm = 41` (`Y = 164`), surrounded by `67 -> 85` banks.
- `IT_LAVA` (`parm3 = 0`, or `2` for fireballs) renders an `8×8`-tile mesh at `FIXED_LAVA_Y = 305` (`hm = 76.25`). Lava beds sit at `hm = 67` (`Y = 268`), surrounded by `hm = 85` (`Y = 340`) banks/islands.
- Both `IT_WATER` and `IT_LAVA` **must cover 100%** of underwater/lava tiles (overlapping `8×8` patches across higher shore/rim banks is intentional and matches original `Level1.ter`).
- **No dry attribute leaks:** `TILE_ATTRIB_WATER` (`23, 24, 42, 43`, `223, 224, 231, 232`) and `TILE_ATTRIB_LAVA` (`137, 138, 151, 152`) must **only** be painted on submerged water/lava beds—never on dry `FLOOR` tiles (they halve walk speed / inflict lava damage).

### StepStones & Jump Ergonomics (`src/Items/Triggers.c`, `src/Player/Player_Control.c`)
- Player single-jump reach is `~2.86` tiles (`400` units); double-jump reach is `~4.3–5.3` tiles (`600–748` units).
- Place `IT_STEPSTONE` directly on flat `hm = 67` lava tiles with `parm = (0, 0, 0, 1)` (reincarnates after sinking), spaced `<= 3.5` tiles center-to-center, and keep nearby `IT_LAVA` patches fireball-free (`parm3 = 0`).

### Ground Textures (Anti-Mosaic Rules)
- Never scatter incompatible ground materials randomly per tile. Use coordinate-parity multi-tile sheets from `Level1.trt`:
  - **Marsh Grass:** `3×3` `[[0, 1, 2], [19, 20, 21], [38, 39, 40]]` (`r % 3, c % 3`)
  - **Marsh Mud / Shore:** `2×2` `[[74, 75], [89, 90]]` (`r % 2, c % 2`) — strictly on water shores/fords
  - **Marsh Water:** `2×2` `[[23, 24], [42, 43]]` (`r % 2, c % 2`)
  - **Volcanic Basalt:** `2×2` `[[148, 149], [164, 165]]` (`r % 2, c % 2`); `207` only on `is_lava_rim`
  - **Lava:** `2×2` `[[137, 138], [151, 152]]` (`r % 2, c % 2`)
  - **Canyon Floor:** `2×2` `[[189, 190], [200, 201]]` (`r % 2, c % 2`); `132` only on cliff-adjacent skirt
  - **Crystal Sand:** `2×2` `[[144, 145], [160, 161]]` (`r % 2, c % 2`)
  - **Lagoon Water:** `2×2` `[[223, 224], [231, 232]]` (`r % 2, c % 2`)
  - **Portal Pads:** `PAD_P1_2X2` (`[169,170 / 185,186]`) on `P1`; `PAD_SAND_4X4` (`127..130 / 143..146 / 159..162 / 175..178`) on `P2` and `P3`.
- Enforced by `validate.py` Check 11 (`mean_rgb_jump < 14.0`, `pct_gt40 < 8.0%`).

### Flat Portals & Eggs
- All 4 portals (`0..3`) and all 23 non-spire eggs (25 total, 5 per species `0..4`, all `parm3 = 1` nests) must sit on `3×3` strictly flat `path == 0` terrain (all 16 vertices in `V_read[r-1:r+3, c-1:c+3]` equal, clear of cliffs/water/lava).
- Spire eggs (`Species 3`) sit on unclipped `2×2` flat `hm = 178` caps with nearby `GasVent`s.
- Enforced by `validate.py` Check 10.

### Streaming, Enemy Caps & Maps
- **`src/System/ProMode.c`:** `SUPERTILE_ACTIVE_RANGE` is tuned to `8` (Normal) / `10` (Pro) with `MAX_ENEMIES = 14` (Normal) / `30` (Pro). Do not raise `SUPERTILE_ACTIVE_RANGE` to cover the whole 1/4-scale map, or `PrimeInitialTerrain()` will saturate `MAX_ENEMIES` on frame 0 in Zone 1 and starve Zones 2 and 4.
- **`Data/Images/Map.tga` vs. `src/Screens/Minimap.c`:** The HUD rotating GPS (`src/Screens/Infobar.c`) reads `Data/Images/Map.tga` (`270×355`, 8-bit RLE colormapped, top-left origin `0x20`). `src/Screens/Minimap.c` builds its lower-left overlay directly from `gTerrainTextureLayer` and crops UVs to non-solid (`path != 9/24`) bounds.

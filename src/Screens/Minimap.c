/****************************/
/*  		MINIMAP.C		*/
/* Full-level terrain overlay*/
/****************************/


/****************************/
/*    EXTERNALS             */
/****************************/

#include "game.h"

extern int		gWindowWidth;
extern int		gWindowHeight;


/****************************/
/*    CONSTANTS             */
/****************************/

#define	MINIMAP_MARGIN_PX		12.0f
#define	MINIMAP_HEIGHT_FRAC		0.22f
#define	MINIMAP_ALPHA			0.85f
#define	MINIMAP_MARKER_SIZE		10.0f
#define	MINIMAP_FIT_PAD_TILES	4			// padding around playable bbox


/****************************/
/*    VARIABLES             */
/****************************/

static GLuint	gMinimapTextureName	= 0;
static Boolean	gMinimapReady		= false;
static Boolean	gMinimapVisible		= true;			// default on for navigation

// Playable-area zoom window in tile UV space (0..1 over full map texture).
static float	gMinimapU0 = 0.0f, gMinimapU1 = 1.0f;
static float	gMinimapV0 = 0.0f, gMinimapV1 = 1.0f;


/****************************/
/*    FUNCTIONS             */
/****************************/


/******************** INIT MINIMAP ********************/
//
// Build a 1-pixel-per-tile overview texture from the loaded terrain.
// Call after LoadLevelArt so gTerrainTextureLayer / gTileDataPtr are valid.
//

void InitMinimap(void)
{
	UInt16		*buf;
	const int	w = gTerrainTileWidth;
	const int	d = gTerrainTileDepth;
	const int	tileSize = OREOMAP_TILE_SIZE;
	const int	texelsPerTile = tileSize * tileSize;

	DisposeMinimap();								// safe if nothing allocated yet

	if (w <= 0 || d <= 0 || gTileDataPtr == nil || gTerrainTextureLayer == nil)
		return;

	buf = (UInt16 *)AllocPtr((long)w * (long)d * sizeof(UInt16));
	GAME_ASSERT(buf);

			/* AVERAGE EACH TILE'S TEXELS INTO ONE PIXEL */

	for (int row = 0; row < d; row++)
	{
		for (int col = 0; col < w; col++)
		{
			UInt16	tile		= gTerrainTextureLayer[row][col];
			int		texMapNum	= tile & TILENUM_MASK;
			UInt32	sumR = 0, sumG = 0, sumB = 0;

			if (texMapNum >= gNumTerrainTextureTiles)
				texMapNum = 0;

			const UInt16 *tileData = gTileDataPtr + (texMapNum * texelsPerTile);

			for (int i = 0; i < texelsPerTile; i++)
			{
				UInt16 p = tileData[i];
				sumR += (p >> 10) & 31;
				sumG += (p >> 5) & 31;
				sumB += p & 31;
			}

			UInt16 avgR = (UInt16)(sumR / texelsPerTile);
			UInt16 avgG = (UInt16)(sumG / texelsPerTile);
			UInt16 avgB = (UInt16)(sumB / texelsPerTile);

			buf[row * w + col] = (UInt16)((avgR << 10) | (avgG << 5) | avgB);
		}
	}

	gMinimapTextureName = Render_LoadTexture(
			GL_RGB,
			w,
			d,
			GL_BGRA,
			GL_UNSIGNED_SHORT_1_5_5_5_REV,
			buf,
			kRendererTextureFlags_ClampBoth);

	DisposePtr((Ptr)buf);

			/* ZOOM-TO-FIT: UV window around playable (non-solid) tiles */

	{
		int		minC = w, maxC = -1, minR = d, maxR = -1;
		int		pad = MINIMAP_FIT_PAD_TILES;

		if (gTerrainPathLayer != nil)
		{
			for (int row = 0; row < d; row++)
			{
				for (int col = 0; col < w; col++)
				{
					UInt16 p = gTerrainPathLayer[row][col] & TILENUM_MASK;
					if (p == PATH_TILE_SOLID_ALL || p == PATH_TILE_SOLID_ALL2)
						continue;
					if (col < minC) minC = col;
					if (col > maxC) maxC = col;
					if (row < minR) minR = row;
					if (row > maxR) maxR = row;
				}
			}
		}

		if (maxC >= minC && maxR >= minR)
		{
			minC = (minC - pad < 0) ? 0 : minC - pad;
			minR = (minR - pad < 0) ? 0 : minR - pad;
			maxC = (maxC + pad >= w) ? w - 1 : maxC + pad;
			maxR = (maxR + pad >= d) ? d - 1 : maxR + pad;

			gMinimapU0 = (float)minC / (float)w;
			gMinimapU1 = (float)(maxC + 1) / (float)w;
			gMinimapV0 = (float)minR / (float)d;
			gMinimapV1 = (float)(maxR + 1) / (float)d;
		}
		else
		{
			gMinimapU0 = 0.0f; gMinimapU1 = 1.0f;
			gMinimapV0 = 0.0f; gMinimapV1 = 1.0f;
		}
	}

	gMinimapReady = true;
}


/******************* DISPOSE MINIMAP ******************/

void DisposeMinimap(void)
{
	if (gMinimapTextureName != 0)
	{
		glDeleteTextures(1, &gMinimapTextureName);
		gMinimapTextureName = 0;
	}

	gMinimapReady = false;
	gMinimapU0 = 0.0f; gMinimapU1 = 1.0f;
	gMinimapV0 = 0.0f; gMinimapV1 = 1.0f;
}


/******************** DRAW MINIMAP ********************/
//
// Screen-space overlay in the lower-left. Safe no-op until InitMinimap.
// Toggle with N (does not steal G/M/Tab/B).
//

void DrawMinimap(void)
{
	float			mapW, mapH;
	float			screenLeft, screenRight, screenTop, screenBottom;
	float			ndcLeft, ndcRight, ndcTop, ndcBottom;
	TQ3Point2D		pts[4];
	TQ3Param2D		uvs[4];
	const uint8_t	quadTris[6] = { 0, 1, 2, 1, 3, 2 };
	GLboolean		wasBlend, wasTex2D, wasTexCoord;

	if (!gMinimapReady)
		return;

			/* TOGGLE VISIBILITY */

	if (GetNewSDLKeyState(SDL_SCANCODE_N))
		gMinimapVisible = !gMinimapVisible;

	if (!gMinimapVisible)
		return;

	if (gTerrainTileWidth <= 0 || gTerrainTileDepth <= 0)
		return;

			/* LAYOUT: LOWER-LEFT, NORTH-UP, ASPECT-CORRECT TO PLAYABLE CROP */

	{
		float	uSpan = gMinimapU1 - gMinimapU0;
		float	vSpan = gMinimapV1 - gMinimapV0;
		float	aspect = (uSpan * (float)gTerrainTileWidth) /
						 (vSpan * (float)gTerrainTileDepth);

		mapH = gWindowHeight * MINIMAP_HEIGHT_FRAC;
		mapW = mapH * aspect;
	}

	screenLeft		= MINIMAP_MARGIN_PX;
	screenRight		= screenLeft + mapW;
	screenBottom	= (float)gWindowHeight - MINIMAP_MARGIN_PX;
	screenTop		= screenBottom - mapH;

	ndcLeft		= 2.0f * screenLeft   / gWindowWidth  - 1.0f;
	ndcRight	= 2.0f * screenRight  / gWindowWidth  - 1.0f;
	ndcTop		= 1.0f - 2.0f * screenTop    / gWindowHeight;
	ndcBottom	= 1.0f - 2.0f * screenBottom / gWindowHeight;

	//		2----3
	//		| \  |
	//		|  \ |
	//		0----1
	pts[0] = (TQ3Point2D){ ndcLeft,  ndcBottom };
	pts[1] = (TQ3Point2D){ ndcRight, ndcBottom };
	pts[2] = (TQ3Point2D){ ndcLeft,  ndcTop };
	pts[3] = (TQ3Point2D){ ndcRight, ndcTop };

	// Crop UVs to playable bbox. V=0 (row 0 / north) at TOP of overlay.
	uvs[0] = (TQ3Param2D){ gMinimapU0, gMinimapV1 };
	uvs[1] = (TQ3Param2D){ gMinimapU1, gMinimapV1 };
	uvs[2] = (TQ3Param2D){ gMinimapU0, gMinimapV0 };
	uvs[3] = (TQ3Param2D){ gMinimapU1, gMinimapV0 };

			/* SAVE GL STATE (renderer state cache lives in Renderer.c) */

	wasBlend	= glIsEnabled(GL_BLEND);
	wasTex2D	= glIsEnabled(GL_TEXTURE_2D);
	wasTexCoord	= glIsEnabled(GL_TEXTURE_COORD_ARRAY);

	glViewport(0, 0, gWindowWidth, gWindowHeight);
	Render_Enter2D();

			/* MAP QUAD */

	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_2D);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	Render_BindTexture(gMinimapTextureName);
	glColor4f(1, 1, 1, MINIMAP_ALPHA);
	glVertexPointer(2, GL_FLOAT, 0, pts);
	glTexCoordPointer(2, GL_FLOAT, 0, uvs);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, quadTris);

			/* 1px DARK BORDER */

	{
		float			inset = 1.0f;
		float			bL = 2.0f * (screenLeft + inset) / gWindowWidth - 1.0f;
		float			bR = 2.0f * (screenRight - inset) / gWindowWidth - 1.0f;
		float			bT = 1.0f - 2.0f * (screenTop + inset) / gWindowHeight;
		float			bB = 1.0f - 2.0f * (screenBottom - inset) / gWindowHeight;
		TQ3Point2D		border[5] = {
			{ bL, bB }, { bR, bB }, { bR, bT }, { bL, bT }, { bL, bB }
		};

		glDisable(GL_TEXTURE_2D);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glColor4f(0, 0, 0, MINIMAP_ALPHA);
		glVertexPointer(2, GL_FLOAT, 0, border);
		glDrawArrays(GL_LINE_STRIP, 0, 5);
	}

			/* PLAYER MARKER */

	if (gPlayerObj != nil)
	{
		float	u = (gMyCoord.x * gOneOver_TERRAIN_POLYGON_SIZE) / (float)gTerrainTileWidth;
		float	v = (gMyCoord.z * gOneOver_TERRAIN_POLYGON_SIZE) / (float)gTerrainTileDepth;
		float	px, py;
		float	fx, fy;			// facing in map/pixel space (+y = south / down)
		float	half = MINIMAP_MARKER_SIZE * 0.5f;
		float	tipLen, baseLen, baseHalf;
		float	uSpan = gMinimapU1 - gMinimapU0;
		float	vSpan = gMinimapV1 - gMinimapV0;
		TQ3Point2D	outline[3], tip[3];

		// Remap full-map UV into the playable crop window.
		if (uSpan > 0.0001f) u = (u - gMinimapU0) / uSpan;
		if (vSpan > 0.0001f) v = (v - gMinimapV0) / vSpan;

		if (u < 0) u = 0;
		if (u > 1) u = 1;
		if (v < 0) v = 0;
		if (v > 1) v = 1;

		px = screenLeft + u * mapW;
		py = screenTop  + v * mapH;

		// Forward motion uses (-sin(Rot.y), -cos(Rot.y)) — see Player_Control.c.
		// Rot.y==0 faces north (-Z), which is toward the top of this north-up map.
		fx = -sin(gPlayerObj->Rot.y);
		fy = -cos(gPlayerObj->Rot.y);

		tipLen		= half;
		baseLen		= half * 0.45f;
		baseHalf	= half * 0.55f;

		#define MARKER_TO_NDC(sx, sy, out) \
			do { \
				(out).x = 2.0f * (sx) / gWindowWidth - 1.0f; \
				(out).y = 1.0f - 2.0f * (sy) / gWindowHeight; \
			} while (0)

		MARKER_TO_NDC(px + fx * tipLen,
					  py + fy * tipLen, tip[0]);
		MARKER_TO_NDC(px - fx * baseLen - fy * baseHalf,
					  py - fy * baseLen + fx * baseHalf, tip[1]);
		MARKER_TO_NDC(px - fx * baseLen + fy * baseHalf,
					  py - fy * baseLen - fx * baseHalf, tip[2]);

		// Slightly larger black outline underneath for contrast
		{
			float oTip = tipLen + 1.5f;
			float oBase = baseLen + 1.0f;
			float oHalf = baseHalf + 1.0f;

			MARKER_TO_NDC(px + fx * oTip,
						  py + fy * oTip, outline[0]);
			MARKER_TO_NDC(px - fx * oBase - fy * oHalf,
						  py - fy * oBase + fx * oHalf, outline[1]);
			MARKER_TO_NDC(px - fx * oBase + fy * oHalf,
						  py - fy * oBase - fx * oHalf, outline[2]);
		}

		#undef MARKER_TO_NDC

		glDisable(GL_TEXTURE_2D);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

		glColor4f(0, 0, 0, 1);
		glVertexPointer(2, GL_FLOAT, 0, outline);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glColor4f(1, 1, 0, 1);						// bright yellow
		glVertexPointer(2, GL_FLOAT, 0, tip);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	Render_Exit2D();

			/* RESTORE GL STATE TO MATCH RENDERER CACHE */

	if (wasBlend)	glEnable(GL_BLEND);				else glDisable(GL_BLEND);
	if (wasTex2D)	glEnable(GL_TEXTURE_2D);		else glDisable(GL_TEXTURE_2D);
	if (wasTexCoord)	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	else				glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glColor4f(1, 1, 1, 1);
}

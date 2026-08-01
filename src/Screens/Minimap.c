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


/****************************/
/*    VARIABLES             */
/****************************/

static GLuint	gMinimapTextureName	= 0;
static Boolean	gMinimapReady		= false;
static Boolean	gMinimapVisible		= true;			// default on for navigation


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

			/* LAYOUT: LOWER-LEFT, NORTH-UP, ASPECT-CORRECT */

	mapH = gWindowHeight * MINIMAP_HEIGHT_FRAC;
	mapW = mapH * ((float)gTerrainTileWidth / (float)gTerrainTileDepth);

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

	// V=0 (row 0 / north) at TOP of overlay — same convention as fullscreen quads.
	uvs[0] = (TQ3Param2D){ 0, 1 };
	uvs[1] = (TQ3Param2D){ 1, 1 };
	uvs[2] = (TQ3Param2D){ 0, 0 };
	uvs[3] = (TQ3Param2D){ 1, 0 };

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
		TQ3Point2D	outline[3], tip[3];

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

// There used to be two separate builds for the game: standard Nanosaur, and
// Nanosaur Extreme. The settings below used to be differentiated with the
// PRO_MODE preprocessor define. However, the source port lets you pick either
// version of the game with the same build.

#include "game.h"

int		PRO_MODE;
int		SUPERTILE_ACTIVE_RANGE;
int		MAX_ENEMIES;
int		MAX_PTERA;
int		MAX_REX;
int		MAX_SPITTER;
int		MAX_STEGO;
int		MAX_TRICER;
int		EXPLODEGEOMETRY_DENOMINATOR;
float	YON_DISTANCE;
float	SONIC_SCREAM_RATE;
float	BLASTER_RATE;
float	HEATSEEK_RATE;
float	TRIBLAST_RATE;
float	NUKE_RATE;

extern	SDL_Window		*gSDLWindow;

void SetProModeSettings(int pro)
{
	PRO_MODE					= pro;
	SUPERTILE_ACTIVE_RANGE		= pro ? 10 : 8;	// ~2.5x stock draw distance (~85x85 tiles) so zones stream smoothly without filling MAX_ENEMIES at frame 0
	MAX_ENEMIES					= pro ? 30 : 14;
	MAX_PTERA					= pro ? 10 : 4;
	MAX_REX						= pro ? 8 : 4;
	MAX_SPITTER					= pro ? 8 : 10;
	MAX_STEGO					= pro ? 10 : 3;
	MAX_TRICER					= pro ? 10 : 4;
	EXPLODEGEOMETRY_DENOMINATOR	= pro ? 2 : 4;
	YON_DISTANCE				= pro ? 22400.0f : 15200.0f;	// 8x stock fog/draw yon (was 2800 / 1900)
	SONIC_SCREAM_RATE			= pro ? 4 : 4;
	BLASTER_RATE				= pro ? 7 : 4;
	HEATSEEK_RATE				= pro ? 2 : 2;
	TRIBLAST_RATE				= pro ? 7 : 3;
	NUKE_RATE					= pro ? .05 : .05;

	// Re-set window title
	if (gSDLWindow)
	{
		static char windowTitle[256];
		SDL_snprintf(windowTitle, sizeof(windowTitle),
			"%s%s %s", GAME_FULL_NAME, PRO_MODE ? " Extreme" : "", GAME_VERSION);
		SDL_SetWindowTitle(gSDLWindow, windowTitle);
	}
}

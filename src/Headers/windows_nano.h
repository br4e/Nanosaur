//
// windows.h
//

#define GAME_VIEW_WIDTH		(640)
#define GAME_VIEW_HEIGHT	(480)

extern void	DumpGWorldToGWorld(GWorldPtr, GWorldPtr, Rect *, Rect *);
extern	void MakeFadeEvent(Boolean	fadeIn);
void SetFullscreenMode(void);

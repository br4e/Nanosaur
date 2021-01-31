#include "Pomme.h"
#include "PommeInit.h"
#include "PommeFiles.h"
#include "PommeGraphics.h"
#include "GamePatches.h"

#include <Quesa.h>
#include <SDL.h>

#include <iostream>

#if __APPLE__
#include <libproc.h>
#include <unistd.h>
#endif

extern "C"
{
	// bare minimum from Windows.c to satisfy externs in game code
	SDL_Window* gSDLWindow = nullptr;
	WindowPtr gCoverWindow = nullptr;
	UInt32* gCoverWindowPixPtr = nullptr;

	// Lets the game know where to find its asset files
	extern FSSpec gDataSpec;


	extern int PRO_MODE;

	// Tell Windows graphics driver that we prefer running on a dedicated GPU if available
#if _WIN32
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
	__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
#endif

	void GameMain(void);
}

static void FindGameData()
{
	fs::path dataPath;

#if __APPLE__
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];

	pid_t pid = getpid();
	int ret = proc_pidpath(pid, pathbuf, sizeof(pathbuf));
	if (ret <= 0)
	{
		throw std::runtime_error(std::string(__func__) + ": proc_pidpath failed: " + std::string(strerror(errno)));
	}

	dataPath = pathbuf;
	dataPath = dataPath.parent_path().parent_path() / "Resources";
#else
	dataPath = "Data";
#endif

	dataPath = dataPath.lexically_normal();

	// Set data spec
	gDataSpec = Pomme::Files::HostPathToFSSpec(dataPath / "Skeletons");

	// Use application resource file
	auto applicationSpec = Pomme::Files::HostPathToFSSpec(dataPath / "Application");
	short resFileRefNum = FSpOpenResFile(&applicationSpec, fsRdPerm);
	UseResFile(resFileRefNum);
}

int CommonMain(int argc, const char** argv)
{
	// Start our "machine"
	Pomme::Init();

	// Initialize SDL video subsystem
	if (0 != SDL_Init(SDL_INIT_VIDEO))
		throw std::runtime_error("Couldn't initialize SDL video subsystem.");

	// Create window
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#ifndef _WIN32
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#endif
//	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
//	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
	gSDLWindow = SDL_CreateWindow(
			"Nanosaur",
			SDL_WINDOWPOS_UNDEFINED,
			SDL_WINDOWPOS_UNDEFINED,
			640,
			480,
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
	if (!gSDLWindow)
		throw std::runtime_error("Couldn't create SDL window.");

	// Set up globals that the game expects
	gCoverWindow = Pomme::Graphics::GetScreenPort();
	gCoverWindowPixPtr = (UInt32*) GetPixBaseAddr(GetGWorldPixMap(gCoverWindow));

	// Clear window
	ExclusiveOpenGLMode_Begin();
	ClearBackdrop(0xFFA5A5A5);
	RenderBackdropQuad(BACKDROP_FILL);
	ExclusiveOpenGLMode_End();

	FindGameData();
#if !(__APPLE__)
	Pomme::Graphics::SetWindowIconFromIcl8Resource(gSDLWindow, 128);
#endif

	// Initialize Quesa
	auto qd3dStatus = Q3Initialize();
	if (qd3dStatus != kQ3Success)
	{
		throw std::runtime_error("Couldn't initialize Quesa.");
	}

	// Start the game
	try
	{
		GameMain();
	}
	catch (Pomme::QuitRequest&)
	{
		// no-op, the game may throw this exception to shut us down cleanly
	}

	// Clean up
	Pomme::Shutdown();

	return 0;
}

int main(int argc, char** argv)
{
	std::string uncaught;

	try
	{
		return CommonMain(argc, const_cast<const char**>(argv));
	}
	catch (std::exception& ex)
	{
		uncaught = ex.what();
	}
	catch (...)
	{
		uncaught = "unknown";
	}

	std::cerr << "Uncaught exception: " << uncaught << "\n";
	SDL_ShowSimpleMessageBox(0, "Uncaught Exception", uncaught.c_str(), nullptr);
	return 1;
}

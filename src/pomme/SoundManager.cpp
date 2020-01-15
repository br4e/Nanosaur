#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>
#include "PommeInternal.h"
#include "cmixer.h"

#define LOG POMME_GENLOG(POMME_DEBUG_SOUND, "SOUN")

static SndChannelPtr headChan = nullptr;
static int nManagedChans = 0;

static cmixer::WavStream* wip_ugly_stream = nullptr;

// Internal channel info
struct ChannelEx {
	SndChannelPtr prevChan;
	bool macChannelStructAllocatedByPomme;
	cmixer::WavStream* stream;
};

static inline ChannelEx& GetEx(SndChannelPtr chan)
{
	return *(ChannelEx*)chan->firstMod;
}

static inline SndChannelPtr* NextOf(SndChannelPtr chan)
{
	return &chan->nextChan;
}

static inline SndChannelPtr* PrevOf(SndChannelPtr chan)
{
	return &GetEx(chan).prevChan;
}

static void Link(SndChannelPtr chan)
{
	if (!headChan) {
		*NextOf(chan) = nullptr;
	}
	else {
		assert(nullptr == *PrevOf(headChan));
		*PrevOf(headChan) = chan;
		*NextOf(chan) = headChan;
	}

	headChan = chan;
	*PrevOf(chan) = nullptr;

	nManagedChans++;
}

static void Unlink(SndChannelPtr chan)
{
	if (headChan == chan)
		headChan = *NextOf(chan);

	if (*PrevOf(chan))
		*NextOf(*PrevOf(chan)) = *NextOf(chan);

	*PrevOf(chan) = nullptr;
	*NextOf(chan) = nullptr;

	nManagedChans--;
}

//-----------------------------------------------------------------------------
// Sound Manager

OSErr GetDefaultOutputVolume(long* stereoLevel)
{
	unsigned short g = (unsigned short)(cmixer::GetMasterGain() / 256.0);
	*stereoLevel = (g << 16) | g;
	return noErr;
}

// See IM:S:2-139, "Controlling Volume Levels".
OSErr SetDefaultOutputVolume(long stereoLevel)
{
	unsigned short left  = 0xFFFF & stereoLevel;
	unsigned short right = 0xFFFF & (stereoLevel >> 16);
	if (right != left)
		TODOMINOR2("setting different volumes for left & right is not implemented");
	LOG << left / 256.0 << "\n";
	cmixer::SetMasterGain(left / 256.0);
	return noErr;
}

// IM:S:2-127
OSErr SndNewChannel(SndChannelPtr* chan, short synth, long init, SndCallBackProcPtr userRoutine)
{
	if (synth != sampledSynth) {
		TODO2("unimplemented synth type " << sampledSynth);
		return unimpErr;
	}

	//---------------------------
	// Do allocs

	bool allocatedByPomme = false;

	auto impl = new ChannelEx;

	if (!*chan) {
		*chan = new SndChannel;
		impl->macChannelStructAllocatedByPomme = true;
	}
	else {
		impl->macChannelStructAllocatedByPomme = false;
	}

	(**chan).firstMod = (Ptr)impl;

	//---------------------------
	// Set up

	Link(*chan);	// Link chan into our list of managed chans
	(**chan).callBack = userRoutine;

	//---------------------------
	// Done

	LOG << "New channel created, total managed channels = " << nManagedChans << "\n";

	return noErr;
}

// IM:S:2-129
OSErr SndDisposeChannel(SndChannelPtr chan, Boolean quietNow)
{
	Unlink(chan);

	auto* ex = &GetEx(chan);

	bool alsoDeleteMacStruct = ex->macChannelStructAllocatedByPomme;
	delete ex;
	chan->firstMod = nullptr;

	if (alsoDeleteMacStruct) {
		delete chan;
	}

	TODOMINOR2("issue flushCmd, quietCmd, etc. (IM: S ");

	return noErr;
}

OSErr SndChannelStatus(SndChannelPtr chan, short theLength, SCStatusPtr theStatus) {
	TODOMINOR();
	return unimpErr;
}

OSErr SndDoImmediate(SndChannelPtr chan, const SndCommand* cmd) {
	TODOMINOR();
	return noErr;
}

OSErr GetSoundHeaderOffset(SndListHandle sndHandle, long* offset) {
	TODOMINOR();
	return unimpErr;
}

OSErr SndStartFilePlay(
	SndChannelPtr						chan,	
	short								fRefNum,
	short								resNum,
	long								bufferSize,
	Ptr									theBuffer,
	/*AudioSelectionPtr*/ void*			theSelection,
	FilePlayCompletionUPP				theCompletion,
	Boolean								async)
{
	if (resNum != 0) {
		TODO2("playing snd resource not implemented yet, resource " << resNum);
		return unimpErr;
	}

	if (!chan) {
		if (async) // async requires passing in a channel
			return badChannel;
		TODO2("nullptr chan for sync play, check IM:S:1-37");
		return unimpErr;
	}

	if (theSelection) {
		TODO2("audio selection record not implemented");
		return unimpErr;
	}

	Pomme::Sound::AudioClip clip = Pomme::Sound::ReadAIFF(Pomme::Files::GetStream(fRefNum));

	if (wip_ugly_stream) {
		delete wip_ugly_stream;
		wip_ugly_stream = nullptr;
	}

	// TODO: 1 stream per channel
	// TODO: don't use new/delete
	// TODO: get rid of the gratuitous buffer copy
	wip_ugly_stream = new cmixer::WavStream(clip.sampleRate, 16, clip.nChannels, std::vector<char>(clip.pcmData));

	wip_ugly_stream->Play();

	if (!async) {
		while (wip_ugly_stream->GetState() != cmixer::CM_STATE_STOPPED) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	TODOMINOR2("plug completion callback");

	return noErr;
}

OSErr SndPauseFilePlay(SndChannelPtr chan) {
	TODOMINOR();
	return unimpErr;
}

OSErr SndStopFilePlay(SndChannelPtr chan, Boolean quietNow) {
	TODOMINOR();
	return unimpErr;
}

NumVersion SndSoundManagerVersion() {
	NumVersion v;
	v.majorRev = 3;
	v.minorAndBugRev = 9;
	v.stage = 0x80;
	v.nonRelRev = 0;
	return v;
}

//-----------------------------------------------------------------------------
// Init Sound Manager

void Pomme::Sound::Init()
{
	cmixer::InitWithSDL();
}

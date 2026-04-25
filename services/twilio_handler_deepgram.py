"""
AI Voice Agent - Twilio bidirectional media stream handler.

Architecture: two concurrent tasks per call
  1. _twilio_loop   : read Twilio WS -> forward mulaw to Deepgram SDK
  2. _pipeline_loop : asyncio.Queue -> ConversationManager -> TTS -> Twilio

Deepgram SDK fires async callbacks which put transcripts in the queue.
put_nowait() is used (not create_task) — safe from any asyncio context.
"""

import asyncio
import audioop
import base64
import json
import os
import time

from fastapi import WebSocket

TTS_CHUNK = 1600  # ~200ms of mulaw @ 8kHz per send


class StreamSession:
    def __init__(self, speech_service):
        self.speech        = speech_service
        self.call_sid      = None
        self.stream_sid    = None
        self.conversation  = None
        self._ws           = None
        self._dg_conn      = None       # Deepgram SDK live connection
        self._muted_until  = 0.0
        self._processing   = False
        self._transcript_q = asyncio.Queue()
        self._filler_mulaw: bytes = b""  # pre-cached "ठीक है जी" for instant playback

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, websocket: WebSocket):
        self._ws = websocket
        await websocket.accept()

        # Twilio sends "connected" before "start" — skip until we get "start"
        msg = None
        try:
            async for raw in websocket.iter_text():
                candidate = json.loads(raw)
                if candidate.get("event") == "start":
                    msg = candidate
                    break
                print(f"[Stream] Pre-start: {candidate.get('event')}")
        except Exception as e:
            print(f"[Stream] Error waiting for start: {e}")
            return

        if msg is None:
            print("[Stream] Never received start event")
            return

        await self._handle_start(msg)

        t_pl = asyncio.create_task(self._pipeline_loop(), name="pl_loop")
        try:
            await self._twilio_loop(websocket)
        except Exception as e:
            from agent.conversation import State
            if not (self.conversation and self.conversation.state == State.DONE):
                print(f"[Stream Error] {self.call_sid}: {e}")
        finally:
            t_pl.cancel()
            await asyncio.gather(t_pl, return_exceptions=True)
            await self._close_deepgram()

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def _handle_start(self, msg: dict):
        from agent.conversation import ConversationManager
        self.call_sid   = msg["start"]["callSid"]
        self.stream_sid = msg["start"]["streamSid"]
        caller_number   = msg["start"].get("customParameters", {}).get("callerNumber", "")
        self.conversation = ConversationManager(caller_phone=caller_number)
        print(f"[Call] {self.call_sid} from {caller_number}")

        await self._open_deepgram()

        # Pre-synthesize filler in background so it's ready before first user turn
        asyncio.create_task(self._cache_filler(), name="cache_filler")

        # Hold mute until greeting finishes; mark event releases it precisely
        self._muted_until = time.monotonic() + 30.0
        greeting = self.conversation.get_greeting()
        print(f"[Agent] {greeting}")
        asyncio.create_task(self._speak(greeting), name="speak_greeting")

    # ------------------------------------------------------------------
    # Twilio loop: forward audio to Deepgram; handle mark/stop
    # ------------------------------------------------------------------

    async def _twilio_loop(self, websocket: WebSocket):
        async for raw in websocket.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            if event == "media":
                # Always forward — never drop during mute.
                # Dropping starves Deepgram WS -> it dies -> no more transcripts.
                if self._dg_conn and msg["media"].get("track", "inbound") == "inbound":
                    chunk = base64.b64decode(msg["media"]["payload"])
                    try:
                        await self._dg_conn.send(chunk)
                    except Exception:
                        pass

            elif event == "mark":
                # Twilio echoes this back exactly when audio finishes on the phone.
                if msg.get("mark", {}).get("name") == "tts_done":
                    self._muted_until = 0.0
                    print("[Audio] Playback complete - mute released")

            elif event == "stop":
                print(f"[Call] Stream stopped: {self.call_sid}")
                break

    # ------------------------------------------------------------------
    # Pipeline loop: consume queue, one turn at a time
    # ------------------------------------------------------------------

    async def _pipeline_loop(self):
        while True:
            try:
                transcript = await self._transcript_q.get()
            except asyncio.CancelledError:
                break
            await self._process(transcript)

    async def _process(self, transcript: str):
        self._processing = True
        try:
            from agent.conversation import State
            # Run LLM processing and filler playback simultaneously.
            # Filler plays instantly (~0.3s) while the LLM thinks — fills the silence.
            response_task = asyncio.create_task(self.conversation.process_turn(transcript))
            await self._play_filler()
            response = await response_task

            if not response:
                print(f"[Pipeline] Empty response for: {transcript!r}")
                return
            print(f"[Agent] {response}")
            await self._speak(response)

            if self.conversation.state == State.DONE:
                await asyncio.sleep(1.5)
                try:
                    await self._ws.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Pipeline error] {e}")
            import traceback; traceback.print_exc()
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # Deepgram SDK connection
    # ------------------------------------------------------------------

    async def _open_deepgram(self):
        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not api_key:
            print("[Deepgram] ERROR: DEEPGRAM_API_KEY not set")
            return
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

            dg = DeepgramClient(api_key)
            self._dg_conn = dg.listen.asyncwebsocket.v("1")

            # Transcript callback — called by SDK when speech_final fires.
            # put_nowait() is safe from any async context; no create_task needed.
            async def _on_transcript(self_inner, result, **kwargs):
                try:
                    alt        = result.channel.alternatives[0]
                    transcript = alt.transcript.strip()
                    if not result.speech_final or not transcript:
                        return
                    now = time.monotonic()
                    if now < self._muted_until:
                        print(f"[STT] Muted ({self._muted_until - now:.1f}s left): {transcript}")
                        return
                    if self._processing:
                        print(f"[STT] Busy: {transcript}")
                        return
                    import re as _re
                    if not _re.search(r'[a-zA-Z\u0900-\u097F]', transcript):
                        return
                    print(f"[STT] {transcript}")
                    self._transcript_q.put_nowait(transcript)
                except Exception as e:
                    print(f"[Deepgram transcript error] {e}")

            self._dg_conn.on(LiveTranscriptionEvents.Transcript, _on_transcript)

            from deepgram import LiveOptions
            options = LiveOptions(
                model="nova-2",
                language="hi",
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
                endpointing=500,
                interim_results=False,
                smart_format=False,
                punctuate=False,
            )
            started = await self._dg_conn.start(options)
            if started:
                print("[Deepgram] Connected")
            else:
                print("[Deepgram] WARNING: start() returned False — check API key / network")
        except Exception as e:
            print(f"[Deepgram] Connection failed: {e}")
            import traceback; traceback.print_exc()
            self._dg_conn = None

    async def _cache_filler(self):
        """Synthesize filler phrase once and store as mulaw bytes."""
        pcm = await self.speech.synthesize("ठीक है जी")
        if pcm:
            self._filler_mulaw = audioop.lin2ulaw(pcm, 2)
            print("[Filler] Pre-cached")

    async def _play_filler(self):
        """Send pre-cached filler audio to Twilio immediately (no mute/mark)."""
        if not self._filler_mulaw or not self.stream_sid:
            return
        for i in range(0, len(self._filler_mulaw), TTS_CHUNK):
            try:
                await self._ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": base64.b64encode(self._filler_mulaw[i:i + TTS_CHUNK]).decode()},
                }))
            except Exception:
                return

    async def _close_deepgram(self):
        if self._dg_conn:
            try:
                await self._dg_conn.finish()
            except Exception:
                pass
            self._dg_conn = None

    # ------------------------------------------------------------------
    # TTS -> Twilio audio
    # ------------------------------------------------------------------

    async def _speak(self, text: str) -> float:
        pcm = await self.speech.synthesize(text)
        if not pcm:
            self._muted_until = 0.0   # TTS failed — let caller speak immediately
            return 0.0

        mulaw    = audioop.lin2ulaw(pcm, 2)
        duration = len(mulaw) / 8000.0
        # Safety fallback; mark event releases mute precisely before this expires
        self._muted_until = time.monotonic() + duration + 2.0

        for i in range(0, len(mulaw), TTS_CHUNK):
            try:
                await self._ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": base64.b64encode(mulaw[i:i + TTS_CHUNK]).decode()},
                }))
            except Exception:
                self._muted_until = 0.0
                return 0.0

        # Mark event — Twilio echoes it back when audio finishes on the phone
        try:
            await self._ws.send_text(json.dumps({
                "event":     "mark",
                "streamSid": self.stream_sid,
                "mark":      {"name": "tts_done"},
            }))
        except Exception:
            pass

        return duration

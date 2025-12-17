import asyncio
import websockets
import json
import logging
import os
import uuid
import re
import threading
import requests
import numpy as np
from faster_whisper import WhisperModel
from freeswitchESL import ESL

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("ConversationalIVR")

LLM_URL = "http://10.16.7.133:8000/test/transcription"
BASE_SAVE_FOLDER = "/usr/local/freeswitch/sounds/en/us/callie/conversationalIVR"
os.makedirs(BASE_SAVE_FOLDER, exist_ok=True)

# Updated model loading per latest reference (medium.en model, CPU, int8)
model = WhisperModel("base.en", device="cpu", compute_type="int8")
logger.info("Loaded faster_whisper base.en model on cpu with int8")

playback_counter = {}
call_hangup_flags = {}
active_websocket_calls = {}

SAMPLE_RATE = 16000
CHUNK_DURATION = 8  # seconds
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_DURATION
CHUNK_SIZE = int(CHUNK_FRAMES * 2)  # 16-bit PCM, 2 bytes per sample

# ESL event listener to flag hangups
def esl_event_listener():
    try:
        con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")
        if not con.connected():
            logger.error("ESL connection failed — cannot monitor hangups.")
            return
        con.events("plain", "CHANNEL_HANGUP")
        logger.info("Subscribed to CHANNEL_HANGUP events for call hangup detection")
        while True:
            e = con.recvEvent()
            if not e:
                continue
            uuid_ = e.getHeader("Caller-Unique-ID") or e.getHeader("Unique-ID")
            if uuid_ and uuid_ in active_websocket_calls:
                
                # --- FIX 1: Explicitly stop the audio stream from FreeSWITCH side on hangup ---
                stop_resp = con.api(f"uuid_audio_stream {uuid_} stop")
                logger.info(f"Stopped audio stream for {uuid_} on hangup: {stop_resp.getBody().strip()}")
                # -----------------------------------------------------------------------------
                
                call_hangup_flags[uuid_] = True
                logger.info(f"Hangup detected for call_id {uuid_}, stopping transcription.")
    except Exception as e:
        logger.error(f"ESL listener crashed: {e}")

threading.Thread(target=esl_event_listener, daemon=True).start()

def uuid_exists(call_uuid):
    try:
        con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")
        if con.connected():
            resp = con.api(f"uuid_exists {call_uuid}")
            return resp.getBody().strip() == "true"
    except Exception as e:
        logger.error(f"Error checking uuid_exists: {e}")
    return False

def wait_for_playback_stop(esl_con, call_uuid):
    esl_con.events("json", "PLAYBACK_STOP")
    while True:
        evt = esl_con.recvEvent()
        if evt and evt.getHeader("Unique-ID") == call_uuid and evt.getHeader("Event-Name") == "PLAYBACK_STOP":
            logger.info(f"Playback finished for call {call_uuid}")
            break

def play_audio_and_transfer(call_uuid, wav_file_path, dest_ext):
    con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")
    if not con.connected():
        logger.error("ESL connection failed for play and transfer")
        return
    con.api(f"uuid_broadcast {call_uuid} {wav_file_path} both")
    wait_for_playback_stop(con, call_uuid)
    
    resp = con.api(f"uuid_transfer {call_uuid} {dest_ext}")
    logger.info(f"Transferred {call_uuid} to {dest_ext}: {resp.getBody().strip()}")
    
    # --- FIX 2: Stop the audio stream after successful transfer ---
    if resp.getBody().strip().startswith("+OK"):
        # Explicitly stop the FreeSWITCH audio stream/WebSocket
        stop_resp = con.api(f"uuid_audio_stream {call_uuid} stop")
        logger.info(f"Stopped audio stream for {call_uuid} after transfer: {stop_resp.getBody().strip()}")
        
        # Set hangup flag to break the Python WebSocket loop
        call_hangup_flags[call_uuid] = True
        logger.info(f"Transfer successful, setting hangup flag for {call_uuid} to close WebSocket.")
    # -------------------------------------------------------------

def parse_multipart_response(resp):
    content_type = resp.headers.get("Content-Type", "")
    logger.info(f"LLM response headers: {content_type}, HTTP {resp.status_code}")
    logger.info(f"LLM raw content length: {len(resp.content)} bytes")

    if "multipart" not in content_type:
        try:
            j = resp.json()
            return {"json": j, "audio": None, "audio_filename": None}
        except Exception:
            return {"json": None, "audio": resp.content, "audio_filename": None}

    # --- ONLY FIXED PART BELOW ---
    match = re.search(r'boundary=\"?([^\";]+)\"?', content_type)
    # --------------------------------

    if not match:
        logger.error("No boundary found in Content-Type.")
        return {"json": None, "audio": None, "audio_filename": None}

    boundary = match.group(1)
    parts = resp.content.split(("--" + boundary).encode())

    json_part, audio_part, filename = None, None, None

    for part in parts:
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        section = part.strip(b"\r\n")
        header_b, _, body = section.partition(b"\r\n\r\n")
        headers = header_b.decode(errors="ignore")

        if "application/json" in headers:
            try:
                json_part = json.loads(body.decode())
                logger.info(f"Found JSON part: {json_part}")
            except Exception as e:
                logger.error(f"JSON parse failed: {e}")

        elif "audio/wav" in headers or "audio/x-wav" in headers:
            m = re.search(r'filename=\"?([^\"]+)\"?', headers)
            filename = m.group(1) if m else "response.wav"
            audio_part = body.strip()
            logger.info(f"Found audio part ({len(audio_part)} bytes, name={filename})")

    return {"json": json_part, "audio": audio_part, "audio_filename": filename}

def send_to_llm(call_id, text):
    try:
        payload = {"call_uuid": call_id, "transcription": text}
        logger.info(f"➡️ Sending transcription to LLM: {payload}")
        resp = requests.post(LLM_URL, json=payload)
        return resp
    except Exception as e:
        logger.error(f"Error sending to LLM: {e}")
        return None

# Helper for filtering short or filler speech
FILLER_WORDS = {
    "the", "a", "an", "um", "uh", "er", "ah", "hm", "hmm",
    "yeah", "yep", "uh-huh", "mm-hmm", "okay", "ok",
    "thank you", "thanks", "thank", "no problem", "sure", "yes", "alright"
}

def is_meaningful_text(text: str) -> bool:
    if not text or len(text.strip()) < 5:
        return False
    cleaned = text.lower().strip()
    if cleaned in FILLER_WORDS:
        return False
    words = cleaned.split()
    if all(word in FILLER_WORDS for word in words):
        return False
    return True

# Audio receiver WS handler
async def audio_receiver(websocket):
    call_id = f"unknown_{uuid.uuid4().hex[:8]}" # Default ID

    try:
        initial_msg = await asyncio.wait_for(websocket.recv(), timeout=10)
    except Exception as e:
        logger.error(f"WebSocket init error: {e}")
        return

    try:
        metadata = json.loads(initial_msg.replace("raw ", ""))
        call_id = metadata.get("call_id", "").strip() or call_id
    except Exception as e:
        logger.warning(f"Invalid metadata: {e}")

    logger.info(f"WebSocket started for call {call_id}")

    active_websocket_calls[call_id] = websocket
    call_hangup_flags[call_id] = False

    TARGET_MAP = {"sales": "5000", "support": "5001", "development": "5002"}

    audio_buffer = bytearray()

    try:
        # Loop breaks on client disconnect, exception, or hangup flag (from ESL/Transfer)
        async for message in websocket:
            if call_hangup_flags.get(call_id):
                logger.info(f"Call {call_id} hangup flagged — stopping loop.")
                break

            if isinstance(message, bytes):
                audio_buffer.extend(message)

                while len(audio_buffer) >= CHUNK_SIZE:
                    chunk_bytes = audio_buffer[:CHUNK_SIZE]
                    audio_buffer = audio_buffer[CHUNK_SIZE:]

                    audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16).copy()
                    audio_np = audio_int16.astype(np.float32) / 32768.0

                    segments, _ = model.transcribe(
                        audio_np,
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=True,
                        word_timestamps=False,
                        language='en',
                        condition_on_previous_text=False,
                    )

                    text = " ".join([s.text.strip() for s in segments if is_meaningful_text(s.text)]).strip()

                    if not text:
                        logger.info(f"No meaningful speech recognized for call {call_id}, skipping LLM call and playback")
                        continue

                    logger.info(f"Whisper recognized: '{text}'")

                    resp = await asyncio.to_thread(send_to_llm, call_id, text)
                    if not resp:
                        logger.error(f"No LLM response for {call_id}")
                        continue

                    if resp.status_code == 200:
                        parsed = parse_multipart_response(resp)
                        llm_json, audio_data = parsed["json"], parsed["audio"]

                        transfer_data = None
                        if llm_json:
                            if "transfer_request" in llm_json:
                                transfer_data = llm_json
                            elif "transfer" in llm_json and isinstance(llm_json["transfer"], dict):
                                transfer_data = llm_json["transfer"]

                        if transfer_data and transfer_data.get("transfer_request"):

                            target = (transfer_data.get("transfer_target") or "").strip().lower()
                            dest_ext = TARGET_MAP.get(target)

                            if dest_ext and audio_data:
                                response_path = os.path.join(BASE_SAVE_FOLDER, f"{call_id}_response.wav")
                                with open(response_path, "wb") as f:
                                    f.write(audio_data)
                                logger.info(f"Saved LLM audio: {response_path} ({len(audio_data)} bytes)")
                                
                                # Calls uuid_transfer and sets hangup flag/stops stream on success
                                await asyncio.to_thread(play_audio_and_transfer, call_id, response_path, dest_ext)

                            elif dest_ext:
                                con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")
                                if con.connected():
                                    resp = con.api(f"uuid_transfer {call_id} {dest_ext}")
                                    logger.info(f"Transferred {call_id} to {dest_ext}: {resp.getBody().strip()}")
                                    
                                    # --- FIX 3: Stop stream and set hangup flag (Non-audio path) ---
                                    if resp.getBody().strip().startswith("+OK"):
                                        stop_resp = con.api(f"uuid_audio_stream {call_id} stop")
                                        logger.info(f"Stopped audio stream for {call_id} after transfer: {stop_resp.getBody().strip()}")
                                        
                                        call_hangup_flags[call_id] = True
                                        logger.info(f"Transfer successful, setting hangup flag for {call_id} to close WebSocket.")
                                    # -----------------------------------------------------------------
                                else:
                                    logger.error("ESL connection failed for transfer.")

                            else:
                                logger.warning(f"Unknown transfer target '{target}' for call {call_id}")

                        else:
                            if audio_data:
                                response_path = os.path.join(BASE_SAVE_FOLDER, f"{call_id}_response.wav")
                                with open(response_path, "wb") as f:
                                    f.write(audio_data)
                                logger.info(f"Saved LLM audio: {response_path} ({len(audio_data)} bytes)")

                                con = ESL.ESLconnection("127.0.0.1", "8021", "ClueCon")
                                if con.connected():
                                    con.api(f"uuid_broadcast {call_id} {response_path} both")
                                    logger.info(f"Played audio to {call_id}")
                                else:
                                    logger.error("ESL connection failed for playback.")

                    else:
                        logger.error(f"LLM HTTP {resp.status_code} for {call_id}")

            else:
                logger.debug(f"Non-bytes message received: {repr(message)}")

    except Exception as e:
        logger.error(f"Error during call loop for {call_id}: {e}")

    finally:
        # --- FIX 4: Explicitly await websocket close to ensure connection cleanup ---
        logger.info(f"Attempting to close WebSocket for call {call_id}")
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error during WebSocket close for {call_id}: {e}")
        # ---------------------------------------------------------------------------

        active_websocket_calls.pop(call_id, None)
        call_hangup_flags.pop(call_id, None)

        # Final transcription attempt for remaining buffer
        if audio_buffer:
            try:
                audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16).copy()
                audio_np = audio_int16.astype(np.float32) / 32768.0

                segments, _ = model.transcribe(
                    audio_np,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                    word_timestamps=False,
                    language='en',
                    condition_on_previous_text=False,
                )

                final_text = " ".join([seg.text.strip() for seg in segments if is_meaningful_text(seg.text)]).strip()
                if final_text:
                    logger.info(f"Final whisper text for {call_id}: {final_text}")
            except Exception as e:
                logger.error(f"Failed final whisper transcription for {call_id}: {e}")

async def main():
    server = await websockets.serve(
        audio_receiver,
        "10.16.7.91",
        8089,
        ping_interval=30,
        ping_timeout=30,
        max_size=None,
    )
    logger.info("Faster-Whisper Conversational IVR server ready at ws://10.16.7.91:8089")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")

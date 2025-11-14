"""
FreeSWITCH client handler for processing transcriptions and managing audio responses.
"""
import os
import glob
import time
import logging
import json
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["freeswitch"])

def make_multipart_response(audio_path: str, llm_text: str, transfer_json: dict, status_code: int = 200):
    """
    Build a multipart/mixed response for FreeSWITCH:
    Part 1 → JSON body (LLM text + transfer info)
    Part 2 → WAV audio bytes
    
    Args:
        audio_path: Path to the audio file
        llm_text: Text response from LLM
        transfer_json: Transfer information
        status_code: HTTP status code (default: 200)
    """

    boundary = "boundary12345"

    # ---- Prepare JSON PART ----
    json_payload = {
        "status": "success",
        "llm_response": llm_text,
        "transfer": transfer_json
    }

    json_part = (
        f"--{boundary}\r\n"
        "Content-Type: application/json\r\n\r\n"
        f"{json.dumps(json_payload)}\r\n"
    ).encode()

    # ---- Prepare AUDIO PART ----
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = (
        f"--{boundary}\r\n"
        "Content-Type: audio/wav\r\n"
        f"Content-Disposition: attachment; filename={os.path.basename(audio_path)}\r\n\r\n"
    ).encode() + audio_bytes + b"\r\n"

    # ---- Final boundary ----
    closing = f"--{boundary}--\r\n".encode()

    # ---- Combine Parts ----
    final_body = json_part + audio_part + closing

    return Response(
        content=final_body,
        media_type=f"multipart/mixed; boundary={boundary}",
        status_code=status_code
    )



class TranscriptionEvent(BaseModel):
    call_uuid: str
    transcription: str

@router.post("/test/transcription")
async def test_transcription(event: TranscriptionEvent):
    """
    Handle test transcription requests.
    
    This endpoint processes a transcription event, generates a response using the LLM,
    converts it to speech, and returns the audio file with metadata.
    """
    from app.main import agent  # Import agent from main to avoid circular imports
    import json
    from datetime import datetime
    
    try:
        # Log the incoming request payload
        request_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "call_uuid": event.call_uuid,
            "transcription": event.transcription,
            "endpoint": "/test/transcription"
        }
        
        logger.info("=== INCOMING REQUEST ===")
        logger.info(json.dumps(request_data, indent=2))
        logger.info("=======================")
        # Process the transcription through the agent
        await agent.handle_transcription(event.dict())
        
        # Find the latest audio file for this UUID
        audio_dir = agent.tts_client.output_dir
        if not os.path.exists(audio_dir):
            logger.error(f"Audio directory not found: {audio_dir}")
            return make_multipart_response(
                json_payload = {
                    "status": "error",
                    "message": "Audio directory not found",
                    "details": {
                        "directory": audio_dir,
                        "error": "Directory does not exist"
                    }
                }
            )
        
        # Find all files matching the pattern for this UUID
        # Pattern matches files like "response:01_uuid_timestamp.wav"
        # pattern = os.path.join(audio_dir, f"response:*_{event.call_uuid}_*.wav")
        pattern = os.path.join(audio_dir, f"response_*_{event.call_uuid}_*.wav")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            logger.warning(f"No audio files found for UUID: {event.call_uuid}")
            return make_multipart_response(
                audio_path="",
                llm_text="No audio file found for the given call UUID",
                transfer_json={
                    "status": "not_found",
                    "message": "No audio file generated for the given call UUID",
                    "details": {
                        "call_uuid": event.call_uuid,
                        "pattern_used": pattern,
                        "suggestion": "Check if the TTS service generated any files"
                    }
                },
                status_code=404
            )
        
        # Get the most recently modified file
        latest_file = max(matching_files, key=os.path.getmtime)
        
        # Verify file exists and has content
        if not os.path.exists(latest_file) or os.path.getsize(latest_file) == 0:
            logger.error(f"Audio file is empty or doesn't exist: {latest_file}")
            return make_multipart_response(
                status_code=500,
                json_payload={
                    "status": "error",
                    "message": "Generated audio file is invalid or empty",
                    "details": {
                        "file_path": latest_file,
                        "file_exists": os.path.exists(latest_file),
                        "file_size_bytes": os.path.getsize(latest_file) if os.path.exists(latest_file) else 0
                    }
                }
            )
        
        # Get the LLM response from the conversation history
        conversation_history = agent.conversation.get_history(event.call_uuid)
        llm_response = next((msg['content'] for msg in reversed(conversation_history) 
                            if msg['role'] == 'assistant'), None)
        # --- STEP 2: Intent extraction ---
        import re

        def extract_intent(text: str):
            match = re.search(r"<intent>(.*?)</intent>", text, re.IGNORECASE)
            if not match:
                return "none"
            return match.group(1).strip().lower()

        intent = extract_intent(llm_response)

        transfer_map = {
            "sales": ("sales", True),
            "support": ("support", True),
            "development": ("development", True),
            "none": ("none", False)
        }

        queue, should_transfer = transfer_map.get(intent, ("none", False))

        transfer_json = {
            "transfer_request": should_transfer,
            "transfer_target": queue
        }
        logger.info(f"Transfer JSON: {transfer_json}")
        
        # Prepare response metadata
        import json
        import base64
        
        file_metadata = {
            "status": "success",
            "file_details": {
                "filename": os.path.basename(latest_file),
                "size_bytes": os.path.getsize(latest_file),
                "last_modified": time.ctime(os.path.getmtime(latest_file)),
                "call_uuid": event.call_uuid
            },
            "conversation": {
                "message_count": len(conversation_history),
                "last_interaction": time.ctime()
            }
        }
        
        # Encode metadata as base64 to safely include in headers
        metadata_str = json.dumps(file_metadata)
        encoded_metadata = base64.urlsafe_b64encode(metadata_str.encode()).decode()
        
        # Log the successful response (without the full LLM response to keep logs clean)
        logger.info(f"Sending audio file. Size: {os.path.getsize(latest_file)} bytes, "
                   f"Call UUID: {event.call_uuid}")
        
        # Return the audio file with minimal, safe headers
        # response = FileResponse(
        #     latest_file, 
        #     media_type="audio/wav",
        #     filename=os.path.basename(latest_file),
        #     headers={
        #         "Content-Disposition": f"attachment; filename={os.path.basename(latest_file)}",
        #         "X-File-Metadata": encoded_metadata,
        #         "X-Status": "success"
        #     }
        # )
        # return response

        return make_multipart_response(
            audio_path=latest_file,
            llm_text=llm_response,
            transfer_json=transfer_json
        )
        
    except Exception as e:
        logger.exception(f"Error processing transcription: {str(e)}")
        return make_multipart_response(
            status_code=500,
            json_payload={
                "status": "error",
                "message": "Internal server error processing transcription",
                "details": str(e)
            }
        )

"""
FreeSWITCH client handler for processing transcriptions and managing audio responses.
"""
import os
import glob
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["freeswitch"])

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
            raise HTTPException(
                status_code=404,
                detail={
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
        pattern = os.path.join(audio_dir, f"response:*_{event.call_uuid}_*.wav")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            logger.warning(f"No audio files found for UUID: {event.call_uuid}")
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "not_found",
                    "message": "No audio file generated for the given call UUID",
                    "details": {
                        "call_uuid": event.call_uuid,
                        "pattern_used": pattern,
                        "suggestion": "Check if the TTS service generated any files"
                    }
                }
            )
        
        # Get the most recently modified file
        latest_file = max(matching_files, key=os.path.getmtime)
        
        # Verify file exists and has content
        if not os.path.exists(latest_file) or os.path.getsize(latest_file) == 0:
            logger.error(f"Audio file is empty or doesn't exist: {latest_file}")
            raise HTTPException(
                status_code=500,
                detail={
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
        
        # Prepare response metadata
        file_metadata = {
            "status": "success",
            "file_details": {
                "filename": os.path.basename(latest_file),
                "size_bytes": os.path.getsize(latest_file),
                "last_modified": time.ctime(os.path.getmtime(latest_file)),
                "call_uuid": event.call_uuid
            },
            "llm_response": llm_response,
            "conversation": {
                "message_count": len(conversation_history),
                "last_interaction": time.ctime()
            }
        }
        
        # Log the successful response
        logger.info(f"Sending audio file with metadata: {file_metadata}")
        
        # Return the audio file with additional headers
        response = FileResponse(
            latest_file, 
            media_type="audio/wav",
            filename=os.path.basename(latest_file),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(latest_file)}",
                "X-File-Metadata": str(file_metadata),
                "X-LLM-Response": str(llm_response)[:500] 
            }
        )
        return response
        
    except Exception as e:
        logger.exception(f"Error processing transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error processing transcription",
                "details": str(e)
            }
        )

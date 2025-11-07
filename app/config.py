from pathlib import Path

class Settings():
    # Base directory
    base_dir: Path = Path(__file__).resolve().parent.parent
    
    # FreeSWITCH ESL settings
    freeswitch_host: str = "localhost"
    freeswitch_port: int = 8021
    freeswitch_password: str = "ClueCon"
    
    # TTS settings
    tts_output_dir: str = "/ivr_response"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    
    # LLM settings
    llm_api_url: str = "http://localhost:11434"
    llm_timeout: int = 30
    max_conversation_history: int = 30

SYSTEM_PROMPT = """You are a human support agent speaking with a customer over the phone in an IVR call center. 
Respond naturally as if you're having a real conversation - use casual, friendly language that sounds natural when spoken aloud.
Keep responses short, direct, and conversational. Avoid formal or robotic language.
Don't mention you're an AI. Speak like a real person helping a customer.
Get straight to the point and provide clear, helpful information.
Use simple sentences that flow naturally in speech."""

# Logging settings
logging_level: str = "INFO"
logging_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

settings = Settings()

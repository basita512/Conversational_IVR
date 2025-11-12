# Conversational IVR

A lightweight API service that integrates LLM and TTS for generating natural-sounding voice responses to customer queries.

## Features

- REST API endpoint for processing transcriptions
- LLM integration for intelligent response generation
- Coqui TTS for natural speech synthesis
- Audio format conversion (8kHz mono WAV)
- Conversation history management
- Async/await architecture
- Detailed request/response logging

## Project Structure

```
app/
├── main.py              # FastAPI application setup
├── config.py            # Application configuration
├── freeswitch_client.py # API endpoints for FreeSWITCH integration
├── llm_client.py        # LLM API client
├── tts_client.py        # Coqui TTS integration
└── conversation.py      # Conversation history management
```

## API Endpoints

### Process Transcription

```
POST /test/transcription
```

**Request Body:**
```json
{
    "call_uuid": "unique-call-identifier",
    "transcription": "Customer's spoken text"
}
```

**Response:**
- Returns audio file with speech response
- Includes metadata in response headers:
  - `X-File-Metadata`: JSON string with file details
  - `X-LLM-Response`: The generated text response

## Prerequisites

1. Python 3.8+
2. LLM API server running (default: http://localhost:11434/api/chat)
3. Required Python packages (install via `pip install -r requirements.txt`)

## Configuration

Edit `app/config.py` to customize:
- TTS model and output directory
- LLM API URL and timeout settings
- Conversation history length

## Running the Service

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

3. The API will be available at `http://localhost:8000`

## Testing

Use the test endpoint with a tool like curl:

```bash
curl -X POST "http://localhost:8000/test/transcription" \
     -H "Content-Type: application/json" \
     -d '{"call_uuid": "test123", "transcription": "Hello, how are you?"}'
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd support-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your settings in `.env`:
```bash
# FreeSWITCH ESL Connection
FREESWITCH_HOST=your-freeswitch-host
FREESWITCH_PORT=8021
FREESWITCH_PASSWORD=your-password

# LLM API Settings
LLM_API_URL=your-llm-api-url
```

3. Create TTS output directory:
```bash
mkdir -p /tmp/tts  # Linux/Mac
mkdir %TEMP%\tts   # Windows
```

## Running

Start the support agent:
```bash
python -m app.main
```

The agent will:
1. Connect to FreeSWITCH ESL
2. Initialize the TTS engine
3. Start listening for transcription events
4. Process events with the LLM
5. Generate and play responses

## Workflow

1. Customer calls -> FreeSWITCH answers
2. FreeSWITCH (mod_vosk) transcribes speech
3. ESL event sent to Support Agent
4. Support Agent:
   - Processes text with LLM
   - Generates speech with TTS
   - Sends playback command to FreeSWITCH
5. FreeSWITCH plays response to customer

## Error Handling

The agent handles various error scenarios:
- ESL connection failures
- LLM API errors
- TTS generation issues
- Audio conversion problems

All errors are logged with appropriate severity levels.

## WebSocket Message Format

**Client sends:** Binary audio data (16kHz, mono, PCM)

**Server sends:**

Connection:
```json
{"type": "connection", "status": "connected", "message": "Ready to receive audio"}
```

Partial transcript:
```json
{"type": "partial_transcript", "text": "partial text..."}
```

Final transcript:
```json
{"type": "final_transcript", "text": "complete text"}
```

LLM response:
```json
{
  "type": "llm_response",
  "text": "AI response",
  "user_message": "original text"
}
```

Error:
```json
{"type": "error", "message": "error description"}
```

## Llama 3 API Format

**Request (first message):**
```json
{
  "system_prompt": "You are a helpful assistant...",
  "user_message": "transcribed text"
}
```

**Request (with conversation history):**
```json
{
  "system_prompt": "You are a helpful assistant...",
  "conversation_history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ],
  "user_message": "follow-up question"
}
```

**Response:**
```json
{
  "response": "AI response text"
}
```

## Audio Requirements

- Sample Rate: 16000 Hz
- Channels: 1 (Mono)
- Format: PCM 16-bit
- Encoding: Raw audio bytes

## Environment Variables

- `LLAMA3_API_URL` - Llama 3 endpoint (default: http://localhost:8001/llama3)
- `LLAMA3_TIMEOUT` - Request timeout in seconds (default: 30)
- `MAX_CONVERSATION_HISTORY` - Max messages to keep in history (default: 10)
- `LOG_LEVEL` - Logging level (default: INFO)

## Dependencies

- fastapi
- uvicorn
- websockets
- httpx
- vosk
- pydantic

See `requirements.txt` for versions.

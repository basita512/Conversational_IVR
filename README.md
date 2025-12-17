# Conversational IVR with FreeSWITCH Integration

A production-ready voice assistant system that combines FreeSWITCH telephony, speech recognition, and AI-powered conversation management to create natural voice interactions. The system supports real-time audio streaming, intelligent conversation handling, and seamless call transfers.

## 🚀 Features

### Core Components
- **RAG System**: Advanced Retrieval-Augmented Generation using ChromaDB and sentence-transformers
- **FreeSWITCH Integration": Seamless telephony integration with real-time audio streaming
- **Speech Recognition**: Powered by Faster-Whisper for accurate speech-to-text
- **LLM Integration**: Intelligent conversation handling with customizable AI models
- **Real-time Audio Processing**: Efficient handling of audio streams with WebSockets
- **Conversation Management**: Context-aware dialogue handling with conversation history
- **Call Transfer**: Intelligent call routing based on conversation context
- **Multi-threaded Architecture**: High-performance handling of concurrent calls
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

### Technical Highlights
- **WebSocket-based Audio Streaming**: Low-latency audio processing
- **Efficient Memory Management**: Chunk-based audio processing for optimal performance
- **Modular Design**: Easy to extend and customize
- **Configuration Management**: Environment-based configuration system
- **Error Handling**: Robust error handling and recovery mechanisms

## 🧠 RAG System (Retrieval-Augmented Generation)

The system implements a sophisticated RAG pipeline that combines the power of large language models with your organization's specific knowledge base for accurate, up-to-date responses.

### RAG Architecture

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│                 │    │                   │    │                 │
│  Document Store │───▶│  Text Processing  │───▶│  Vector Store   │
│  (.docx files)  │    │  & Chunking      │    │  (ChromaDB)     │
└─────────────────┘    └───────────────────┘    └────────┬────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│                 │    │                   │    │                 │
│  User Query    │───▶│  Query Processing  │◀──▶│  LLM (Llama 3)  │
│  (Speech/Text)  │    │  & Retrieval      │    │  with RAG       │
└─────────────────┘    └───────────────────┘    └─────────────────┘
```

### Data Folder Structure
```
Data/
├── Modelfile          # LLM system prompt and RAG instructions
├── inject_kb.py       # Script to process and vectorize documents
└── vector_db/         # Vector store directory (created automatically)
```

### Setting Up the RAG System

1. **Prepare Your Documents**
   - Place your `.docx` files in the `Data/` folder
   - The system will automatically process and chunk the content
   - Supported formats: `.docx` (Word documents)

2. **Install RAG Dependencies**
   ```bash
   pip install python-docx chromadb sentence-transformers nltk
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

3. **Initialize the RAG System**
   ```bash
   cd Data
   python inject_kb.py
   ```
   This will:
   - Process all documents in the `KB_FILES` list
   - Split content into semantic chunks (200 tokens each)
   - Generate embeddings using `all-MiniLM-L6-v2` model
   - Store vectors in ChromaDB for efficient retrieval

### Advanced Retrieval System

The system implements a sophisticated two-stage retrieval process with re-ranking for highly accurate context selection:

1. **First-Stage Retrieval**
   - Uses `BAAI/bge-large-en-v1.5` for initial document embedding
   - Retrieves top-N (default: 10) most similar chunks using cosine similarity
   - Optimized for recall to ensure relevant chunks aren't missed

2. **Second-Stage Re-ranking**
   - Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise relevance scoring
   - Re-ranks initial results based on query-document interaction
   - Selects top-K (default: 1) most relevant chunks for final context

### Customization Options

#### Retrieval Parameters
- **Initial Retrieval Count**: Adjust `initial_results` in `_retrieve_context()`
  ```python
  # In app/llm_client.py
  context = await self._retrieve_context(query, initial_results=15, final_results=3)
  ```

- **Final Results Count**: Control how many chunks are used for context
  ```python
  # In app/llm_client.py
  context = await self._retrieve_context(query, initial_results=10, final_results=2)
  ```

#### Model Customization
- **Embedding Model**: 
  - Current: `BAAI/bge-large-en-v1.5`
  - Can be replaced with any SentenceTransformer model
  - Update in `llm_client.py`:
    ```python
    model = SentenceTransformer("your-model-name")
    ```

- **Re-ranker Model**:
  - Current: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Can be replaced with other cross-encoder models
  - Update in `llm_client.py`:
    ```python
    reranker = CrossEncoder('your-cross-encoder-model')
    ```

## 🔧 Customizing the LLM Model

You can customize the LLM's behavior by modifying the `Data/Modelfile` and creating a custom model using Ollama.

### 1. Edit the Modelfile

Open `Data/Modelfile` in a text editor and modify the system prompt and model configuration as needed. For example:

```dockerfile
# Specify the base model (replace with your preferred model)
FROM llama2

# Set system prompt
SYSTEM """
You are a helpful AI assistant. Provide accurate, helpful responses based on the provided context.
- Keep responses concise and professional
- If you don't know the answer, say so
- Always maintain a helpful and friendly tone
"""

# Set model parameters
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9
```

### 2. Create and Install the Custom Model

1. Open a terminal in the `Data` directory
2. Run the following command to create and install your custom model:

```bash
cd Data
ollama create custom-llm -f Modelfile
```

3. Pull the base model (if not already downloaded):
```bash
ollama pull llama2  # or your chosen base model
```

4. Start using your custom model by referencing it as `custom-llm` in your API calls.

### 3. Update Configuration

Update your application's configuration to use the new model:

```python
# In your .env file
LLM_MODEL=custom-llm
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Conversational_IVR
```

### 2. Set Up Virtual Environment
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Main application
pip install -r requirements.txt

# FreeSWITCH client (only needed on FreeSWITCH server)
cd freeswitch
pip install -r requirements.txt
cd ..
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following variables:
```env
# Main Application
LLM_API_URL=http://localhost:8000/test/transcription
LOG_LEVEL=INFO

# FreeSWITCH ESL Configuration
FREESWITCH_HOST=localhost
FREESWITCH_PORT=8021
FREESWITCH_PASSWORD=ClueCon

# TTS Configuration
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
TTS_OUTPUT_DIR=./tts_output
```

## ⚙️ Configuration

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Update the configuration** in `.env`:
   ```env
   # FreeSWITCH ESL Connection
   FREESWITCH_HOST=your-freeswitch-host
   FREESWITCH_PORT=8021
   FREESWITCH_PASSWORD=your-password

   # LLM API Settings
   LLM_API_URL=your-llm-api-url
   
   # TTS Settings
   TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
   TTS_OUTPUT_DIR=./tts_output
   
   # Application Settings
   MAX_CONVERSATION_HISTORY=10
   LOG_LEVEL=INFO
   ```

3. **Create TTS output directory**
   ```bash
   # Linux/Mac
   mkdir -p /tmp/tts
   
   # Windows
   mkdir %TEMP%\tts
   ```

## 🚀 Running the System

### 1. Start the Main Application
```bash
# In the project root directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the FreeSWITCH Client
```bash
# In a separate terminal
cd freeswitch
python ivr_client.py
```

### 3. Make a Test Call
Dial one of the configured extensions (5000, 5001, or 5002) from any SIP phone registered to your FreeSWITCH server.

### 4. Monitor Logs
Check the application logs for real-time debugging:
```bash
tail -f logs/app.log
```

## 🌐 System Architecture

### Component Diagram
```
+----------------+     +------------------+     +----------------+
|                |     |                  |     |                |
|  FreeSWITCH    |<--->|  IVR Client     |<--->|  Main App      |
|  (SIP/RTP)     |     |  (ivr_client.py)|     |  (FastAPI)     |
|                |     |                  |     |                |
+----------------+     +------------------+     +--------+-------+
                                                       |
                                                       |
                                          +------------v-------------+
                                          |                          |
                                          |  LLM API                 |
                                          |  (e.g., Ollama, OpenAI)  |
                                          |                          |
                                          +--------------------------+
```

### Data Flow
1. Call arrives at FreeSWITCH
2. `ivr.lua` script handles the call and sets up audio streaming
3. Audio is streamed to `ivr_client.py` via WebSocket
4. Speech is transcribed using Faster-Whisper
5. Transcription is sent to the main application
6. Main application processes the text with LLM
7. Response is converted to speech and sent back to the caller

## 📡 API Endpoints

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

**Response Headers:**
- `X-File-Metadata`: JSON string with audio file details
- `X-LLM-Response`: Generated text response from LLM

**Response Body:**
- Audio file in WAV format (16kHz, mono)


## 📚 Additional Resources

- [FreeSWITCH Documentation](https://freeswitch.org/confluence/display/FREESWITCH/FreeSWITCH+Documentation)
- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📡 WebSocket API

### Connection
- **Endpoint:** `ws://localhost:8000/ws`
- **Protocol:** Binary WebSocket

### Message Types

#### Client → Server
- Binary audio data (16kHz, mono, PCM)

#### Server → Client
```json
{
  "type": "connection",
  "status": "connected",
  "message": "Ready to receive audio"
}

{
  "type": "partial_transcript",
  "text": "partial transcription..."
}

{
  "type": "final_transcript",
  "text": "complete transcription"
}

{
  "type": "llm_response",
  "text": "AI generated response",
  "user_message": "original user message"
}

{
  "type": "error",
  "message": "error description"
}
```

## 🧪 Testing

Test the transcription endpoint:

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
     -H "Content-Type: application/json" \
     -d '{"call_uuid": "test123", "transcription": "Hello, how can I help you today?"}'
```

## Audio Requirements

- Sample Rate: 16000 Hz
- Channels: 1 (Mono)
- Format: PCM 16-bit
- Encoding: Raw audio bytes

## 🛠️ System Requirements

### Main Application
- Python 3.8+
- FreeSWITCH 1.10+ with ESL enabled
- LLM API endpoint (compatible with OpenAI-like API)
- Coqui TTS server (for text-to-speech synthesis)

### FreeSWITCH Requirements
- mod_audio_stream
- mod_esl
- mod_vlc (optional, for additional codec support)
- Properly configured audio codecs (PCM, OPUS, etc.)

### Hardware Requirements
- CPU: 8+ cores recommended
- RAM: 8GB+ (16GB recommended for production)
- Storage: SSD recommended for better I/O performance
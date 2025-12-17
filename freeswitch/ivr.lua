-- conversational_ivr.lua
-- FreeSWITCH Lua script for mod_audio_stream + Vosk + NLU

freeswitch.consoleLog("INFO", "Starting conversational IVR\n")

-- Answer the call
session:answer()

-- Sleep for 1 second
session:sleep(1000)
--session:setVariable("tts_engine", "flite")
--session:setVariable("tts_voice", "slt")
--session:execute("speak", "Hello! Welcome to Zenius Voice Assistance. How can I help you today?")

--session:speak("Hello, welcome to the FreeSWITCH conversational IVR.")

-- Play welcome prompt (optional)
session:streamFile("/usr/local/freeswitch/sounds/en/us/callie/welcome_zenius.wav")

-- Get the UUID of this call
local uuid = session:get_uuid()
freeswitch.consoleLog("INFO", "Call UUID: " .. uuid .. "\n")

-- Create API object
local api = freeswitch.API()

-- WebSocket URL of your Vosk/AI server
local wss_url = "ws://10.16.7.91:8089"

-- Prepare metadata JSON string including call ID
local metadata = string.format('{"call_id":"%s"}', uuid)

-- Command to start streaming 16 kHz raw PCM with metadata
local cmd = string.format("%s start %s mono 16000 raw %s", uuid, wss_url, metadata)
freeswitch.consoleLog("INFO", "Executing: uuid_audio_stream " .. cmd .. "\n")

-- Execute the streaming command
api:execute("uuid_audio_stream", cmd)

-- Keep the session alive until hangup
while session:ready() do
    session:sleep(1000)
end

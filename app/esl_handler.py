"""
ESL event handler for FreeSWITCH events.
"""
try:
    import ESL
except Exception:
    ESL = None
    # optionally provide a lightweight stub class for local dev
    
from typing import Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ESLHandler:
    def __init__(self, host: str, port: int, password: str):
        """Initialize ESL connection handler.

        Args:
            host: FreeSWITCH ESL host
            port: FreeSWITCH ESL port
            password: FreeSWITCH ESL password
        """
        self.host = host
        self.port = port
        self.password = password
        self.con = None
        self.event_handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """Establish connection to FreeSWITCH ESL."""
        try:
            self.con = ESL.ESLconnection(self.host, self.port, self.password)
            if not self.con.connected():
                logger.error("Failed to connect to FreeSWITCH ESL")
                return False
            logger.info("Successfully connected to FreeSWITCH ESL")
            return True
        except Exception as e:
            logger.error(f"Error connecting to FreeSWITCH ESL: {e}")
            return False

    def register_handler(self, event_name: str, handler: Callable):
        """Register a handler for a specific event.

        Args:
            event_name: Name of the ESL event
            handler: Callback function to handle the event
        """
        self.event_handlers[event_name] = handler

    async def start_listening(self):
        """Start listening for ESL events."""
        if not self.con or not self.con.connected():
            logger.error("Not connected to FreeSWITCH ESL")
            return

        # Subscribe to the CUSTOM events (for transcriptions)
        self.con.events('plain', 'CUSTOM')
        
        while self.con.connected():
            event = self.con.recvEvent()
            if event:
                await self._handle_event(event)

    async def _handle_event(self, event: Any):
        """Handle incoming ESL events.

        Args:
            event: ESL event object
        """
        event_name = event.getHeader('Event-Name')
        event_subclass = event.getHeader('Event-Subclass')

        # Handle transcription events
        if event_name == 'CUSTOM' and event_subclass == 'mod_vosk::transcription':
            call_uuid = event.getHeader('Unique-ID')
            transcription = event.getHeader('Transcription')
            
            handler = self.event_handlers.get('transcription')
            if handler:
                await handler({
                    'call_uuid': call_uuid,
                    'transcription': transcription
                })
            else:
                logger.warning("No handler registered for transcription events")

    async def send_playback_command(self, uuid: str, file_path: str) -> bool:
        """Send a playback command to FreeSWITCH.

        Args:
            uuid: Call UUID
            file_path: Path to the audio file to play

        Returns:
            bool: True if command was sent successfully
        """
        if not self.con or not self.con.connected():
            logger.error("Not connected to FreeSWITCH ESL")
            return False

        try:
            cmd = f'uuid_displace {uuid} start {file_path} 0 mux'
            result = self.con.api(cmd)
            if result and result.getBody() == '+OK':
                logger.info(f"Successfully sent playback command for {file_path}")
                return True
            else:
                logger.error(f"Failed to send playback command: {result.getBody() if result else 'No response'}")
                return False
        except Exception as e:
            logger.error(f"Error sending playback command: {e}")
            return False

    def disconnect(self):
        """Disconnect from FreeSWITCH ESL."""
        if self.con:
            self.con.disconnect()
            logger.info("Disconnected from FreeSWITCH ESL")
"""
Conversation history management.
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Conversation:
    def __init__(self):
        """Initialize conversation manager."""
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, uuid: str, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            uuid: Call UUID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        if uuid not in self.conversations:
            self.conversations[uuid] = []
        
        self.conversations[uuid].append({
            'role': role,
            'content': content
        })

        # Keep only the last 10 messages to avoid context length issues
        if len(self.conversations[uuid]) > 30:
            self.conversations[uuid] = self.conversations[uuid][-30:]

    def get_history(self, uuid: str) -> List[Dict[str, str]]:
        """Get conversation history for a specific call.

        Args:
            uuid: Call UUID

        Returns:
            List of message dictionaries
        """
        return self.conversations.get(uuid, [])

    def clear_history(self, uuid: str):
        """Clear conversation history for a specific call.

        Args:
            uuid: Call UUID
        """
        if uuid in self.conversations:
            del self.conversations[uuid]
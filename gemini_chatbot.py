"""
AI-Powered Chatbot for Decision Assistance
Uses Gemini API for natural language understanding and contextual responses
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

# Load API key from .env
load_dotenv()


class LanguageDetector:
    """Simple language detection for common languages"""
    
    LANGUAGE_PATTERNS = {
        "es": r"\b(como|qué|por|para|si|es|está|tú|yo|nosotros)\b",  # Spanish
        "fr": r"\b(comment|quoi|pour|si|c'est|je|vous|nous|ca)\b",  # French
        "de": r"\b(wie|was|für|wenn|ist|ich|du|wir|das)\b",  # German
        "pt": r"\b(como|o|que|para|se|é|você|eu|nós)\b",  # Portuguese
        "it": r"\b(come|cosa|per|se|è|io|tu|noi|che)\b",  # Italian
        "ja": r"[\u3040-\u309F\u30A0-\u30FF]",  # Japanese
        "ko": r"[\uAC00-\uD7AF]",  # Korean
        "zh": r"[\u4E00-\u9FFF]",  # Chinese
        "hi": r"[\u0900-\u097F]",  # Hindi
        "ar": r"[\u0600-\u06FF]",  # Arabic
    }
    
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """
        Detect language from text.
        Returns language code or None if English (default).
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for lang_code, pattern in LanguageDetector.LANGUAGE_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return lang_code
        
        return None  # Default to English


class AIDecisionChatbot:
    """
    An AI chatbot that understands natural language and provides context-aware
    decision assistance based on user's decision history.
    Supports multilingual interaction.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize the chatbot with Gemini API"""
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.chat_history = []
        self.user_context = None
        self.conversation_state = None
        self.detected_language = None
        self.language_names = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "hi": "Hindi",
            "ar": "Arabic",
        }

    def set_user_context(self, decisions_summary: str, constraints_summary: str):
        """
        Set user context based on their decision history.
        This helps the AI provide personalized suggestions.
        """
        self.user_context = {
            "decisions": decisions_summary,
            "constraints": constraints_summary,
            "updated_at": datetime.now().isoformat(),
        }

    def _build_system_prompt(self) -> str:
        """Build a comprehensive system prompt for the chatbot"""
        base_prompt = """You are an intelligent, empathetic Decision Assistant AI powered by Google Gemini. Your role is to help users make better decisions by:

1. **Understanding Context**: Listen carefully to what users are considering and understand their goals, constraints, and concerns.

2. **Providing Insights**: Based on their decision history, help them identify patterns, recurring constraints, and lessons learned from past decisions.

3. **Natural Conversation**: Engage in friendly, natural language conversations. Ask clarifying questions to understand their situation better.

4. **Language Flexibility**: Respond in the user's preferred language or the language they're using to communicate with you. Be culturally sensitive and adapt your communication style.

5. **Actionable Advice**: Provide practical suggestions based on their past experiences and current situation.

6. **Emotional Intelligence**: Recognize when users are facing difficult decisions and be supportive while remaining objective.

7. **Summarization**: When asked, provide concise summaries of decisions and insights.

**Key Capabilities:**
- Suggest solutions based on past decision patterns
- Identify recurring constraints and help address them
- Analyze trade-offs and alternatives
- Detect risks and opportunities
- Provide motivation and clarity
- Support multilingual conversations

**Guidelines:**
- Be conversational, not robotic
- Ask follow-up questions to deepen understanding
- Reference their past decisions when relevant
- Always respect privacy and decision autonomy
- Admit when you need more information
- Prioritize the user's values and preferences
- Use simple, clear language
- Be encouraging and supportive
"""

        if self.user_context:
            base_prompt += f"\n\n**User's Decision Context:**\n{self.user_context['decisions']}"
            base_prompt += f"\n\n**Common Constraints:**\n{self.user_context['constraints']}"

        return base_prompt

    def chat(self, user_message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        Maintains conversation history for context.
        Automatically detects and responds in user's language.
        """
        try:
            # Detect language
            detected = LanguageDetector.detect_language(user_message)
            if detected:
                self.detected_language = detected
            
            # Add user message to history
            self.chat_history.append({"role": "user", "parts": [user_message]})

            # Build the system prompt
            system_prompt = self._build_system_prompt()
            
            # Add language instruction if non-English detected
            if self.detected_language:
                lang_name = self.language_names.get(self.detected_language, "the user's language")
                system_prompt += f"\n\nIMPORTANT: The user is communicating in {lang_name}. Please respond entirely in {lang_name}."

            # Create a new chat session with history
            chat = self.model.start_chat(history=[])

            # Send system prompt first
            chat.send_message(system_prompt)

            # Send conversation history
            for msg in self.chat_history[:-1]:  # All except the current message
                if msg["role"] == "user":
                    chat.send_message(msg["parts"][0])
                else:
                    chat.send_message(msg["parts"][0])

            # Send current user message and get response
            response = chat.send_message(user_message)
            assistant_response = response.text

            # Add assistant response to history
            self.chat_history.append({"role": "assistant", "parts": [assistant_response]})

            return assistant_response

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            return error_msg

    def get_decision_insight(
        self, decision_title: str, goal: str, situation: str, constraints: str = ""
    ) -> str:
        """
        Get AI insight for a specific decision scenario.
        This is a focused query for decision-making support.
        """
        prompt = f"""Based on the user's decision history and the following situation, provide insightful suggestions:

**Decision Title:** {decision_title}
**Goal:** {goal}
**Situation:** {situation}
{"**Constraints:** " + constraints if constraints else ""}

Please provide:
1. A brief assessment of the situation
2. Relevant patterns from their past decisions
3. 2-3 specific recommendations
4. Potential risks to consider
5. Questions they should ask themselves

Keep the response concise but actionable."""

        return self.chat(prompt)

    def analyze_constraints(self, constraints_list: List[str]) -> str:
        """Analyze recurring constraints and suggest strategies"""
        constraints_text = "\n".join(f"- {c}" for c in constraints_list)

        prompt = f"""Looking at these recurring constraints in the user's decisions:

{constraints_text}

Please provide:
1. Pattern identification - what themes do you see?
2. Root cause analysis - why might these constraints keep appearing?
3. Strategic solutions - how could they proactively address these constraints?
4. Prevention strategies - how to avoid similar constraints in future decisions?

Be specific and actionable."""

        return self.chat(prompt)

    def generate_reflection(self, decision_history_text: str) -> str:
        """Generate a reflective analysis of decision-making patterns"""
        prompt = f"""Provide a thoughtful reflection on the user's decision-making patterns based on this history:

{decision_history_text}

Please analyze:
1. **Decision-Making Style** - What's their typical approach?
2. **Strengths** - What do they do well?
3. **Growth Areas** - Where could they improve?
4. **Recurring Themes** - What patterns emerge?
5. **Advice for Future** - What wisdom would you offer?

Be encouraging but honest."""

        return self.chat(prompt)

    def get_multilingual_response(
        self, user_message: str, detected_language: str = None
    ) -> str:
        """
        Get a response that can be in user's language.
        The model will detect language and respond accordingly.
        """
        if detected_language:
            instruction = f"\n\nIMPORTANT: The user is communicating in {detected_language}. Please respond in {detected_language}."
        else:
            instruction = "\n\nRespond in the same language the user is using."

        enhanced_message = user_message + instruction
        return self.chat(enhanced_message)

    def clear_history(self):
        """Clear the conversation history"""
        self.chat_history = []
        self.detected_language = None

    def get_conversation_history(self) -> List[Dict]:
        """Return the full conversation history"""
        return self.chat_history

    def set_conversation_history(self, history: List[Dict]):
        """Set the conversation history (for loading from session state)"""
        self.chat_history = history

    def create_summary(self) -> str:
        """Create a summary of the conversation for reference"""
        if not self.chat_history:
            return "No conversation history yet."

        summary_prompt = """Provide a brief summary of this conversation in 2-3 sentences, highlighting:
- The main decision or topic discussed
- Key insights or recommendations provided
- Any action items or next steps

Keep it concise."""

        return self.chat(summary_prompt)
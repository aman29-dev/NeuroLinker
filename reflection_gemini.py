import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

class GeminiReflectionAI:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.api_key = None
        if enabled:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.api_key = api_key
                genai.configure(api_key=api_key)

    def reflect(self, decision_history_text: str) -> str:
        if not self.enabled or not self.api_key:
            return "Reflection AI disabled or API key missing."

        try:
            # Call Gemini model
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(decision_history_text)
            return response.text
        except Exception as e:
            return f"Reflection AI failed: {str(e)}"
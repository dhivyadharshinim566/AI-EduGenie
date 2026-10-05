import json
import os
import re

try:
    from google import genai
except ImportError:
    genai = None

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = None
        if genai and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    @property
    def configured(self):
        return self.client is not None

    def _generate(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Copy .env.example to .env, "
                "add your Gemini API key, then restart the server."
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text.strip()

    def answer(self, question: str) -> str:
        return self._generate(
            "You are EduGenie, a helpful educational assistant. "
            "Answer the student's question accurately and concisely.\n\n"
            f"Question:\n{question}"
        )

    def summarize(self, text: str) -> str:
        return self._generate(
            "Summarize the following educational passage into clear, concise "
            "study notes. Preserve important facts and context.\n\n"
            f"Passage:\n{text}"
        )

    def learning_path(self, topic: str) -> str:
        return self._generate(
            "Create a personalized learning path for the topic below. "
            "Organize it from beginner to advanced. Include difficulty levels, "
            "step-by-step topics, suggested resources such as videos/articles/books, "
            "and a simple progression plan.\n\n"
            f"Topic:\n{topic}"
        )

    def generate_quiz(self, passage: str, count: int = 3):
        prompt = f"""
Create exactly {count} multiple-choice questions from the passage below.
Return ONLY valid JSON with this structure:
[
  {{
    "question": "string",
    "options": ["A", "B", "C", "D"],
    "answer": "the exact correct option text"
  }}
]
Do not use Markdown fences.

Passage:
{passage}
"""
        raw = self._generate(prompt)
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.I)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                raise ValueError("Quiz response is not a list.")
            return data
        except Exception as exc:
            raise RuntimeError(f"Quiz JSON parsing failed: {exc}\nRaw response: {raw}")

import os

class ExplanationService:
    """
    Uses LaMini-Flan-T5 locally when available, matching the project documentation.
    If the local model is unavailable, it falls back to Gemini so the application
    remains usable after a normal setup.
    """
    def __init__(self, gemini_service):
        self.gemini = gemini_service
        self.pipeline = None
        self.local_available = False
        if os.getenv("USE_LOCAL_EXPLAINER", "true").lower() == "true":
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text2text-generation",
                    model="MBZUAI/LaMini-Flan-T5-783M"
                )
                self.local_available = True
            except Exception:
                self.pipeline = None
                self.local_available = False

    def explain(self, topic: str) -> str:
        prompt = (
            "Explain the following topic in simple language for a student. "
            "Use short paragraphs or bullet points and avoid unnecessary complexity.\n\n"
            f"Topic: {topic}"
        )
        if self.pipeline:
            result = self.pipeline(prompt, max_new_tokens=220, do_sample=False)
            if result and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
        return self.gemini._generate(prompt)

# publisher/analysis_modules/readability_analysis.py

from .base_pov import BasePOV
import textstat
import re

class ReadabilityAnalysis(BasePOV):
    def __init__(self, text):
        super().__init__(text)

    def preprocess_text(self):
        # Normalize and clean the text
        text = self.text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def analyze(self):
        clean_text = self.preprocess_text()
        # Calculate readability score
        try:
            score = textstat.flesch_reading_ease(clean_text)
            # Adjust score to be within 0 to 100
            score = max(0.0, min(score, 100.0))
            score = round(score, 2)
        except Exception:
            score = 0.0
        return {'readability_analysis': score}


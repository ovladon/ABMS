# publisher/analysis_modules/genre_analysis.py

from .base_pov import BasePOV
from transformers import pipeline

class GenreAnalysis(BasePOV):
    def __init__(self, text):
        super().__init__(text)
        # Using a transformer-based zero-shot classifier
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.labels = ["Political Speech", "Scientific", "Historical Document", "Legal", "Story", "Finance", "Entertainment", "Sports"]

    def analyze(self):
        result = self.classifier(self.text, candidate_labels=self.labels)
        # Select the label with the highest score
        genre = result['labels'][0]
        return {'genre_analysis': genre}


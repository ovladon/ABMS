# publisher/analysis_modules/reliability_analysis.py

from .base_pov import BasePOV
from transformers import pipeline

fact_check_pipeline = pipeline("text-classification", model="microsoft/deberta-xlarge-mnli")

class ReliabilityAnalysis(BasePOV):
    def __init__(self, text):
        super().__init__(text)

    def analyze(self):
        result = fact_check_pipeline(self.text[:512])[0]
        reliability_score = result['score'] if result['label'] == 'ENTAILMENT' else 0
        return {'reliability_analysis': reliability_score}


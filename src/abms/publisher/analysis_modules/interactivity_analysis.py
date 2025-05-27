# publisher/analysis_modules/interactivity_analysis.py

from .base_pov import BasePOV
import spacy

nlp = spacy.load('en_core_web_sm')

class InteractivityAnalysis(BasePOV):
    def __init__(self, text):
        super().__init__(text)

    def analyze(self):
        doc = nlp(self.text)
        question_count = sum(1 for sent in doc.sents if sent.text.strip().endswith('?'))
        cta_phrases = ['click here', 'sign up', 'join us', 'contact us', 'learn more', 'subscribe', 'get started', 'buy now']
        cta_count = sum(self.text.lower().count(phrase) for phrase in cta_phrases)
        total_sentences = len(list(doc.sents))
        interactivity_score = (question_count + cta_count) / total_sentences if total_sentences else 0
        return {'interactivity_analysis': interactivity_score}


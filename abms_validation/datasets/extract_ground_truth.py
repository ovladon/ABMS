import pickle
import os
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
import numpy as np

class GroundTruthExtractor:
    def __init__(self):
        pass
    
    def extract_newsgroups_labels(self):
        """Extract category labels from 20newsgroups"""
        with open('/home/vlad/CSML/ISI/Aspect_Based_Metadata_System/Paper_ABMS_latex/Information Processing & Management (Elsevier)/datasets/20 Newsgroups Dataset/newsgroups_train.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Map categories to your genre aspects
        category_map = {
            'alt.atheism': 'Religious',
            'comp.graphics': 'Technical',
            'comp.os.ms-windows.misc': 'Technical',
            'comp.sys.ibm.pc.hardware': 'Technical',
            'comp.sys.mac.hardware': 'Technical',
            'comp.windows.x': 'Technical',
            'misc.forsale': 'Commercial',
            'rec.autos': 'Recreational',
            'rec.motorcycles': 'Recreational',
            'rec.sport.baseball': 'Sports',
            'rec.sport.hockey': 'Sports',
            'sci.crypt': 'Scientific',
            'sci.electronics': 'Scientific',
            'sci.med': 'Scientific',
            'sci.space': 'Scientific',
            'soc.religion.christian': 'Religious',
            'talk.politics.guns': 'Political',
            'talk.politics.mideast': 'Political',
            'talk.politics.misc': 'Political',
            'talk.religion.misc': 'Religious'
        }
        
        ground_truth = []
        texts = []
        for i, text in enumerate(data.data[:1000]):  # Limit to 1000 for speed
            category = data.target_names[data.target[i]]
            mapped_genre = category_map.get(category, 'General')
            ground_truth.append({
                'text': text,
                'genre': mapped_genre,
                'category': category,
                'readability': flesch_reading_ease(text),
                'complexity': flesch_kincaid_grade(text)
            })
        
        return ground_truth
    
    def extract_imdb_sentiment(self):
        """Extract sentiment labels from IMDB"""
        ground_truth = []
        
        # Process positive reviews
        pos_dir = '/home/vlad/CSML/ISI/Aspect_Based_Metadata_System/Paper_ABMS_latex/Information Processing & Management (Elsevier)/datasets/IMDB Movie Reviews/aclImdb/train/pos'
        for filename in os.listdir(pos_dir)[:500]:  # Limit for speed
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                ground_truth.append({
                    'text': text,
                    'sentiment': 'positive',
                    'sentiment_score': 1.0,
                    'readability': flesch_reading_ease(text)
                })
        
        # Process negative reviews
        neg_dir = '/home/vlad/CSML/ISI/Aspect_Based_Metadata_System/Paper_ABMS_latex/Information Processing & Management (Elsevier)/datasets/IMDB Movie Reviews/aclImdb/train/neg'
        for filename in os.listdir(neg_dir)[:500]:  # Limit for speed
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                ground_truth.append({
                    'text': text,
                    'sentiment': 'negative',
                    'sentiment_score': -1.0,
                    'readability': flesch_reading_ease(text)
                })
        
        return ground_truth
    
    def save_ground_truth(self, data, filename):
        with open(f'ground_truth_{filename}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} samples to ground_truth_{filename}.pkl")

# Run extraction
extractor = GroundTruthExtractor()

# Extract and save ground truth
newsgroups_gt = extractor.extract_newsgroups_labels()
imdb_gt = extractor.extract_imdb_sentiment()

extractor.save_ground_truth(newsgroups_gt, 'newsgroups')
extractor.save_ground_truth(imdb_gt, 'imdb')

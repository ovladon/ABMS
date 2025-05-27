#!/usr/bin/env python3
"""
Corrected ABMS Evaluation Script
Fixes warnings and provides cleaner output while preserving core functionality
"""

import numpy as np
import pandas as pd
import pickle
import time
import warnings
from pathlib import Path
import sys
import os
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='An input array is constant')

# Add current directory to path for imports
sys.path.append('.')

def safe_correlation(x, y):
    """Calculate correlation with proper NaN handling"""
    try:
        # Convert to numpy arrays and remove NaN values
        x_clean = np.array(x, dtype=float)
        y_clean = np.array(y, dtype=float)
        
        # Check for constant arrays
        if len(set(x_clean)) <= 1 or len(set(y_clean)) <= 1:
            return np.nan, np.nan
            
        # Check for sufficient valid data points
        valid_mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
        if np.sum(valid_mask) < 3:  # Need at least 3 points for correlation
            return np.nan, np.nan
            
        return stats.pearsonr(x_clean[valid_mask], y_clean[valid_mask])
    except Exception:
        return np.nan, np.nan

def safe_classification_metrics(y_true, y_pred):
    """Calculate classification metrics with proper zero division handling"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, f1, precision, recall
    except Exception as e:
        print(f"Warning: Classification metrics calculation failed: {e}")
        return 0.0, 0.0, 0.0, 0.0

def load_newsgroups_sample(n_samples=50):
    """Load sample from 20 newsgroups dataset"""
    try:
        with open('datasets/newsgroups_train.pkl', 'rb') as f:
            newsgroups = pickle.load(f)
        
        # Sample n_samples from each of a few categories for diversity
        selected_categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian', 'talk.politics.guns']
        samples_per_category = n_samples // len(selected_categories)
        
        texts = []
        labels = []
        categories = []
        
        for category in selected_categories:
            if category in newsgroups.target_names:
                cat_idx = newsgroups.target_names.index(category)
                cat_indices = [i for i, label in enumerate(newsgroups.target) if label == cat_idx]
                
                # Sample from this category
                selected_indices = np.random.choice(cat_indices, 
                                                  min(samples_per_category, len(cat_indices)), 
                                                  replace=False)
                
                for idx in selected_indices:
                    texts.append(newsgroups.data[idx])
                    labels.append(newsgroups.target[idx])
                    categories.append(category)
        
        return texts[:n_samples], labels[:n_samples], categories[:n_samples]
    
    except FileNotFoundError:
        print("Error: Newsgroups dataset not found. Please run download_datasets.py first.")
        return [], [], []

def load_imdb_sample(n_samples=50):
    """Load sample from IMDB dataset"""
    try:
        pos_dir = Path('datasets/aclImdb/train/pos')
        neg_dir = Path('datasets/aclImdb/train/neg')
        
        texts = []
        labels = []
        
        # Load positive samples
        pos_files = list(pos_dir.glob('*.txt'))[:n_samples//2]
        for file_path in pos_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append('positive')
        
        # Load negative samples  
        neg_files = list(neg_dir.glob('*.txt'))[:n_samples//2]
        for file_path in neg_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append('negative')
        
        return texts, labels
    
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
        return [], []

def process_batch_with_abms(texts, batch_size=10):
    """Process texts with ABMS system in batches"""
    try:
        # Import the batch processor
        from batch_processor import BatchProcessor
        
        # Initialize processor
        processor = BatchProcessor()
        
        # Process in batches
        all_results = []
        processing_times = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            start_time = time.time()
            batch_results = processor.process_batch(batch_texts)
            end_time = time.time()
            
            all_results.extend(batch_results)
            processing_times.append(end_time - start_time)
        
        return all_results, processing_times
    
    except ImportError:
        print("Error: batch_processor.py not found. Processing individually...")
        return process_texts_individually(texts)

def process_texts_individually(texts):
    """Fallback: process texts one by one"""
    # Import key analysis modules
    try:
        from analysis_modules.sentiment_analysis import SentimentAnalysis
        from analysis_modules.readability_analysis import ReadabilityAnalysis
        from analysis_modules.cognitive_analysis import CognitiveAnalysis
        from analysis_modules.genre_analysis import GenreAnalysis
        from analysis_modules.complexity_analysis import ComplexityAnalysis
    except ImportError as e:
        print(f"Error importing analysis modules: {e}")
        return [], []
    
    results = []
    processing_times = []
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")
        start_time = time.time()
        
        # Process with key modules
        result = {}
        modules = [
            ('sentiment_analysis', SentimentAnalysis),
            ('readability_analysis', ReadabilityAnalysis),
            ('cognitive_analysis', CognitiveAnalysis),
            ('genre_analysis', GenreAnalysis),
            ('complexity_analysis', ComplexityAnalysis)
        ]
        
        for name, module_class in modules:
            try:
                analyzer = module_class(text)
                module_result = analyzer.analyze()
                result.update(module_result)
            except Exception as e:
                print(f"Warning: {name} failed for text {i+1}: {e}")
                result[name] = 0.0  # Default value
        
        results.append(result)
        processing_times.append(time.time() - start_time)
    
    return results, processing_times

def evaluate_abms():
    """Main evaluation function"""
    print("=== ABMS Evaluation Report ===\n")
    
    # Load datasets
    print("Loading datasets...")
    newsgroups_texts, newsgroups_labels, newsgroups_categories = load_newsgroups_sample(50)
    imdb_texts, imdb_labels = load_imdb_sample(50)
    
    if not newsgroups_texts or not imdb_texts:
        print("Error: Could not load datasets. Please check dataset files.")
        return
    
    print(f"Loaded {len(newsgroups_texts)} newsgroups samples")
    print(f"Loaded {len(imdb_texts)} IMDB samples\n")
    
    # Combine datasets
    all_texts = newsgroups_texts + imdb_texts
    all_labels = newsgroups_labels + [1 if label == 'positive' else 0 for label in imdb_labels]
    
    # Process with ABMS
    print("Processing texts with ABMS...")
    results, processing_times = process_batch_with_abms(all_texts)
    
    if not results:
        print("Error: ABMS processing failed.")
        return
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # 1. CORRELATION ANALYSIS
    print("1. CORRELATION ANALYSIS")
    print("-" * 30)
    
    # Calculate correlations with external measures
    if 'readability_analysis' in df_results.columns:
        # Readability vs text length
        text_lengths = [len(text) for text in all_texts]
        r, p = safe_correlation(df_results['readability_analysis'], text_lengths)
        if not np.isnan(r):
            print(f"Readability correlation: r={r:.3f}, p={p:.3f}")
        else:
            print("Readability correlation: Unable to calculate (constant values)")
    
    # Cognitive complexity vs readability
    if 'cognitive_analysis' in df_results.columns and 'readability_analysis' in df_results.columns:
        r, p = safe_correlation(df_results['cognitive_analysis'], df_results['readability_analysis'])
        if not np.isnan(r):
            print(f"Cognitive complexity correlation: r={r:.3f}, p={p:.3f}")
        else:
            print("Cognitive complexity correlation: Unable to calculate")
    
    # Sentiment vs IMDB labels
    if 'sentiment_analysis' in df_results.columns:
        imdb_sentiment_results = df_results['sentiment_analysis'].iloc[-len(imdb_labels):]
        imdb_binary_labels = [1 if label == 'positive' else -1 for label in imdb_labels]
        r, p = safe_correlation(imdb_sentiment_results, imdb_binary_labels)
        if not np.isnan(r):
            print(f"Sentiment correlation: r={r:.3f}, p={p:.3f}")
        else:
            print("Sentiment correlation: Unable to calculate (constant values)")
    
    print()
    
    # 2. CLASSIFICATION ACCURACY
    print("2. CLASSIFICATION ACCURACY")
    print("-" * 30)
    
    # Genre classification on newsgroups
    if 'genre_analysis' in df_results.columns:
        newsgroups_genre_pred = df_results['genre_analysis'].iloc[:len(newsgroups_texts)]
        
        # Map genre predictions to categories for evaluation
        le = LabelEncoder()
        try:
            genre_encoded = le.fit_transform(newsgroups_genre_pred)
            category_encoded = le.fit_transform(newsgroups_categories)
            
            accuracy, f1, precision, recall = safe_classification_metrics(category_encoded, genre_encoded)
            print(f"Genre classification accuracy: {accuracy:.3f}")
            print(f"Genre F1-score: {f1:.3f}")
        except Exception as e:
            print(f"Genre classification: Unable to evaluate ({e})")
    
    # Sentiment classification on IMDB
    if 'sentiment_analysis' in df_results.columns:
        imdb_sentiment_pred = df_results['sentiment_analysis'].iloc[-len(imdb_labels):]
        imdb_sentiment_binary = [1 if s > 0 else 0 for s in imdb_sentiment_pred]
        imdb_labels_binary = [1 if label == 'positive' else 0 for label in imdb_labels]
        
        accuracy, f1, precision, recall = safe_classification_metrics(imdb_labels_binary, imdb_sentiment_binary)
        print(f"Sentiment classification accuracy: {accuracy:.3f}")
        print(f"Sentiment F1-score: {f1:.3f}")
    
    print()
    
    # 3. PERFORMANCE ANALYSIS
    print("3. PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    avg_time = np.mean(processing_times)
    total_chars = sum(len(text) for text in all_texts)
    total_time = sum(processing_times)
    chars_per_second = total_chars / total_time if total_time > 0 else 0
    
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Processing rate: {chars_per_second:.0f} characters/second")
    print()
    
    # 4. ASPECT CORRELATIONS
    print("4. ASPECT CORRELATIONS")
    print("-" * 30)
    
    # Calculate correlations between aspects
    numerical_columns = df_results.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 1:
        correlation_matrix = df_results[numerical_columns].corr()
        
        # Find high correlations
        print("High aspect correlations (|r| > 0.5):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    print(f"  {col1} ↔ {col2}: r={corr_val:.3f}")
    
    # Save detailed results
    results_data = {
        'df_results': df_results,
        'processing_times': processing_times,
        'texts': all_texts,
        'labels': all_labels,
        'newsgroups_categories': newsgroups_categories,
        'imdb_labels': imdb_labels
    }
    
    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\nDetailed results saved to evaluation_results.pkl")
    
    return results_data

if __name__ == "__main__":
    evaluate_abms()

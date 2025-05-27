import pickle
import json
import time
from your_abms_module import ABMS  # Replace with your actual ABMS import

class ABMSBatchProcessor:
    def __init__(self):
        self.abms = ABMS()  # Initialize your ABMS system
        
    def process_ground_truth_file(self, filename, output_filename):
        """Process a ground truth file and generate ABMS aspects"""
        
        # Load ground truth
        with open(f'ground_truth_{filename}.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
        
        results = []
        total = len(ground_truth)
        
        print(f"Processing {total} samples from {filename}...")
        
        for i, sample in enumerate(ground_truth):
            print(f"Processing {i+1}/{total} ({i/total*100:.1f}%)")
            
            try:
                # Run ABMS analysis on the text
                abms_result = self.abms.analyze_text(sample['text'])
                
                # Combine ground truth with ABMS results
                combined_result = {
                    'ground_truth': sample,
                    'abms_aspects': abms_result,
                    'processing_time': abms_result.get('processing_time', 0)
                }
                
                results.append(combined_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save results
        with open(f'abms_results_{output_filename}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved {len(results)} processed samples to abms_results_{output_filename}.pkl")

# Run batch processing
processor = ABMSBatchProcessor()
processor.process_ground_truth_file('newsgroups', 'newsgroups')
processor.process_ground_truth_file('imdb', 'imdb')

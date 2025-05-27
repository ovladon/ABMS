import pickle
import time
import hashlib
import gc
from analysis_modules import (
    ActionabilityAnalysis, AudienceAppropriatenessAnalysis, CognitiveAnalysis,
    ComplexityAnalysis, ControversialityAnalysis, CulturalContextAnalysis,
    EmotionalPolarityAnalysis, EthicalConsiderationsAnalysis, FormalismAnalysis,
    GenreAnalysis, HumorAnalysis, IntentionalityAnalysis, InteractivityAnalysis,
    LexicalDiversityAnalysis, ModalityAnalysis, MultimodalityAnalysis,
    NarrativeStyleAnalysis, NoveltyAnalysis, ObjectivityAnalysis,
    PersuasivenessAnalysis, QuantitativeAnalysis, QualitativeAnalysis,
    ReadabilityAnalysis, ReliabilityAnalysis, SentimentAnalysis,
    SocialOrientationAnalysis, SpecificityAnalysis, SpatialAnalysis,
    SyntacticComplexityAnalysis, TemporalAnalysis
)

class ABMSBatchProcessor:
    def __init__(self):
        # List of all your analysis modules
        self.analysis_modules = [
            ActionabilityAnalysis, AudienceAppropriatenessAnalysis, CognitiveAnalysis,
            ComplexityAnalysis, ControversialityAnalysis, CulturalContextAnalysis,
            EmotionalPolarityAnalysis, EthicalConsiderationsAnalysis, FormalismAnalysis,
            GenreAnalysis, HumorAnalysis, IntentionalityAnalysis, InteractivityAnalysis,
            LexicalDiversityAnalysis, ModalityAnalysis, MultimodalityAnalysis,
            NarrativeStyleAnalysis, NoveltyAnalysis, ObjectivityAnalysis,
            PersuasivenessAnalysis, QuantitativeAnalysis, QualitativeAnalysis,
            ReadabilityAnalysis, ReliabilityAnalysis, SentimentAnalysis,
            SocialOrientationAnalysis, SpecificityAnalysis, SpatialAnalysis,
            SyntacticComplexityAnalysis, TemporalAnalysis
        ]
        
    def analyze_single_text(self, text):
        """Analyze a single text using all ABMS modules"""
        start_time = time.time()
        analysis_results = {}
        
        # Run each analysis module
        for module_class in self.analysis_modules:
            try:
                module_instance = module_class(text)
                result = module_instance.analyze()
                analysis_results.update(result)
            except Exception as e:
                print(f"Error in {module_class.__name__}: {e}")
                # Use default values for failed modules
                module_name = module_class.__name__.lower().replace('analysis', '_analysis')
                if 'sentiment' in module_name:
                    analysis_results[module_name] = 0.0
                elif 'readability' in module_name:
                    analysis_results[module_name] = 50.0
                else:
                    analysis_results[module_name] = 0.0
        
        # Add processing time and data hash
        processing_time = time.time() - start_time
        data_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        analysis_results['processing_time'] = processing_time
        analysis_results['data_hash'] = data_hash
        
        return analysis_results
        
    def process_ground_truth_file(self, filename, output_filename, max_samples=None):
        """Process a ground truth file and generate ABMS aspects"""
        
        # Load ground truth
        with open(f'ground_truth_{filename}.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
        
        # Limit samples if specified (for speed during testing)
        if max_samples:
            ground_truth = ground_truth[:max_samples]
        
        results = []
        total = len(ground_truth)
        
        print(f"Processing {total} samples from {filename}...")
        
        for i, sample in enumerate(ground_truth):
            print(f"Processing {i+1}/{total} ({i/total*100:.1f}%)")
            
            try:
                # Run ABMS analysis on the text
                abms_result = self.analyze_single_text(sample['text'])
                
                # Combine ground truth with ABMS results
                combined_result = {
                    'ground_truth': sample,
                    'abms_aspects': abms_result
                }
                
                results.append(combined_result)
                
                # Clean up memory every 10 samples
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Save results
        with open(f'abms_results_{output_filename}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved {len(results)} processed samples to abms_results_{output_filename}.pkl")
        
        # Print sample results for verification
        if results:
            print("\nSample ABMS aspects from first result:")
            sample_aspects = results[0]['abms_aspects']
            for key, value in list(sample_aspects.items())[:5]:  # Show first 5 aspects
                print(f"  {key}: {value}")

# Run batch processing
if __name__ == "__main__":
    processor = ABMSBatchProcessor()
    
    # Process smaller samples first for testing (50 samples each)
    print("Processing Newsgroups dataset...")
    processor.process_ground_truth_file('newsgroups', 'newsgroups', max_samples=50)
    
    print("\nProcessing IMDB dataset...")
    processor.process_ground_truth_file('imdb', 'imdb', max_samples=50)
    
    print("\nBatch processing completed!")

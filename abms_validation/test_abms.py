import sys
import os
sys.path.append('/home/vlad/CSML/ISI/Aspect_Based_Metadata_System/Paper_ABMS_latex/Information Processing & Management (Elsevier)/Seventh Aspect Based Metadata System/abms_validation')  # Add current directory to path

# Test imports
try:
    from analysis_modules import (
        ActionabilityAnalysis, SentimentAnalysis, ReadabilityAnalysis,
        CognitiveAnalysis, GenreAnalysis
    )
    print("✅ Analysis modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all your analysis_modules files are in the current directory")
    exit(1)

# Test with simple text
test_text = "This is a simple test. Please click here to learn more about our exciting new product! We believe this innovation will revolutionize the industry."

print("\nTesting key ABMS modules...")

modules_to_test = [
    ('Sentiment Analysis', SentimentAnalysis),
    ('Readability Analysis', ReadabilityAnalysis), 
    ('Actionability Analysis', ActionabilityAnalysis),
    ('Cognitive Analysis', CognitiveAnalysis),
    ('Genre Analysis', GenreAnalysis)
]

results = {}
for name, module_class in modules_to_test:
    try:
        module_instance = module_class(test_text)
        result = module_instance.analyze()
        results.update(result)
        print(f"✅ {name}: {result}")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        # Continue with other tests

print(f"\n🎉 ABMS test completed! Successfully tested {len(results)} aspects.")
print("If you see errors above, fix them before proceeding to Step 2.")

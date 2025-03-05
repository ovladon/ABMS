# aspect_based_metadata_generator.py

import zlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

def generate_aspect_based_metadata(analysis_results, encryption_key):
    # Define the alphabetical order of aspects
    aspect_order = [
        'actionability_analysis',
        'audience_appropriateness_analysis',
        'cognitive_analysis',
        'complexity_analysis',
        'controversiality_analysis',
        'cultural_context_analysis',
        'emotional_polarity_analysis',
        'ethical_considerations_analysis',
        'formalism_analysis',
        'genre_analysis',
        'humor_analysis',
        'intentionality_analysis',
        'interactivity_analysis',
        'lexical_diversity_analysis',
        'modality_analysis',
        'multimodality_analysis',
        'narrative_style_analysis',
        'novelty_analysis',
        'objectivity_analysis',
        'persuasiveness_analysis',
        'quantitative_analysis',
        'qualitative_analysis',
        'readability_analysis',
        'reliability_analysis',
        'sentiment_analysis',
        'social_orientation_analysis',
        'specificity_analysis',
        'spatial_analysis',
        'syntactic_complexity_analysis',
        'temporal_analysis'
    ]

    binary_data = ''

    for aspect in aspect_order:
        score = analysis_results.get(aspect)
        if score is None:
            print(f"Aspect '{aspect}' is missing in analysis_results. Defaulting to zero or appropriate default.")
            # Assign default value based on aspect type
            if aspect in numerical_aspects():
                score = numerical_aspects()[aspect][0]  # Default to min value
            elif aspect in categorical_aspects():
                # Default to the first category
                category_mapping, _ = categorical_aspects()[aspect]
                score = category_mapping[0]
            else:
                score = 0  # Default to zero
        print(f"Processing aspect '{aspect}' with score '{score}'")
        binary_score = encode_aspect(aspect, score)
        if binary_score is None:
            # Handle the case where encoding failed
            print(f"Error encoding aspect '{aspect}' with score '{score}'.")
            return None
        binary_data += binary_score

    # Convert binary string to bytes
    try:
        binary_int = int(binary_data, 2)
        binary_bytes = binary_int.to_bytes((binary_int.bit_length() + 7) // 8, byteorder='big')
    except ValueError as e:
        print(f"Error converting binary data to bytes: {e}")
        return None

    # Compress the data
    compressed_data = zlib.compress(binary_bytes)

    # Encrypt the data
    cipher = AES.new(encryption_key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(compressed_data, AES.block_size))
    iv = cipher.iv

    # Combine IV and ciphertext
    encrypted_data = iv + ct_bytes

    # Encode with Base64
    aspect_based_metadata = base64.b64encode(encrypted_data).decode('utf-8')

    return aspect_based_metadata

def encode_aspect(aspect, score):
    if score is None:
        print(f"Warning: Score for aspect '{aspect}' is None. Defaulting to zero.")
        score = 0

    if aspect in numerical_aspects():
        min_value, max_value, bits = numerical_aspects()[aspect]
        try:
            # Ensure score is within the expected range
            score = float(score)
            score = max(min_value, min(score, max_value))
            normalized_score = int((score - min_value) / (max_value - min_value) * (2 ** bits - 1))
            binary_score = format(normalized_score, f'0{bits}b')
        except (ValueError, TypeError) as e:
            print(f"Error encoding numerical aspect '{aspect}': {e}")
            return None
    elif aspect in categorical_aspects():
        category_mapping, bits = categorical_aspects()[aspect]
        int_score = None
        # Reverse mapping to find the key corresponding to the value
        for key, value in category_mapping.items():
            if value == score:
                int_score = key
                break
        if int_score is None:
            print(f"Warning: Score '{score}' not found in mapping for aspect '{aspect}'. Defaulting to zero.")
            int_score = 0
        binary_score = format(int_score, f'0{bits}b')
    else:
        # Default to zeros if aspect is not recognized
        bits = 8  # Default to 8 bits for unrecognized aspects
        binary_score = '0' * bits
    return binary_score

def numerical_aspects():
    return {
        'actionability_analysis': (0.0, 1.0, 16),
        'cognitive_analysis': (0.0, 20.0, 16),
        'complexity_analysis': (0.0, 1.0, 16),
        'controversiality_analysis': (0.0, 1.0, 16),
        'emotional_polarity_analysis': (0.0, 1.0, 16),
        'formalism_analysis': (0.0, 1.0, 16),
        'humor_analysis': (0.0, 1.0, 16),
        'interactivity_analysis': (0.0, 1.0, 16),
        'lexical_diversity_analysis': (0.0, 1.0, 16),
        'novelty_analysis': (0.0, 1.0, 16),
        'objectivity_analysis': (0.0, 1.0, 16),
        'persuasiveness_analysis': (0.0, 1.0, 16),
        'quantitative_analysis': (0.0, 1.0, 16),
        'qualitative_analysis': (0.0, 1.0, 16),
        'readability_analysis': (0.0, 100.0, 16),
        'reliability_analysis': (0.0, 1.0, 16),
        'sentiment_analysis': (-1.0, 1.0, 16),
        'social_orientation_analysis': (0.0, 1.0, 16),
        'specificity_analysis': (0.0, 1.0, 16),
        'syntactic_complexity_analysis': (0.0, 1.0, 16),
        'temporal_analysis': (0.0, 1.0, 16),
    }

def categorical_aspects():
    return {
        'audience_appropriateness_analysis': ({0: 'Children', 1: 'Middle School', 2: 'High School', 3: 'Adult'}, 2),
        'cultural_context_analysis': ({0: 'General', 1: 'Cultural Specific'}, 1),
        'ethical_considerations_analysis': ({0: 'Low', 1: 'Medium', 2: 'High'}, 2),
        'genre_analysis': ({0: 'Political Speech', 1: 'News', 2: 'Story', 3: 'Academic', 4: 'Legal', 5: 'Scientific', 6: 'Finance', 7: 'Entertainment', 8: 'Sports', 9: 'Historical Document'}, 4),
        'intentionality_analysis': ({0: 'Informative', 1: 'Persuasive', 2: 'Narrative', 3: 'Descriptive', 4: 'Expository', 5: 'Instructional'}, 3),
        'modality_analysis': ({0: 'Textual', 1: 'Visual', 2: 'Auditory', 3: 'Multimedia'}, 2),
        'multimodality_analysis': ({0: 'text', 1: 'image', 2: 'audio', 3: 'video', 4: 'interactive'}, 3),
        'narrative_style_analysis': ({0: 'First_Person', 1: 'Second_Person', 2: 'Third_Person'}, 2),
        'spatial_analysis': ({0: 'General', 1: 'Local', 2: 'Regional', 3: 'Global'}, 2),
    }


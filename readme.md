# Aspect-Based Metadata System

The Aspect-Based Metadata System (ABMS) is an open-source tool designed to generate a secure, encrypted metadata signature from input content by analyzing it through multiple analytical modules. The system leverages free, state-of-the-art tools for natural language processing, sentiment analysis, and multimodal content processing (text, images, audio, video) to produce a comprehensive **aspect-based metadata** descriptor. This descriptor can be used to verify content integrity, enhance machine-to-machine communication, and improve metadata standardization and storage efficiency.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Publisher Application](#publisher-application)
  - [Reader Application](#reader-application)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The ABMS processes various types of input content (such as text, PDFs, DOCX files, images, audio, video, and structured data) through a suite of analytical modules. It computes scores for multiple aspects of the content (e.g., sentiment, readability, objectivity) and aggregates these into a secure metadata signature. The **aspect-based metadata**—can be stored, transmitted, and later decoded by authorized readers to verify content meaning and integrity.

The system comprises two main components:

1. **Publisher Application:**  
   Ingests data, analyzes it via multiple modules, and produces the aspect-based metadata by encoding, compressing, and encrypting the aggregated scores. It also computes a hash of the original data to ensure data integrity.

2. **Reader Application:**  
   Allows users to input the metadata tag and corresponding encryption key (in hexadecimal format) to decode the analysis results. It then presents the inferred content categories and synergies using an extensive knowledge base.

## Features

- **Multi-Modal Analysis:**  
  Supports text, PDF, DOCX, image, audio, video, and structured data.

- **Aspect-Based Metadata Generation:**  
  Aggregates analytical scores into a binary representation that is compressed and encrypted, ensuring secure and verifiable metadata.

- **Content Speculation:**  
  Uses an extensible knowledge base to infer content categories and identify synergies between different aspects.

- **Integrity & Security:**  
  Incorporates hashing (SHA-256) to “stamp” the original data and AES encryption to protect the metadata signature.

- **Resource-Aware Processing:**  
  Implements dynamic resource throttling to ensure that CPU and memory usage never exceed 70% of available resources, thereby preventing system overload while still utilizing the processor for computations.

- **Open Source & Free Tools:**  
  Built exclusively using free and open-source tools to maximize accessibility and reproducibility.

## Architecture

The system is composed of several interrelated modules:

- **Data Ingestion Module:**  
  Handles file uploads and extracts text from various file types using libraries such as PyPDF2, docx2txt, and pytesseract.

- **Analysis Modules:**  
  A collection of analytical tools that compute scores for different aspects (e.g., sentiment, readability, objectivity) of the input content.

- **Metadata Generation Module:**  
  Encodes the aspect scores into a binary string, compresses it using zlib, and encrypts it with AES (using a randomly generated key).

- **Verification Module:**  
  Provides functionality to verify that the decoded metadata matches the original analysis results.

- **Publisher & Reader Applications:**  
  Implemented as Streamlit apps; the Publisher App processes the input and generates the metadata, and the Reader App decodes and displays the metadata.

## Installation

### Prerequisites

- **Python 3.8+**
- **pip**

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/aspect-based-metadata.git
   cd aspect-based-metadata

2. **Install Dependencies:**

Use the provided requirements.txt file:
pip install -r requirements.txt

3. **Additional Setup:**

        Vosk Model:
        The system will download the Vosk model automatically if it is not found in the models directory.

        BLIP Model:
        The BLIP image captioning model is loaded via the Hugging Face Transformers library. GPU support is recommended if available.

### Usage
**Publisher Application**

The Publisher App processes input data and generates the aspect-based metadata.

    1. Run the Publisher App:
    streamlit run publisher_app.py
    
    2. Upload Data:
    Upload a file (text, PDF, DOCX, image, audio, video, CSV, Excel) or enter text manually.

    3. Start Analysis:
    Click the "Start Analysis" button to begin processing. The app will display progress and resource usage. When complete, it shows the generated metadata and the encryption key.

    4. Save Results:
    You can save the analysis results as a JSON file for future reference.

**Reader Application**

The Reader App decodes the aspect-based metadata to reconstruct the analysis results.

    1. Run the Reader App:
    streamlit run reader_app.py
    
    2. Input Metadata:
    Paste the aspect-based metadata tag and the corresponding encryption key (in hexadecimal format) into the provided fields.

    3. Decode and View Results:
    Click the "Decode Ultimate Tag" button to view the reconstructed analysis results, inferred content categories, and synergies.

**Contributing**

Contributions are welcome! If you would like to contribute enhancements, bug fixes, or documentation improvements, please follow these steps:

    Fork the repository.
    Create a new branch for your feature or bug fix.
    Commit your changes with clear messages.
    Open a pull request describing your modifications.

## License

 Aspect-Based Metadata System (ABMS) © 2025 by Belciug Vlad, Pelican Elena is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

## Acknowledgements

We gratefully acknowledge all the researchers and developers whose pioneering work in metadata optimization and machine-to-machine communication has paved the way for this project. Their foundational contributions have enabled significant advances in content analysis, data verification, and secure information exchange.


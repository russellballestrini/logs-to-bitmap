# Visual Anomaly Detection for HTTP Requests

A machine learning system that converts HTTP request logs into bitmap images for computer vision-based anomaly detection. This project explores an unconventional approach to cybersecurity: treating security logs as visual data that ML models can analyze as images.

## Features

- **Visual Log Analysis**: Converts HTTP requests to bitmap images for ML analysis
- **Anomaly Detection**: Uses Isolation Forest with computer vision features to detect suspicious patterns
- **Multi-format Support**: Generates BMP, JPEG, PNG, and WebP images for robust feature extraction
- **Comprehensive Testing**: Unit, integration, and functional test suites for diagnosing ML performance issues
- **HTTP Server**: Pyramid-based web server for data generation and testing
- **Reproducible Science**: All experiments use fixed random seeds and documented methodologies

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate training data:**
   ```bash
   make run-server          # Start server (terminal 1)
   make run-crawler-1k      # Generate 1K samples with anomalies (terminal 2)
   ```

3. **Train anomaly detection model:**
   ```bash
   make train-model-1k      # Train on 1K dataset
   ```

4. **Test on fresh data:**
   ```bash
   make clean && make run-crawler-1k    # Generate fresh test data
   python3 utils/anomaly_detection.py images/ --load-model models/model_1k_samples_*.pkl
   ```

5. **Run diagnostic tests:**
   ```bash
   make test                # Run all tests
   make test-functional     # Test overfitting diagnosis
   ```

## Project Structure

```
logs-to-bitmap/
├── app.py                    # Main HTTP server with bitmap generation
├── Makefile                  # Build automation and test runner
├── requirements.txt          # Python dependencies
├── utils/
│   ├── anomaly_detection.py  # ML anomaly detection with Isolation Forest
│   ├── crawler.py            # Original test crawler (100 requests)
│   └── crawler_1k.py         # 1K crawler with random anomaly injection
├── tests/                    # Comprehensive test suite
│   ├── unit/                 # Component-level testing
│   ├── integration/          # End-to-end workflow testing
│   ├── functional/           # ML performance diagnosis tests
│   └── run_tests.py          # Test runner script
├── logs/                     # Generated log files (gitignored)
├── images/                   # Generated image files (gitignored)
├── models/                   # Trained ML models (gitignored)
├── anomaly_scores/           # Detection results (gitignored)
└── zips/                     # Distribution packages (gitignored)
```

## Usage

### Data Generation
```bash
make run-server         # Start the Pyramid server
make run-crawler        # Run test crawler (100 requests)
make run-crawler-1k     # Generate 1K requests with 10 random anomalies
make clean              # Remove all generated files
```

### Machine Learning Workflow
```bash
# Train models on different dataset sizes
make train-model-100    # Train on 100-sample dataset
make train-model-1k     # Train on 1K-sample dataset  

# Analyze datasets for anomalies
make anomaly-detect     # Run detection on existing images

# Model management
make compare-models     # Compare trained model sizes and metadata
make zip                # Create timestamped backup
```

### Testing and Diagnosis
```bash
# Comprehensive test suite
make test                    # Run all tests (unit, integration, functional)
make test-unit              # Test individual components  
make test-integration       # Test end-to-end workflows
make test-functional        # Test ML performance issues

# Focused diagnostics
make test-overfitting       # Cross-validation overfitting detection
make test-reproducibility   # Random seed and reproducibility tests
```

### Advanced Analysis
```bash
# Load existing model for analysis
python3 utils/anomaly_detection.py images/ --load-model models/model_1k_samples_*.pkl

# Train with custom parameters
python3 utils/anomaly_detection.py images/ --contamination 0.05 --estimators 200

# Analyze specific file types
python3 utils/anomaly_detection.py images/ --file-type jpg --contamination 0.01
```

## Machine Learning Approach

### Visual Feature Extraction
- **Color Histograms**: 128-dimensional HSV color distribution analysis
- **Text Density**: Ratio of non-white pixels indicating content density  
- **Projection Analysis**: Horizontal/vertical text distribution patterns
- **Format Robustness**: Extracts features from BMP, JPEG, PNG, and WebP formats

### Anomaly Detection Model
- **Algorithm**: Isolation Forest for unsupervised anomaly detection
- **Feature Space**: 136-dimensional vectors per image
- **Contamination**: Configurable expected anomaly rate (default 1%)
- **Random Sampling**: One format per request to prevent format bias

### Cross-Dataset Validation
- **Overfitting Detection**: Cross-validation tests reveal generalization issues  
- **Temporal Robustness**: Tests model performance across different timestamps
- **Reproducibility**: Fixed random seeds ensure consistent results
- **Format Consistency**: Validates detection across multiple image formats

## Performance Notes

- **Training Performance**: 90%+ recall on training data
- **Cross-Dataset Performance**: ~70% recall on fresh datasets
- **Testing Framework**: Comprehensive test suite diagnoses ML performance issues

## Requirements

- Python 3.7+
- scikit-learn (Isolation Forest, preprocessing)
- OpenCV (cv2) for computer vision features
- Pillow (PIL) for image processing
- NumPy for numerical operations
- Pyramid web framework
- Requests for HTTP client functionality

## License

Open source - see repository for details.
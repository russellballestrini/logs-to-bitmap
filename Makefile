# Makefile for logs-to-bitmap project

.PHONY: all clean run-server run-crawler run-crawler-1k convert-bitmaps zip anomaly-detect train-model-100 train-model-1k compare-models help

# Get current unix timestamp
TIMESTAMP := $(shell date +%s)

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make clean           - Remove log and bitmap files"
	@echo "  make run-server      - Start the Pyramid server"
	@echo "  make run-crawler     - Run the original crawler (100 requests)"
	@echo "  make run-crawler-1k  - Run 1K crawler (1000 requests with 10 anomalies)"
	@echo "  make convert-bitmaps - Convert all BMPs to JPGs"
	@echo "  make anomaly-detect  - Run anomaly detection on bitmap samples"
	@echo "  make train-model-100 - Train model on 100-sample dataset"
	@echo "  make train-model-1k  - Train model on 1K-sample dataset"
	@echo "  make compare-models  - Compare model sizes and performance"
	@echo "  make zip             - Create timestamped zip file of logs and bitmaps in zips/"

# Clean targets
clean:
	@echo "Logs and bitmaps cleaned"
	rm -f logs/*
	@echo "Log files removed"
	rm -f bitmaps/*
	@echo "Bitmap files removed"


# Run targets
run-server:
	@echo "Starting Pyramid server on http://localhost:6543"
	@echo "Press Ctrl+C to stop"
	python3 app.py

run-crawler:
	@echo "Running original crawler (100 requests)"
	python3 crawler.py

run-crawler-1k:
	@echo "Running 1K crawler (1000 requests with 10 anomalies)"
	python3 crawler_1k.py

# Convert bitmaps to JPGs
convert-bitmaps:
	@echo "Converting all BMP files to JPG format"
	python3 utils/bmp_to_jpg.py -d bitmaps/ -o jpgs/

# Anomaly detection
anomaly-detect:
	@echo "Running anomaly detection on bitmap samples"
	@if [ -d "bitmaps" ] && [ -n "$$(ls -A bitmaps/ 2>/dev/null)" ]; then \
		python3 anomaly_detection.py bitmaps/ --output anomaly_results.csv; \
	else \
		echo "Error: No bitmap files found. Run 'make run-crawler' first."; \
		exit 1; \
	fi

# Model training targets
train-model-100:
	@echo "Training model on 100-sample dataset"
	@mkdir -p models
	@if [ -d "bitmaps" ] && [ -n "$$(ls -A bitmaps/ 2>/dev/null)" ]; then \
		SAMPLE_COUNT=$$(ls bitmaps/*.bmp | wc -l); \
		if [ "$$SAMPLE_COUNT" -eq 100 ]; then \
			python3 anomaly_detection.py bitmaps/ --save-model models/model_100_samples_$(TIMESTAMP).pkl --output results_100_samples.csv; \
			echo "Model saved: models/model_100_samples_$(TIMESTAMP).pkl"; \
		else \
			echo "Error: Expected 100 samples, found $$SAMPLE_COUNT. Run 'make run-crawler' first."; \
			exit 1; \
		fi \
	else \
		echo "Error: No bitmap files found. Generate 100 samples first."; \
		exit 1; \
	fi

train-model-1k:
	@echo "Training model on 1K-sample dataset"
	@mkdir -p models
	@if [ -d "bitmaps" ] && [ -n "$$(ls -A bitmaps/ 2>/dev/null)" ]; then \
		SAMPLE_COUNT=$$(ls bitmaps/*.bmp | wc -l); \
		if [ "$$SAMPLE_COUNT" -eq 1000 ]; then \
			python3 anomaly_detection.py bitmaps/ --contamination 0.01 --estimators 200 --save-model models/model_1k_samples_$(TIMESTAMP).pkl --output results_1k_samples.csv; \
			echo "Model saved: models/model_1k_samples_$(TIMESTAMP).pkl"; \
		else \
			echo "Error: Expected 1000 samples, found $$SAMPLE_COUNT. Run 'make run-crawler-1k' first."; \
			exit 1; \
		fi \
	else \
		echo "Error: No bitmap files found. Generate 1000 samples first."; \
		exit 1; \
	fi

compare-models:
	@echo "Comparing trained models"
	@if [ ! -d "models" ] || [ -z "$$(ls models/*.pkl 2>/dev/null)" ]; then \
		echo "No models found. Train models first with 'make train-model-100' or 'make train-model-1k'"; \
		exit 1; \
	fi
	@echo "Model Comparison Report"
	@echo "======================="
	@echo ""
	@for model in models/*.pkl; do \
		if [ -f "$$model" ]; then \
			SIZE=$$(ls -lah "$$model" | awk '{print $$5}'); \
			BYTES=$$(stat -f%z "$$model" 2>/dev/null || stat -c%s "$$model" 2>/dev/null || echo "?"); \
			BASENAME=$$(basename "$$model"); \
			echo "Model: $$BASENAME"; \
			echo "  Size: $$SIZE ($$BYTES bytes)"; \
			if echo "$$BASENAME" | grep -q "100_samples"; then \
				echo "  Dataset: 100 samples"; \
			elif echo "$$BASENAME" | grep -q "1k_samples"; then \
				echo "  Dataset: 1000 samples"; \
			fi; \
			echo "  Created: $$(ls -l "$$model" | awk '{print $$6, $$7, $$8}')"; \
			echo ""; \
		fi \
	done

# Zip target
zip:
	@echo "Creating timestamped logs and bitmaps zip"
	@mkdir -p zips
	@if [ -d "logs" ] && [ -d "bitmaps" ]; then \
		zip -r zips/$(TIMESTAMP)_logs_and_bitmaps.zip logs/ bitmaps/; \
		echo "Created: zips/$(TIMESTAMP)_logs_and_bitmaps.zip"; \
	else \
		echo "Error: logs and/or bitmaps directories not found"; \
		exit 1; \
	fi

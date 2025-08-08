# Makefile for logs-to-bitmap project

.PHONY: all clean run-server run-crawler convert-bitmaps zip help

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
	@echo "  make convert-bitmaps - Convert all BMPs to JPGs"
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

# Convert bitmaps to JPGs
convert-bitmaps:
	@echo "Converting all BMP files to JPG format"
	python3 utils/bmp_to_jpg.py -d bitmaps/ -o jpgs/

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

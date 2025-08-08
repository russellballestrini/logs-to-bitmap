# Logs to Bitmap

A Python-based HTTP request logging system that converts web server logs into bitmap images for visual analysis and data preservation.

## Features

- **HTTP Server**: Pyramid-based web server with two endpoints (`/` and `/hello`)
- **Request Logging**: Sequential numbering and comprehensive logging of all HTTP requests
- **Bitmap Generation**: Converts logs to grayscale bitmap images with intelligent text wrapping
- **Format Conversion**: BMP to JPG converter for compatibility with various image viewers
- **Automated Testing**: Crawler scripts to generate test data with different user agent patterns
- **Build System**: Makefile with common workflow automation

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   make run-server
   ```

3. **Generate test data** (in another terminal):
   ```bash
   make run-crawler
   ```

4. **Create distribution package:**
   ```bash
   make zip
   ```

## Project Structure

```
logs-to-bitmap/
├── app.py              # Main HTTP server with bitmap generation
├── crawler.py          # Test crawler with mixed user agents
├── crawler_v2.py       # Enhanced crawler with different patterns
├── logs_to_bitmap.py   # Standalone log-to-bitmap converter
├── Makefile           # Build automation
├── requirements.txt   # Python dependencies
├── utils/
│   └── bmp_to_jpg.py  # BMP to JPG converter utility
├── logs/              # Generated log files (gitignored)
├── bitmaps/           # Generated bitmap files (gitignored)
├── jpgs/              # Converted JPG files (gitignored)
└── zips/              # Distribution packages (gitignored)
```

## Usage

### Server Operations
```bash
make run-server         # Start the Pyramid server
make run-crawler        # Run test crawler (100 requests)
```

### File Management
```bash
make clean              # Remove log and bitmap files
make convert-bitmaps    # Convert all BMPs to JPGs
make zip                # Create timestamped distribution package
```

### Manual Conversion
```bash
# Convert single BMP file
python3 utils/bmp_to_jpg.py image.bmp

# Convert directory of BMPs
python3 utils/bmp_to_jpg.py -d bitmaps/ -o jpgs/

# Convert standalone logs to bitmaps
python3 logs_to_bitmap.py
```

## Technical Details

### Bitmap Generation
- **Fixed Width**: 800px for consistent formatting
- **Font Metrics**: Uses actual font measurements for text wrapping
- **Smart Wrapping**: Respects word boundaries when possible
- **Grayscale Format**: 8-bit grayscale BMP for data integrity

### Request Logging
- **Sequential Numbering**: Thread-safe request counter (000001-000100)
- **Comprehensive Data**: Timestamps, headers, endpoints, user agents
- **Dual Format**: Both text logs and bitmap images generated

### Data Patterns
- **Endpoint Distribution**: ~66% requests to `/`, ~33% to `/hello`
- **User Agent Variation**: 1 random request per 100 uses different user agent
- **Timing**: Variable delays between requests (50-150ms)

## File Formats

- **Logs**: Human-readable text files with structured data
- **Bitmaps**: Grayscale BMP files for data preservation
- **JPGs**: Converted images for viewing compatibility
- **Zips**: Timestamped distribution packages

## Requirements

- Python 3.7+
- Pyramid web framework
- Pillow (PIL) for image processing
- Requests for HTTP client functionality

## License

Open source - see repository for details.
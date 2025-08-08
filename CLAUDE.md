# Claude Code Notes

## BMP to JPG Conversion Requirement

Claude Code's API cannot read BMP (bitmap) files directly. When attempting to read a BMP file, it returns the following error:

```
API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"messages.106.content.0.tool_result.content.0.image.source.base64.media_type: Input should be 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'"}}
```

### Solution

A BMP to JPG converter utility has been created in `utils/bmp_to_jpg.py` specifically to allow Claude Code to view the bitmap data.

### Usage

1. **Convert single BMP file:**
   ```bash
   python3 utils/bmp_to_jpg.py image.bmp
   ```

2. **Convert all BMPs in a directory:**
   ```bash
   python3 utils/bmp_to_jpg.py -d bitmaps/ -o jpgs/
   ```

3. **Adjust quality (default 95):**
   ```bash
   python3 utils/bmp_to_jpg.py image.bmp -q 90
   ```

### Important Notes

- The JPG conversion is purely for Claude Code viewing purposes
- The original BMP files contain the actual data and should be preserved
- The BMP generation parameters (font size, character width, line height) have been optimized to fit more text content in the images

## Git Commit Messages

When making commits, do not include Claude Code attribution (e.g., "🤖 Generated with Claude Code" or "Co-Authored-By: Claude") in commit messages. Keep commit messages clean and focused on the changes made.
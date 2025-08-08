import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class LogToBitmap:
    def __init__(self, font_size=12, char_width=7, line_height=14):
        # Monospaced font parameters
        self.font_size = font_size
        self.char_width = char_width  # Width of each character in pixels
        self.line_height = line_height  # Height of each line in pixels
        
        # Try to use a monospaced font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                # Fallback to default font
                self.font = ImageFont.load_default()
                print("Warning: Using default font, character spacing may not be perfect")
    
    def get_text_width(self, text):
        """Get actual pixel width of text using font metrics"""
        bbox = self.font.getbbox(text)
        return bbox[2] - bbox[0]
    
    def wrap_text(self, text, max_width_pixels):
        """Wrap text to fit within max_width_pixels"""
        if self.get_text_width(text) <= max_width_pixels:
            return [text]
        
        wrapped_lines = []
        words = text.split(' ')
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if self.get_text_width(test_line) <= max_width_pixels:
                current_line = test_line
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, force break
                    wrapped_lines.append(word)
                    current_line = ""
        
        if current_line:
            wrapped_lines.append(current_line)
        
        return wrapped_lines
    
    def parse_log_file(self, filepath):
        """Parse log file and return ordered fields"""
        fields = {}
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Parse the log file
        lines = content.strip().split('\n')
        headers_section = False
        headers = []
        
        for line in lines:
            if line.strip() == "Headers:":
                headers_section = True
                continue
            
            if headers_section:
                if line.startswith('  '):
                    headers.append(line.strip())
            else:
                if ':' in line:
                    key, value = line.split(':', 1)
                    fields[key.strip()] = value.strip()
        
        # Sort headers for consistency
        headers.sort()
        fields['Headers'] = headers
        
        return fields
    
    def create_bitmap(self, fields, output_path):
        """Create a grayscale bitmap from log fields"""
        # Calculate max pixel width for text (800px minus padding)
        max_text_width = 800 - 20
        
        # Define consistent field order
        field_order = [
            'Timestamp',
            'Request ID',
            'Endpoint',
            'Method',
            'URL',
            'Client Address',
            'User-Agent',
            'Headers'
        ]
        
        # Build text lines with consistent ordering and wrapping
        lines = []
        for field in field_order:
            if field in fields:
                if field == 'Headers':
                    lines.append(f"{field}:")
                    for header in fields[field]:
                        header_line = f"  {header}"
                        wrapped = self.wrap_text(header_line, max_text_width)
                        lines.extend(wrapped)
                else:
                    field_line = f"{field}: {fields[field]}"
                    wrapped = self.wrap_text(field_line, max_text_width)
                    lines.extend(wrapped)
        
        # Fixed width of 800px
        width = 800
        height = len(lines) * self.line_height + 20  # Add padding
        
        # Create grayscale image (mode 'L' for 8-bit grayscale)
        img = Image.new('L', (width, height), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y_position = 10
        for line in lines:
            draw.text((10, y_position), line, font=self.font, fill=0)  # Black text
            y_position += self.line_height
        
        # Save as BMP
        img.save(output_path, 'BMP')
        
        return img
    
    def process_all_logs(self, logs_dir='logs', output_dir='bitmaps'):
        """Process all log files in the logs directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all log files
        log_files = glob.glob(os.path.join(logs_dir, '*.log'))
        log_files.sort()  # Sort for consistent processing
        
        print(f"Found {len(log_files)} log files to process")
        
        for i, log_file in enumerate(log_files):
            basename = os.path.basename(log_file)
            output_name = basename.replace('.log', '.bmp')
            output_path = os.path.join(output_dir, output_name)
            
            # Parse and convert
            fields = self.parse_log_file(log_file)
            self.create_bitmap(fields, output_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(log_files)} files")
        
        print(f"Conversion complete! Bitmaps saved to {output_dir}/")
        
        # Find the request with different user agent
        for log_file in log_files:
            fields = self.parse_log_file(log_file)
            if 'User-Agent' in fields and 'Chrome/92.0' in fields['User-Agent']:
                print(f"\nRequest with different user agent found in: {log_file}")
                print(f"User-Agent: {fields['User-Agent']}")
                break

if __name__ == "__main__":
    converter = LogToBitmap()
    converter.process_all_logs()
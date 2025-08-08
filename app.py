from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config
from wsgiref.simple_server import make_server
import logging
import datetime
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('bitmaps', exist_ok=True)

# Global request counter with thread safety
request_counter = 0
counter_lock = threading.Lock()

class BitmapGenerator:
    def __init__(self, font_size=12, char_width=7, line_height=14):
        self.font_size = font_size
        self.char_width = char_width
        self.line_height = line_height
        
        # Try to use a monospaced font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                self.font = ImageFont.load_default()
    
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
    
    def create_bitmap(self, log_data, bitmap_path):
        """Create a grayscale bitmap from log data"""
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
        
        # Build text lines with wrapping
        lines = []
        for field in field_order:
            if field in log_data:
                if field == 'Headers':
                    lines.append(f"{field}:")
                    for header in log_data[field]:
                        header_line = f"  {header}"
                        wrapped = self.wrap_text(header_line, max_text_width)
                        lines.extend(wrapped)
                else:
                    field_line = f"{field}: {log_data[field]}"
                    wrapped = self.wrap_text(field_line, max_text_width)
                    lines.extend(wrapped)
        
        # Fixed width of 800px
        width = 800
        height = len(lines) * self.line_height + 20
        
        # Create grayscale image
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y_position = 10
        for line in lines:
            draw.text((10, y_position), line, font=self.font, fill=0)
            y_position += self.line_height
        
        # Save as BMP
        img.save(bitmap_path, 'BMP')

# Initialize bitmap generator
bitmap_gen = BitmapGenerator()

def request_counter_middleware(handler, registry):
    """Middleware to count requests"""
    def middleware(request):
        global request_counter
        with counter_lock:
            request_counter += 1
            request.request_number = request_counter
        return handler(request)
    return middleware

def log_request(request, endpoint):
    """Log each request to a separate file and create bitmap"""
    user_agent = request.headers.get('User-Agent', 'Unknown')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    request_id = str(uuid.uuid4())[:8]
    request_num = getattr(request, 'request_number', 0)
    
    # Include request number at the beginning of filenames
    filename = f"logs/{request_num:06d}_request_{timestamp}_{request_id}.log"
    bitmap_filename = f"bitmaps/{request_num:06d}_request_{timestamp}_{request_id}.bmp"
    
    # Prepare log data
    log_data = {
        'Timestamp': datetime.datetime.now().isoformat(),
        'Request ID': request_id,
        'Endpoint': endpoint,
        'Client Address': request.client_addr,
        'User-Agent': user_agent,
        'Method': request.method,
        'URL': request.url,
        'Headers': []
    }
    
    # Collect and sort headers
    headers = []
    for header, value in request.headers.items():
        headers.append(f"{header}: {value}")
    headers.sort()
    log_data['Headers'] = headers
    
    # Write log file
    with open(filename, 'w') as f:
        f.write(f"Timestamp: {log_data['Timestamp']}\n")
        f.write(f"Request ID: {log_data['Request ID']}\n")
        f.write(f"Endpoint: {log_data['Endpoint']}\n")
        f.write(f"Client Address: {log_data['Client Address']}\n")
        f.write(f"User-Agent: {log_data['User-Agent']}\n")
        f.write(f"Method: {log_data['Method']}\n")
        f.write(f"URL: {log_data['URL']}\n")
        f.write(f"Headers:\n")
        for header in headers:
            f.write(f"  {header}\n")
    
    # Create bitmap
    bitmap_gen.create_bitmap(log_data, bitmap_filename)
    
    logger.info(f"Request #{request_num} to {endpoint} from {request.client_addr} with User-Agent: {user_agent} - Logged to {filename} and {bitmap_filename}")

@view_config(route_name='home')
def home_view(request):
    log_request(request, '/')
    return Response('Welcome to the home page!')

@view_config(route_name='hello')
def hello_view(request):
    log_request(request, '/hello')
    return Response('Hello, World!')

def main():
    config = Configurator()
    
    # Add request counter middleware
    config.add_tween('__main__.request_counter_middleware')
    
    config.add_route('home', '/')
    config.add_route('hello', '/hello')
    config.scan()
    
    app = config.make_wsgi_app()
    server = make_server('localhost', 6543, app)
    print('Server started at http://localhost:6543')
    server.serve_forever()

if __name__ == '__main__':
    main()
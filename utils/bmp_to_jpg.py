#!/usr/bin/env python3
"""
BMP to JPG converter utility
Converts BMP files to JPG format so they can be read by Claude Code
"""

import os
import sys
import argparse
from PIL import Image
from pathlib import Path


def convert_bmp_to_jpg(bmp_path, output_path=None, quality=95):
    """
    Convert a BMP file to JPG format
    
    Args:
        bmp_path: Path to the BMP file
        output_path: Path for the output JPG file (optional)
        quality: JPG quality (1-100, default 95)
    
    Returns:
        Path to the created JPG file
    """
    bmp_path = Path(bmp_path)
    
    if not bmp_path.exists():
        raise FileNotFoundError(f"BMP file not found: {bmp_path}")
    
    if not bmp_path.suffix.lower() == '.bmp':
        raise ValueError(f"File is not a BMP: {bmp_path}")
    
    # Determine output path
    if output_path is None:
        output_path = bmp_path.with_suffix('.jpg')
    else:
        output_path = Path(output_path)
    
    # Open and convert the image
    try:
        with Image.open(bmp_path) as img:
            # Convert to RGB if necessary (BMP might be in different modes)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        print(f"Converted: {bmp_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        raise Exception(f"Error converting {bmp_path}: {str(e)}")


def convert_directory(input_dir, output_dir=None, quality=95):
    """
    Convert all BMP files in a directory to JPG format
    
    Args:
        input_dir: Directory containing BMP files
        output_dir: Directory for output JPG files (optional)
        quality: JPG quality (1-100, default 95)
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")
    
    # Use same directory if output not specified
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all BMP files
    bmp_files = list(input_dir.glob('*.bmp')) + list(input_dir.glob('*.BMP'))
    
    if not bmp_files:
        print(f"No BMP files found in {input_dir}")
        return
    
    print(f"Found {len(bmp_files)} BMP files to convert")
    
    converted = 0
    errors = []
    
    for bmp_file in bmp_files:
        try:
            output_path = output_dir / bmp_file.with_suffix('.jpg').name
            convert_bmp_to_jpg(bmp_file, output_path, quality)
            converted += 1
        except Exception as e:
            errors.append((bmp_file, str(e)))
    
    print(f"\nConversion complete: {converted}/{len(bmp_files)} files converted")
    
    if errors:
        print(f"\nErrors encountered:")
        for file, error in errors:
            print(f"  {file}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BMP files to JPG format",
        epilog="Examples:\n"
               "  Convert single file: %(prog)s image.bmp\n"
               "  Convert with custom output: %(prog)s image.bmp -o output.jpg\n"
               "  Convert directory: %(prog)s -d ./bitmaps/\n"
               "  Convert to different directory: %(prog)s -d ./bitmaps/ -o ./jpgs/",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('input_file', nargs='?', help='BMP file to convert')
    input_group.add_argument('-d', '--directory', help='Convert all BMP files in directory')
    
    # Output options
    parser.add_argument('-o', '--output', help='Output file/directory path')
    parser.add_argument('-q', '--quality', type=int, default=95, 
                       help='JPG quality (1-100, default: 95)')
    
    args = parser.parse_args()
    
    try:
        if args.directory:
            # Directory conversion mode
            convert_directory(args.directory, args.output, args.quality)
        else:
            # Single file conversion mode
            convert_bmp_to_jpg(args.input_file, args.output, args.quality)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
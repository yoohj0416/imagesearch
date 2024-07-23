#!/bin/bash

# Input directory
FOLDER_DIR="firstframes"
# Output directory
OUTPUT_DIR="firstframes_jpeg"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert PNG to JPEG
for img in "$FOLDER_DIR"/*.png; do
    # Extract the filename without the extension and directory
    filename=$(basename "${img%.*}")
    # Convert and save in the output directory with quality 85
    convert "$img" -quality 85 "$OUTPUT_DIR/${filename}.jpg"
done

echo "Conversion completed."
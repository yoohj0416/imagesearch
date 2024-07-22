#!/bin/bash

# Input directory
FOLDER_DIR="firstframes"
# Output directory
OUTPUT_DIR="firstframes_jpeg"

# Convert PNG to JPEG
for img in "$FOLDER_PATH"/*.png; do
    filename="${img%.*}"
    convert "$img" -quality 85 "$OUTPUT_DIR/${filename}.jpg"
done

echo "Conversion completed."
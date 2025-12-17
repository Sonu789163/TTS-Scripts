#!/bin/bash

# Audio Converter for TTS Reference Voice
# Converts any audio file to the proper WAV format for Coqui TTS

if [ $# -eq 0 ]; then
    echo "Usage: ./convert_audio.sh <input_file> [output_file]"
    echo ""
    echo "Example:"
    echo "  ./convert_audio.sh myvoice.m4a myvoice_converted.wav"
    echo "  ./convert_audio.sh recording.mp3"
    echo ""
    echo "If output file is not specified, it will be: <input>_converted.wav"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.*}_converted.wav}"

echo "🎵 Converting Audio File for TTS"
echo "================================"
echo ""
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg is not installed!"
    echo "   Install it with: brew install ffmpeg"
    exit 1
fi

# Convert to proper WAV format
# - 22050 Hz sample rate (good for voice)
# - Mono (1 channel)
# - PCM 16-bit signed integer format
echo "🔄 Converting..."
ffmpeg -i "$INPUT_FILE" \
    -ar 22050 \
    -ac 1 \
    -sample_fmt s16 \
    "$OUTPUT_FILE" \
    -y \
    -hide_banner \
    -loglevel error

if [ $? -eq 0 ]; then
    echo "✅ Conversion successful!"
    echo ""
    echo "📊 File Info:"
    file "$OUTPUT_FILE"
    echo ""
    echo "📁 Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "✅ Ready to use as reference voice!"
    echo ""
    echo "To use this file, update web_voice_agent_integration.py:"
    echo "   reference_voice = \"$OUTPUT_FILE\""
else
    echo "❌ Conversion failed!"
    exit 1
fi

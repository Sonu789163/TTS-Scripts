#!/bin/bash

# Start script for Cloned Voice TTS Server

echo "=================================================="
echo "🎙️  STARTING CLONED VOICE TTS SERVER"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "📂 Working directory: $SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "coqui_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run setup first:"
    echo "   ./setup_tts_server.sh"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source coqui_env/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Check if reference voice exists
if [ ! -f "sonuRecording_clean.wav" ]; then
    echo "⚠️  Warning: Reference voice file 'sonuRecording_clean.wav' not found"
    echo "   The server may not work correctly."
    echo ""
fi

# Start the server
echo "🚀 Starting TTS server on http://localhost:5000..."
echo "   Press Ctrl+C to stop"
echo ""
echo "=================================================="
echo ""

python web_voice_agent_integration.py

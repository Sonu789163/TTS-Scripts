#!/usr/bin/env python3
"""
Web-Based Realtime Voice Agent with Voice Cloning
Flask API for your voice agent frontend - OPTIMIZED FOR SPEED & 16kHz
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from TTS.api import TTS
import os
import uuid
import threading
import time
from pathlib import Path
import json
import librosa
import soundfile as sf
import numpy as np
import re
import noisereduce as nr
import torch

# Fix for PyTorch 2.6+ weights_only issue
# Allow safe loading of XTTS model checkpoints
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig])
except Exception as e:
    print(f"⚠️ Warning: Could not configure safe globals: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global TTS instance
tts_model = None
reference_voice = "sonuRecording_converted.wav"  # Properly converted WAV file (22050Hz, mono, PCM)
output_dir = Path("agent_audio_outputs")
output_dir.mkdir(exist_ok=True)

# Conversation state
conversations = {}


def load_tts_model():
    """Load TTS model on startup"""
    global tts_model
    print("📥 Loading XTTS v2 model...")
    tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False
    )
    print("✅ Model loaded successfully!")


def clean_text_for_tts(text):
    """
    Clean and normalize text for better TTS pronunciation
    - Removes all punctuation marks that shouldn't be spoken
    - Keeps only natural pauses (periods, commas)
    - Fixes common pronunciation issues
    """
    # Step 1: Remove all special punctuation marks that shouldn't be spoken
    # Remove: ! ? ; : " ' ` ~ @ # $ % ^ & * ( ) [ ] { } < > / \ | _ - + = 
    special_chars_to_remove = ['!', '?', ';', ':', '"', "'", '`', '~', '@', '#', 
                                '$', '%', '^', '&', '*', '(', ')', '[', ']', 
                                '{', '}', '<', '>', '/', '\\', '|', '_', '+', '=']
    
    for char in special_chars_to_remove:
        text = text.replace(char, '')
    
    # Step 2: Replace hyphens with spaces (for compound words)
    text = text.replace('-', ' ')
    
    # Step 3: Fix common pronunciation issues for English words
    pronunciation_fixes = {
        # Fix elongated vowel sounds
        r'\bname\b': 'name',
        r'\bteam\b': 'team',
        r'\bfrom\b': 'from',
        r'\bfull\b': 'full',
    }
    
    # Apply pronunciation fixes
    for pattern, replacement in pronunciation_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Step 4: Normalize punctuation (only periods and commas remain)
    # Replace multiple periods with single period
    text = re.sub(r'\.{2,}', '.', text)
    # Replace multiple commas with single comma
    text = re.sub(r',{2,}', ',', text)
    
    # Step 5: Add natural pauses after greetings
    text = re.sub(r'\b(Hi|Hello|Namaste)\b', r'\1,', text, flags=re.IGNORECASE)
    
    # Step 6: Ensure proper spacing around punctuation
    # Add space after periods and commas
    text = re.sub(r'([.,])(?=[^\s])', r'\1 ', text)
    # Remove space before periods and commas
    text = re.sub(r'\s+([.,])', r'\1', text)
    
    # Step 7: Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Step 8: Trim
    text = text.strip()
    
    # Step 9: Remove trailing comma if exists
    if text.endswith(','):
        text = text[:-1].strip()
    
    # Step 10: Ensure text ends with period for natural ending
    if text and text[-1] not in '.,':
        text += '.'
    
    return text


def split_long_text(text, max_length=200):
    """
    Split long text into smaller chunks to avoid XTTS token limit errors.
    Splits at sentence boundaries (periods, commas) when possible.
    
    Args:
        text: Text to split
        max_length: Maximum characters per chunk (default: 200)
    
    Returns:
        List of text chunks
    """
    # If text is short enough, return as single chunk
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences (periods)
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_length
        if len(current_chunk) + len(sentence) + 2 > max_length:  # +2 for '. '
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip() + '.')
                current_chunk = ""
            
            # If single sentence is too long, split by commas
            if len(sentence) > max_length:
                parts = sentence.split(',')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    if len(current_chunk) + len(part) + 2 > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip() + ',')
                            current_chunk = ""
                        
                        # If even a single part is too long, force split by words
                        if len(part) > max_length:
                            words = part.split()
                            for word in words:
                                if len(current_chunk) + len(word) + 1 > max_length:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                        current_chunk = ""
                                current_chunk += word + " "
                        else:
                            current_chunk = part + ", "
                    else:
                        current_chunk += part + ", "
            else:
                current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def concatenate_audio_files(file_paths):
    """
    Concatenates multiple audio files into a single WAV file.
    Assumes all input files are 16kHz mono WAV.
    
    Args:
        file_paths: List of paths to audio files to concatenate.
    
    Returns:
        Path to the concatenated audio file, or None if an error occurs.
    """
    if not file_paths:
        return None
    
    combined_audio = []
    sr = 16000  # Expected sample rate
    
    for f_path in file_paths:
        try:
            audio, current_sr = librosa.load(f_path, sr=None, dtype=np.float32)
            if current_sr != sr:
                print(f"Warning: Audio file {f_path} has sample rate {current_sr}Hz, expected {sr}Hz. Resampling.")
                audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
            combined_audio.append(audio)
        except Exception as e:
            print(f"Error loading audio file {f_path}: {e}")
            return None
    
    if not combined_audio:
        return None
        
    final_audio = np.concatenate(combined_audio)
    
    # Normalize combined audio to prevent clipping
    if np.max(np.abs(final_audio)) > 0:
        final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
    
    output_filename = output_dir / f"{str(uuid.uuid4())}_combined.wav"
    sf.write(str(output_filename), final_audio, sr, subtype='PCM_16')
    
    # Clean up individual chunk files
    for f_path in file_paths:
        try:
            Path(f_path).unlink()
        except OSError as e:
            print(f"Error deleting temporary audio file {f_path}: {e}")
            
    print(f"✅ Concatenated {len(file_paths)} audio chunks into {output_filename.name}")
    return str(output_filename)


def generate_speech(text, language="en", speed=5, temperature=0.1):
    """
    Generate speech with cloned voice - OPTIMIZED FOR QUALITY & CLARITY
    Automatically splits long texts to avoid token limit errors.
    
    Args:
        text: Text to speak
        language: Language code (en, hi, ta, te, etc.)
        speed: Playback speed multiplier (5 = very fast, user preference)
        temperature: Voice expressiveness (0.1 = consistent, user preference)
    
    Returns:
        Path to generated audio file (16kHz WAV, noise-reduced)
    """
    if not text or text.strip() == "":
        return None
    
    # Clean text for better TTS
    text = clean_text_for_tts(text)
    
    print(f"🧹 Cleaned text: {text}")
    print(f"📏 Text length: {len(text)} characters")
    
    # Check if text is too long and needs splitting
    # Increased to 400 to avoid splitting normal sentences
    MAX_TEXT_LENGTH = 400  # Characters
    
    if len(text) > MAX_TEXT_LENGTH:
        print(f"⚠️  Text is long ({len(text)} chars), splitting into chunks...")
        chunks = split_long_text(text, MAX_TEXT_LENGTH)
        print(f"📝 Split into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"🎤 Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            chunk_audio = generate_speech_chunk(chunk, language, speed, temperature)
            if chunk_audio:
                audio_chunks.append(chunk_audio)
        
        # Concatenate all audio chunks
        if audio_chunks:
            return concatenate_audio_files(audio_chunks)
        else:
            return None
    else:
        # Text is short enough, generate normally
        return generate_speech_chunk(text, language, speed, temperature)


def generate_speech_chunk(text, language="en", speed=5, temperature=0.1):
    """
    Generate speech for a single chunk of text (internal function).
    """
    # Generate unique filename
    audio_id = str(uuid.uuid4())
    temp_file = output_dir / f"{audio_id}_temp.wav"
    output_file = output_dir / f"{audio_id}.wav"
    
    try:
        # Step 1: Generate speech with TTS (24kHz output)
        # Optimized for CLEAR, EXPRESSIVE voice
        print(f"🎤 Generating ({language}): {text[:80]}...")
        tts_model.tts_to_file(
            text=text,
            speaker_wav=reference_voice,
            language=language,
            file_path=str(temp_file),
            speed=5.0,  # Generate at 1.0 for best quality
            temperature=0.1,  # Increased for more natural, expressive voice (was 0.1)
            length_penalty=1.0,
            repetition_penalty=15.0,  # Increased to prevent repetition and artifacts
            top_k=50,  # Increased for better voice variety (was 10)
            top_p=0.90,  # Increased for smoother speech (was 0.85)
            split_sentences=False  # Keep as one continuous speech
        )
        
        # Step 2: Load audio and process for clarity & speed
        print(f"⚡ Processing audio (enhanced clarity, speed={speed}x, 16kHz)...")
        
        # Load audio (will be 24kHz from XTTS) - force float64 to avoid numpy errors
        audio, sr = librosa.load(str(temp_file), sr=None, dtype=np.float64)
        
        # Apply GENTLE noise reduction to remove background noise without affecting voice quality
        # Reduced prop_decrease from 0.5 to 0.3 for more natural sound
        audio_clean = nr.reduce_noise(
            y=audio, 
            sr=sr, 
            stationary=True, 
            prop_decrease=0.3,  # Gentle noise reduction (was 0.5)
            freq_mask_smooth_hz=500,  # Smooth frequency masking
            time_mask_smooth_ms=50  # Smooth time masking
        )
        
        # Apply speed adjustment using resampling
        # NOTE: Only apply if speed is different from 1.0
        if speed != 5.0:
            # Resample to achieve speed change
            audio_clean = librosa.resample(audio_clean, orig_sr=sr, target_sr=int(sr * speed))
            sr_adjusted = int(sr * speed)
        else:
            sr_adjusted = sr
        
        # Resample to 16kHz for voice agent compatibility
        audio_16k = librosa.resample(audio_clean, orig_sr=sr_adjusted, target_sr=16000)
        
        # Normalize audio to prevent clipping and ensure consistent volume
        # Use peak normalization for clearer sound
        if np.max(np.abs(audio_16k)) > 0:
            audio_16k = audio_16k / np.max(np.abs(audio_16k)) * 0.95
        
        # Convert to float32 for saving
        audio_16k = audio_16k.astype(np.float32)
        
        # Save final audio at 16kHz
        sf.write(str(output_file), audio_16k, 16000, subtype='PCM_16')
        
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()
        
        print(f"✅ Generated: {output_file.name} (16kHz, {speed}x speed, {language}, noise-reduced)")
        return str(output_file)
    
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        return None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "reference_voice": reference_voice,
        "reference_voice_exists": os.path.exists(reference_voice)
    })


@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech with cloned voice
    
    Request body:
    {
        "text": "Text to speak",
        "language": "hi",  // optional, default: hi
        "speed": 5,      // optional, default: 1.5
        "temperature": 0.1 // optional, default: 0.75
    }
    
    Response:
    {
        "success": true,
        "audio_id": "uuid",
        "audio_url": "/api/audio/uuid"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' in request body"
            }), 400
        
        text = data['text']
        language = data.get('language', 'en')  # Default: English
        speed = data.get('speed', 5)  # Default: 1.8x (fast but clear)
        temperature = data.get('temperature', 0.75)  # Default: 0.75 (natural & expressive)
        
        # Generate speech
        audio_file = generate_speech(text, language, speed, temperature)
        
        if not audio_file:
            return jsonify({
                "success": False,
                "error": "Failed to generate speech"
            }), 500
        
        audio_id = Path(audio_file).stem
        
        return jsonify({
            "success": True,
            "audio_id": audio_id,
            "audio_url": f"/api/audio/{audio_id}",
            "text": text,
            "language": language
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_audio(audio_id):
    """
    Get generated audio file
    
    Returns:
        WAV audio file
    """
    audio_file = output_dir / f"{audio_id}.wav"
    
    if not audio_file.exists():
        return jsonify({
            "success": False,
            "error": "Audio file not found"
        }), 404
    
    return send_file(
        audio_file,
        mimetype='audio/wav',
        as_attachment=False
    )


@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """
    Start a new conversation session
    
    Response:
    {
        "success": true,
        "session_id": "uuid",
        "greeting": "नमस्ते! मैं आपकी एडमिशन काउंसलर हूं।"
    }
    """
    try:
        session_id = str(uuid.uuid4())
        
        conversations[session_id] = {
            "history": [],
            "created_at": time.time(),
            "language": "hi"
        }
        
        greeting = "नमस्ते! मैं आपकी एडमिशन काउंसलर हूं। मैं आपकी कैसे मदद कर सकती हूं?"
        
        # Add greeting to history
        conversations[session_id]["history"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": time.time()
        })
        
        # Generate greeting audio
        audio_file = generate_speech(greeting, language="hi")
        audio_id = Path(audio_file).stem if audio_file else None
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "greeting": greeting,
            "audio_id": audio_id,
            "audio_url": f"/api/audio/{audio_id}" if audio_id else None
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/conversation/message', methods=['POST'])
def send_message():
    """
    Send a message in a conversation
    
    Request body:
    {
        "session_id": "uuid",
        "message": "User message",
        "language": "hi"  // optional
    }
    
    Response:
    {
        "success": true,
        "response": "Agent response text",
        "audio_id": "uuid",
        "audio_url": "/api/audio/uuid"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'session_id' not in data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'session_id' or 'message' in request body"
            }), 400
        
        session_id = data['session_id']
        user_message = data['message']
        language = data.get('language', 'hi')
        
        # Check if session exists
        if session_id not in conversations:
            return jsonify({
                "success": False,
                "error": "Invalid session_id"
            }), 404
        
        # Add user message to history
        conversations[session_id]["history"].append({
            "role": "user",
            "content": user_message,
            "timestamp": time.time()
        })
        
        # Generate response (integrate with your LLM here)
        response_text = generate_response(user_message, language)
        
        # Add response to history
        conversations[session_id]["history"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": time.time()
        })
        
        # Generate speech
        audio_file = generate_speech(response_text, language)
        audio_id = Path(audio_file).stem if audio_file else None
        
        return jsonify({
            "success": True,
            "response": response_text,
            "audio_id": audio_id,
            "audio_url": f"/api/audio/{audio_id}" if audio_id else None
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/conversation/history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history"""
    if session_id not in conversations:
        return jsonify({
            "success": False,
            "error": "Invalid session_id"
        }), 404
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "history": conversations[session_id]["history"]
    })


def generate_response(user_message, language="hi"):
    """
    Generate response to user message
    
    TODO: Integrate with your LLM (OpenAI, Gemini, etc.)
    """
    # Simple keyword-based responses (replace with actual LLM)
    responses_hi = {
        "नमस्ते": "नमस्ते! मैं आपकी एडमिशन काउंसलर हूं। मैं आपकी कैसे मदद कर सकती हूं?",
        "admission": "मैं आपको कॉलेज एडमिशन में मदद कर सकती हूं। आप किस कोर्स में रुचि रखते हैं?",
        "engineering": "Engineering एक बहुत अच्छा विकल्प है। आप किस branch में interested हैं?",
        "help": "मैं आपको कॉलेज एडमिशन, कोर्स सिलेक्शन और करियर गाइडेंस में मदद कर सकती हूं।",
    }
    
    responses_en = {
        "hello": "Hello! I'm your admission counselor. How can I help you today?",
        "admission": "I can help you with college admissions. Which course are you interested in?",
        "engineering": "Engineering is a great choice! Which branch are you interested in?",
        "help": "I can help you with college admissions, course selection, and career guidance.",
    }
    
    responses = responses_hi if language == "hi" else responses_en
    
    # Simple keyword matching
    user_lower = user_message.lower().strip()
    for key, response in responses.items():
        if key in user_lower:
            return response
    
    # Default response
    if language == "hi":
        return "मुझे समझ नहीं आया। क्या आप फिर से बता सकते हैं?"
    else:
        return "I didn't understand that. Could you please rephrase?"


@app.route('/api/voices', methods=['GET'])
def list_voices():
    """List available reference voices"""
    voices = []
    
    for file in Path(".").glob("*.wav"):
        if "my_hindi_voice" in file.name or "indian_voice" in file.name:
            voices.append({
                "name": file.name,
                "path": str(file),
                "size": file.stat().st_size
            })
    
    for file in Path(".").glob("*.mp3"):
        if "my_hindi_voice" in file.name:
            voices.append({
                "name": file.name,
                "path": str(file),
                "size": file.stat().st_size
            })
    
    return jsonify({
        "success": True,
        "voices": voices,
        "current_voice": reference_voice
    })


@app.route('/api/voices/set', methods=['POST'])
def set_reference_voice():
    """
    Set reference voice
    
    Request body:
    {
        "voice_path": "path/to/voice.wav"
    }
    """
    global reference_voice
    
    try:
        data = request.get_json()
        
        if not data or 'voice_path' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'voice_path' in request body"
            }), 400
        
        voice_path = data['voice_path']
        
        if not os.path.exists(voice_path):
            return jsonify({
                "success": False,
                "error": f"Voice file not found: {voice_path}"
            }), 404
        
        reference_voice = voice_path
        
        return jsonify({
            "success": True,
            "reference_voice": reference_voice
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def cleanup_old_audio_files():
    """Cleanup audio files older than 1 hour"""
    while True:
        try:
            current_time = time.time()
            for audio_file in output_dir.glob("*.wav"):
                file_age = current_time - audio_file.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    audio_file.unlink()
                    print(f"🗑️ Deleted old audio file: {audio_file.name}")
        except Exception as e:
            print(f"❌ Error in cleanup: {e}")
        
        time.sleep(600)  # Run every 10 minutes


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎙️ WEB-BASED REALTIME VOICE AGENT")
    print("="*70)
    
    # Check reference voice
    if not os.path.exists(reference_voice):
        print(f"\n❌ Error: Reference voice not found: {reference_voice}")
        print("Please make sure your reference voice file exists.")
        exit(1)
    
    print(f"\n✅ Reference voice: {reference_voice}")
    
    # Load TTS model
    load_tts_model()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_audio_files, daemon=True)
    cleanup_thread.start()
    
    print("\n" + "="*70)
    print("🚀 Starting Flask server...")
    print("="*70)
    print("\n📡 API Endpoints:")
    print("   GET  /api/health                 - Health check")
    print("   POST /api/tts                    - Text to speech")
    print("   GET  /api/audio/<audio_id>       - Get audio file")
    print("   POST /api/conversation/start     - Start conversation")
    print("   POST /api/conversation/message   - Send message")
    print("   GET  /api/conversation/history   - Get history")
    print("   GET  /api/voices                 - List voices")
    print("   POST /api/voices/set             - Set reference voice")
    
    print("\n🌐 Server running at: http://localhost:5000")
    print("="*70 + "\n")
    
    # Run Flask app
    # Use PORT environment variable for cloud deployment (Render, Railway, etc.)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )

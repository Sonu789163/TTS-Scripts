# 🎙️ TTS Server - Cloned Voice with XTTS

Voice cloning TTS server using Coqui XTTS v2 for real-time voice agents.

---

## 🚀 Quick Deploy to Render

### **Prerequisites**
- GitHub account
- Render account ([render.com](https://render.com))

### **Deployment Steps**

1. **Fork/Clone this repository**

2. **Go to Render Dashboard**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   - **Name**: `tts-server`
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Instance Type**: 
     - `Free` - For testing (slower, spins down after 15 min)
     - `Starter` ($7/month) - For production (faster, always on)

4. **Deploy**
   - Click "Create Web Service"
   - Wait 10-15 minutes for build

5. **Get Your URL**
   - After deployment: `https://tts-server-xxxx.onrender.com`

6. **Test**
   ```bash
   curl https://tts-server-xxxx.onrender.com/api/health
   ```

---

## 🐳 Local Docker Development

### **Build Image**
```bash
docker build -t tts-server .
```

### **Run Container**
```bash
docker run -p 5000:5000 tts-server
```

### **Test**
```bash
curl http://localhost:5000/api/health
```

---

## 📋 API Endpoints

### **Health Check**
```bash
GET /api/health
```

### **Generate Speech**
```bash
POST /api/tts
Content-Type: application/json

{
  "text": "Hello, this is a test",
  "language": "en",
  "speed": 1.5,
  "temperature": 0.75
}
```

### **Get Audio**
```bash
GET /api/audio/<audio_id>
```

---

## ⚙️ Configuration

### **Environment Variables**
- `PORT` - Server port (default: 5000)

### **Reference Voice**
- Place your reference audio file as `sonuRecording_converted.wav`
- Format: WAV, 16-bit PCM, mono, 16-22kHz

---

## 🔧 Local Development

### **Setup**
```bash
# Create virtual environment
python3.11 -m venv coqui_env

# Activate
source coqui_env/bin/activate  # Mac/Linux
# or
coqui_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Run**
```bash
python web_voice_agent_integration.py
```

---

## 📊 Performance

### **Render Free Tier**
- Latency: 10-20 seconds
- Spins down after 15 min inactivity
- Wake-up time: 30-60 seconds

### **Render Starter ($7/month)**
- Latency: 5-10 seconds
- Always on
- Better performance

### **With GPU (GCP/AWS)**
- Latency: 2-4 seconds
- Requires GPU setup
- Higher cost (~$150-200/month)

---

## 🛠️ Tech Stack

- **TTS Engine**: Coqui XTTS v2
- **Framework**: Flask
- **Audio Processing**: librosa, soundfile, noisereduce
- **ML**: PyTorch, Transformers

---

## 📝 License

This project uses Coqui TTS which requires:
- Commercial license from Coqui for commercial use
- Or agreement to CPML for non-commercial use

---

## 🆘 Troubleshooting

### **Deployment fails on Render**
- Check build logs for errors
- Ensure `requirements.txt` is correct
- Verify `sonuRecording_converted.wav` exists

### **Slow performance**
- Upgrade to Starter plan ($7/month)
- Or use ElevenLabs API for faster results

### **Out of memory**
- Upgrade instance type
- Reduce model size (not recommended)

---

## 🔗 Integration

### **With Voice Agent Backend**

Update your backend's `clonedVoice.js`:

```javascript
const TTS_API_URL = 'https://tts-server-xxxx.onrender.com';
```

---

## 📞 Support

For issues or questions, please open a GitHub issue.

---

**Deployed with ❤️ using Render**

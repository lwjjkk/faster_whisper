import socket  # Add socket for UDP functionality
import pyaudio                          # pip install pyaudio
import numpy as np                      # pip install numpy
import wave                             # pip install wave
import os                               # pip install os
import tempfile                         # pip install tempfile
from faster_whisper import WhisperModel # pip install faster-whisper
import webrtcvad                        # pip install webrtcvad
                                        # pip install cuda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# UDP Configuration
UDP_IP = "127.0.0.1"  # localhost
UDP_PORT = 11999
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Keyword mappings (both English and Chinese)
keywords = {
    "angry": "angry", "生氣": "angry",
    "excited": "excited", "興奮": "excited",
    "dancing": "dancing", "跳舞": "dancing",
    "running": "running", "跑步": "running",
    "sad": "sad", "傷心": "sad",
    "happy": "happy", "快樂": "happy",
    "headshake": "headshake", "搖頭": "headshake",
    "cheering": "cheering", "歡呼": "cheering",
    "talking": "talking", "說話": "talking"
}

# Function to send UDP messages
def send_udp(message):
    try:
        udp_socket.sendto(message.encode(), (UDP_IP, UDP_PORT))
        print(f"UDP sent: {message}")
    except Exception as e:
        print(f"UDP error: {e}")

# Configuration for PyAudio
FORMAT = pyaudio.paInt16      # 16-bit int sampling format
CHANNELS = 1                  # Mono audio
RATE = 16000                 # Sampling rate expected by Whisper (16kHz)
CHUNK = 1024                 # Buffer size

# Initialize PyAudio and Whisper
p = pyaudio.PyAudio()
model = WhisperModel("turbo", device="cuda", compute_type="float16")
vad = webrtcvad.Vad(2) # Set VAD mode (0-3, 3 is most aggressive)

# Open microphone stream
def is_speech(frame, sample_rate):
    return vad.is_speech(frame, sample_rate)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting streaming microphone transcription. Press Ctrl+C to stop.")

try:
    while True:
        frames = []
        silence_chunks = 0
        speaking = False

        print("\nListening for speech...Press Ctrl+C to stop.")

        while True:
            data = stream.read(CHUNK)
            # VAD expects 20ms, 30ms, or 10ms frames. 1024 samples at 16kHz is 64ms, so trim or split as needed.
            frame = data[:640]  # 20ms at 16kHz, 16-bit mono = 640 bytes
            if is_speech(frame, RATE):
                frames.append(data)
                silence_chunks = 0
                speaking = True
            else:
                if speaking:
                    silence_chunks += 1
                    frames.append(data)
                # Stop after 1 second of silence (adjust as needed)
                if silence_chunks > int(RATE / CHUNK * 1):
                    break

        if frames:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

            # Transcribe audio chunk with faster-whisper
            segments, info = model.transcribe(tmp_file.name, beam_size=5)

            # Print detected language
            print(f"🌐 Language detected: {info.language} (Confidence: {info.language_probability:.2f})")

            # Print transcription segments and check for keywords
            for segment in segments:
                text = segment.text.lower()
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                
                # Check for keywords in the transcription
                for keyword, response in keywords.items():
                    if keyword.lower() in text:
                        send_udp(response)
                        print(f"Keyword detected: {keyword} -> Sent: {response}")
                        break  # Only send one response per segment

            # Remove temporary audio file
            os.remove(tmp_file.name)

except KeyboardInterrupt:
    print("\n✅ Stopped transcription.")

finally:
    # Clean up PyAudio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    udp_socket.close()

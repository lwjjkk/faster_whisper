import socket  # Add socket for UDP functionality
import pip
import pyaudio
import wave
import tempfile
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

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
RECORD_SECONDS = 5           # Duration per audio chunk

# Initialize PyAudio and Whisper
p = pyaudio.PyAudio()
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Starting streaming microphone transcription. Press Ctrl+C to stop.")

try:
    while True:
        print(f"\n🎤 Recording {RECORD_SECONDS} seconds...Press Ctrl+C to stop.")
        frames = []

        # Read audio data chunk by chunk
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Save recorded chunk to a temporary WAV file
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

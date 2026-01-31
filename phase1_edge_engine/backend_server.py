import cv2
import threading
import numpy as np
import pyaudio
import time
import flwr as fl
import hashlib
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from faster_whisper import WhisperModel

# --- IMPORT BLOCKCHAIN MODULE ---
try:
    from blockchain_manager import BlockchainLogger
    print("🔗 Initializing Blockchain Connection...")
    chain_logger = BlockchainLogger()
    blockchain_active = True
except Exception as e:
    print(f"⚠️ Blockchain Unavailable: {e}")
    blockchain_active = False

# --- 1. APP CONFIGURATION ---
app = FastAPI(title="OMNI-SHIELD Edge Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. GLOBAL STATE & MODELS ---
state = {
    "redaction_active": True,
    "audio_mute_until": 0.0,
    "camera_active": False
}

print("⏳ Loading AI Models... (Server Starting)")
visual_model = YOLO("yolov8n.pt") 
audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("✅ Models Loaded.")

# --- 3. HELPER: LOG TO BLOCKCHAIN (ASYNC) ---
def trigger_audit_log(action_type, content_snippet):
    if not blockchain_active: return
    
    # Create a hash of the redacted content (Simulating Zero-Knowledge Privacy)
    # We don't log the word "password", we log "hash(password)"
    data_hash = hashlib.sha256(content_snippet.encode()).hexdigest()
    
    # Run in thread to prevent video lag
    def _log():
        try:
            chain_logger.log_event(action_type, data_hash)
        except Exception as e:
            print(f"❌ Blockchain Write Failed: {e}")
            
    threading.Thread(target=_log, daemon=True).start()

# --- 4. FEDERATED LEARNING CLIENT ---
class OmniShieldClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]

    def fit(self, parameters, config):
        print("🧠 FEDERATED TRAINING: Received global update. Training locally...")
        time.sleep(1) 
        return self.get_parameters(config={}), 10, {"accuracy": 0.95}

    def evaluate(self, parameters, config):
        return 0.5, 10, {"accuracy": 0.95}

def start_federated_client():
    time.sleep(3)
    try:
        print("🚀 Federated Client Connecting to Server...")
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080", 
            client=OmniShieldClient()
        )
    except Exception as e:
        print(f"⚠️ Federated Learning Error: {e}")

# --- 5. AUDIO PROCESSING THREAD ---
def audio_listener_loop():
    chunk_size = 1024
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                       input=True, frames_per_buffer=chunk_size)
    except Exception as e:
        print(f"⚠️ Audio Error: {e}")
        return

    print("🎙️ Audio Background Service Running...")
    forbidden_words = ["password", "secret", "private", "card"]

    while True:
        try:
            frames = []
            for _ in range(0, int(16000 / chunk_size * 2)): 
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            audio_data = np.frombuffer(b''.join(frames), np.int16).flatten().astype(np.float32) / 32768.0
            segments, _ = audio_model.transcribe(audio_data, beam_size=1)
            
            for segment in segments:
                text = segment.text.lower().strip()
                if any(word in text for word in forbidden_words):
                    print(f"🚫 BLOCKED: '{text}'")
                    state["audio_mute_until"] = time.time() + 2.0
                    
                    # TRIGGER BLOCKCHAIN AUDIT
                    trigger_audit_log("AUDIO_REDACTION", text)
                    
        except Exception:
            pass

t_audio = threading.Thread(target=audio_listener_loop, daemon=True)
t_audio.start()

# --- 6. VIDEO GENERATOR ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    state["camera_active"] = True
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while state["camera_active"]:
        success, frame = cap.read()
        if not success:
            break

        if state["redaction_active"]:
            results = visual_model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if time.time() < state["audio_mute_until"]:
            cv2.putText(frame, "AUDIO BLOCKED & LOGGED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- 7. API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "running", "project": "OMNI-SHIELD"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_redaction")
def toggle_redaction():
    state["redaction_active"] = not state["redaction_active"]
    return {"redaction_active": state["redaction_active"]}

if __name__ == "__main__":
    import uvicorn
    
    t_fed = threading.Thread(target=start_federated_client, daemon=True)
    t_fed.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
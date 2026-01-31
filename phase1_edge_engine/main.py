import cv2
import threading
import numpy as np
import pyaudio
import time
from ultralytics import YOLO
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
redaction_active = True        # Master switch for redaction
blur_intensity = (99, 99)      # Kernel size for Gaussian Blur (must be odd numbers)
forbidden_words = ["password", "secret", "private", "card"] # Words to mute
audio_mute_duration = 2.0      # Seconds to mute when a word is detected

# --- INITIALIZE MODELS ---
print("Loading YOLOv8 model (this may take a moment)...")
visual_model = YOLO("yolov8n.pt")  # Auto-downloads the 'nano' (fastest) model

print("Loading Whisper model...")
# 'tiny' is fastest for CPU; use 'base' or 'small' if you have a GPU
audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# --- SHARED STATE ---
audio_mute_until = 0.0  # Timestamp: Audio is muted until this time

def process_audio_stream():
    """
    Continuously listens to the microphone, transcribes in real-time,
    and sets a mute flag if a forbidden word is detected.
    """
    global audio_mute_until
    
    # PyAudio Setup
    chunk_size = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    
    print("🎤 Audio Listener Active. Speak 'password' to test muting.")

    while True:
        # 1. Capture 2 seconds of audio for transcription context
        frames = []
        for _ in range(0, int(rate / chunk_size * 2)): 
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        
        # 2. Convert raw bytes to float32 numpy array for Whisper
        audio_data = np.frombuffer(b''.join(frames), np.int16).flatten().astype(np.float32) / 32768.0

        # 3. Transcribe
        segments, _ = audio_model.transcribe(audio_data, beam_size=5, language="en")
        
        for segment in segments:
            text = segment.text.lower().strip()
            if text:
                print(f"heard: {text}") # Debug print
                
            # 4. Check for forbidden words
            if any(word in text for word in forbidden_words):
                print(f"🚫 SENSITIVE AUDIO DETECTED: '{text}' -> MUTING!")
                audio_mute_until = time.time() + audio_mute_duration

def main():
    global audio_mute_until
    
    # Start Audio Thread
    audio_thread = threading.Thread(target=process_audio_stream, daemon=True)
    audio_thread.start()

    # Start Video Capture (Webcam 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("📷 Visual Engine Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. VISUAL REDACTION (YOLO) ---
        if redaction_active:
            # Run YOLO inference
            results = visual_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Check class ID (0 is 'person' in COCO dataset)
                    cls_id = int(box.cls[0])
                    
                    # You can change this to only redact faces if you train a custom model.
                    # Standard YOLOv8 detects full 'person'. For this prototype, we blur the 'person'.
                    if cls_id == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract Region of Interest (ROI)
                        roi = frame[y1:y2, x1:x2]
                        
                        # Apply Gaussian Blur
                        if roi.size > 0:
                            blurred_roi = cv2.GaussianBlur(roi, blur_intensity, 0)
                            frame[y1:y2, x1:x2] = blurred_roi
                            
                            # Draw a red border to show it's detected
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "REDACTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- 2. AUDIO REDACTION INDICATOR ---
        if time.time() < audio_mute_until:
            # Overlay a big "MUTED" icon/text
            cv2.putText(frame, "🔇 AUDIO MUTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow("OMNI-SHIELD Phase 1 (Prototype)", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
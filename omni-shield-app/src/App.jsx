import { useState } from 'react';
import './App.css';

function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState("System Ready");

  const toggleCamera = async () => {
    if (!isStreaming) {
      setIsStreaming(true);
      setStatus("Secure Stream Active - Redacting...");
    } else {
      setIsStreaming(false);
      setStatus("Stream Stopped");
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>OMNI-SHIELD</h1>
        <span className="status-badge">{status}</span>
      </header>

      <main className="main-content">
        <div className="video-wrapper">
          {isStreaming ? (
            <img 
              src="http://localhost:8000/video_feed" 
              alt="Secure Redaction Stream" 
              className="live-feed"
            />
          ) : (
            <div className="placeholder">
              <div className="lock-icon">🔒</div>
              <h2>Edge Engine Offline</h2>
              <p>Click Start to initialize local redaction</p>
            </div>
          )}
        </div>

        <div className="control-panel">
          <button onClick={toggleCamera} className={isStreaming ? "btn stop" : "btn start"}>
            {isStreaming ? "TERMINATE STREAM" : "INITIALIZE SECURE CAM"}
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
import { useState, useRef, useEffect, useCallback } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const MAX_CANVAS_W = 660;
const MAX_CANVAS_H = 560;

/* ── Tiny icon components ─────────────────────────────────────────────── */
const SpinnerIcon = () => (
  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);
const XIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);
const RefreshIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25
         8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
);
const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
  </svg>
);
const PencilIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487z" />
  </svg>
);

/* ── Helper ──────────────────────────────────────────────────────────── */
const maskValue = (val) => {
  const s = String(val ?? '');
  return s.length <= 2 ? `${s}***` : `${s.slice(0, 2)}***`;
};

/* ── Main component ──────────────────────────────────────────────────── */
export default function RedactionEditor({ originalUrl, aiBoxes, fileHash, originalFile, ownerAddress, onFinalise, onReRun }) {
  const canvasRef    = useRef(null);
  const imgElRef     = useRef(null);   // native Image object
  const [imgLoaded,  setImgLoaded]  = useState(false);
  const [scale,      setScale]      = useState({ x: 1, y: 1 });
  const [canvasSize, setCanvasSize] = useState({ w: MAX_CANVAS_W, h: MAX_CANVAS_H });

  // Valid AI boxes only (need pixel coords)
  const validAiBoxes = (aiBoxes || []).filter(
    (b) => Array.isArray(b.box) && b.box.length === 4,
  );

  // Debug: log incoming AI boxes on mount so coordinate issues are visible in console
  useEffect(() => {
    console.log('[RedactionEditor] aiBoxes on mount:', aiBoxes);
    console.log('[RedactionEditor] validAiBoxes (have .box[4]):', validAiBoxes);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [activeAiBoxes, setActiveAiBoxes] = useState(validAiBoxes);
  const [userBoxes,     setUserBoxes]     = useState([]);
  const [history,       setHistory]       = useState([]); // stack of previous userBoxes states
  const [selectedBox,   setSelectedBox]   = useState(null); // {type:'ai'|'user', index}
  const [drawMode,      setDrawMode]      = useState(false);

  // Drawing state (refs to avoid stale closures in mouse handlers)
  const drawingRef   = useRef(false);
  const drawStartRef = useRef(null);
  const [drawPreview, setDrawPreview] = useState(null); // {x,y,w,h} in canvas px

  const [isFinalising, setIsFinalising] = useState(false);

  /* ── Load image, compute scale ─────────────────────────────────────── */
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgElRef.current = img;
      const sx = MAX_CANVAS_W / img.naturalWidth;
      const sy = MAX_CANVAS_H / img.naturalHeight;
      const s  = Math.min(sx, sy, 1); // never upscale
      setScale({ x: s, y: s });
      setCanvasSize({
        w: Math.round(img.naturalWidth  * s),
        h: Math.round(img.naturalHeight * s),
      });
      setImgLoaded(true);
    };
    img.src = originalUrl;
  }, [originalUrl]);

  /* ── Redraw canvas ─────────────────────────────────────────────────── */
  const redraw = useCallback(() => {
    if (!imgLoaded || !canvasRef.current || !imgElRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const { w, h } = canvasSize;
    const { x: sx, y: sy } = scale;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(imgElRef.current, 0, 0, w, h);

    // AI / custom boxes
    activeAiBoxes.forEach((b, i) => {
      const [x1, y1, x2, y2] = b.box;
      const dx = x1 * sx, dy = y1 * sy, dw = (x2 - x1) * sx, dh = (y2 - y1) * sy;
      const isCustom = b.source === 'custom';
      ctx.fillStyle   = isCustom ? 'rgba(123,28,28,0.55)' : 'rgba(220,38,38,0.35)';
      ctx.strokeStyle = isCustom ? 'rgba(220,38,38,0.9)'  : 'rgba(239,68,68,0.8)';
      ctx.fillRect(dx, dy, dw, dh);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(dx, dy, dw, dh);
      if (selectedBox?.type === 'ai' && selectedBox?.index === i) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(dx - 1, dy - 1, dw + 2, dh + 2);
      }
    });

    // User boxes — black solid
    userBoxes.forEach((b, i) => {
      const dx = b.xmin * sx, dy = b.ymin * sy;
      const dw = (b.xmax - b.xmin) * sx, dh = (b.ymax - b.ymin) * sy;
      ctx.fillStyle = 'rgba(0,0,0,0.88)';
      ctx.fillRect(dx, dy, dw, dh);
      if (selectedBox?.type === 'user' && selectedBox?.index === i) {
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(dx - 1, dy - 1, dw + 2, dh + 2);
      }
    });

    // Live draw preview
    if (drawPreview) {
      const { x, y, w: pw, h: ph } = drawPreview;
      ctx.fillStyle   = 'rgba(0,0,0,0.25)';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.fillRect(x, y, pw, ph);
      ctx.strokeRect(x, y, pw, ph);
      ctx.setLineDash([]);
    }
  }, [imgLoaded, canvasSize, scale, activeAiBoxes, userBoxes, selectedBox, drawPreview]);

  useEffect(() => { redraw(); }, [redraw]);

  /* ── Canvas coordinate helpers ─────────────────────────────────────── */
  const getCanvasPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width  / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
  };

  const hitTest = (pos) => {
    const { x: sx, y: sy } = scale;
    for (let i = userBoxes.length - 1; i >= 0; i--) {
      const b = userBoxes[i];
      if (pos.x >= b.xmin * sx && pos.x <= b.xmax * sx &&
          pos.y >= b.ymin * sy && pos.y <= b.ymax * sy) {
        return { type: 'user', index: i };
      }
    }
    for (let i = activeAiBoxes.length - 1; i >= 0; i--) {
      const [x1, y1, x2, y2] = activeAiBoxes[i].box;
      if (pos.x >= x1 * sx && pos.x <= x2 * sx &&
          pos.y >= y1 * sy && pos.y <= y2 * sy) {
        return { type: 'ai', index: i };
      }
    }
    return null;
  };

  /* ── Mouse events ──────────────────────────────────────────────────── */
  const handleMouseDown = (e) => {
    if (!imgLoaded) return;
    const pos = getCanvasPos(e);

    if (drawMode) {
      // Draw mode: always start a potential draw, deselect any selection
      setSelectedBox(null);
      drawingRef.current = true;
      drawStartRef.current = pos;
      setDrawPreview({ x: pos.x, y: pos.y, w: 0, h: 0 });
    } else {
      // Select mode: click to select/deselect, no drawing
      const hit = hitTest(pos);
      if (hit) {
        setSelectedBox(hit);
      } else {
        setSelectedBox(null);
      }
      drawingRef.current = false;
    }
  };

  const handleMouseMove = (e) => {
    if (!drawingRef.current || !drawStartRef.current) return;
    const pos  = getCanvasPos(e);
    const x    = Math.min(drawStartRef.current.x, pos.x);
    const y    = Math.min(drawStartRef.current.y, pos.y);
    const w    = Math.abs(pos.x - drawStartRef.current.x);
    const h    = Math.abs(pos.y - drawStartRef.current.y);
    setDrawPreview({ x, y, w, h });
  };

  const handleMouseUp = (e) => {
    if (!drawingRef.current || !drawStartRef.current) return;
    const pos = getCanvasPos(e);
    const dw  = Math.abs(pos.x - drawStartRef.current.x);
    const dh  = Math.abs(pos.y - drawStartRef.current.y);

    if (dw >= 10 && dh >= 10) {
      const { x: sx, y: sy } = scale;
      const xmin = Math.min(drawStartRef.current.x, pos.x) / sx;
      const ymin = Math.min(drawStartRef.current.y, pos.y) / sy;
      const xmax = Math.max(drawStartRef.current.x, pos.x) / sx;
      const ymax = Math.max(drawStartRef.current.y, pos.y) / sy;
      // Save snapshot before mutating (cap at 20 entries)
      setHistory(prev => [...prev.slice(-19), userBoxes]);
      setUserBoxes((prev) => [...prev, { xmin, ymin, xmax, ymax, label: 'manual' }]);
    }

    drawingRef.current   = false;
    drawStartRef.current = null;
    setDrawPreview(null);
  };

  /* ── Keyboard shortcuts ────────────────────────────────────────────── */
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      // Ctrl+Z — undo last user-drawn box
      if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        if (history.length === 0) return;
        const prev = history[history.length - 1];
        setHistory(h => h.slice(0, -1));
        setUserBoxes(prev);
        setSelectedBox(null);
        return;
      }

      // Del / Backspace — delete selected box
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (!selectedBox) return;
        e.preventDefault();
        if (selectedBox.type === 'user') {
          setHistory(prev => [...prev.slice(-19), userBoxes]);
          setUserBoxes(prev => prev.filter((_, i) => i !== selectedBox.index));
        } else {
          setActiveAiBoxes(prev => prev.filter((_, i) => i !== selectedBox.index));
        }
        setSelectedBox(null);
        return;
      }

      // R — toggle draw mode
      if (e.key === 'r' || e.key === 'R') {
        setDrawMode(prev => !prev);
        return;
      }

      // Escape — cancel draw / deselect / exit draw mode
      if (e.key === 'Escape') {
        if (drawingRef.current) {
          drawingRef.current = false;
          drawStartRef.current = null;
          setDrawPreview(null);
        } else if (selectedBox !== null) {
          setSelectedBox(null);
        } else if (drawMode) {
          setDrawMode(false);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [history, userBoxes, selectedBox, drawMode]);

  const removeBox = useCallback(({ type, index }) => {
    if (type === 'ai') {
      setActiveAiBoxes((prev) => prev.filter((_, i) => i !== index));
    } else {
      setUserBoxes((prev) => prev.filter((_, i) => i !== index));
    }
    setSelectedBox(null);
  }, []);

  /* ── Finalise ──────────────────────────────────────────────────────── */
  const handleFinalise = async () => {
    setIsFinalising(true);
    try {
      const allBoxes = [
        ...activeAiBoxes.map((b) => ({
          xmin: b.box[0], ymin: b.box[1], xmax: b.box[2], ymax: b.box[3],
          label: b.label || 'ai',
        })),
        ...userBoxes,
      ];
      const res = await fetch(`${API_BASE_URL}/redact/finalise`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ file_hash: fileHash, original_file: originalFile || '', boxes: allBoxes, owner_address: ownerAddress || '' }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      onFinalise(await res.json());
    } catch (err) {
      alert(`Finalise failed: ${err.message}`);
    } finally {
      setIsFinalising(false);
    }
  };

  /* ── Render ────────────────────────────────────────────────────────── */
  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Review Redactions</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            {drawMode
              ? 'Draw mode — drag to add boxes · click empty area to deselect'
              : 'Select mode — click a box to select · press R to draw'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Draw mode toggle */}
          <button
            onClick={() => setDrawMode(prev => !prev)}
            title="Toggle draw mode (R)"
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-sm font-medium
                        border transition-all
                        ${drawMode
                          ? 'border-indigo-500/70 text-indigo-300 bg-indigo-500/15'
                          : 'border-slate-600/70 text-slate-400 hover:text-white hover:border-indigo-500/60'}`}
          >
            <PencilIcon />
            Draw <span className="opacity-60 text-xs">(R)</span>
          </button>
          <button
            onClick={onReRun}
            className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium
                       border border-slate-600/70 text-slate-400 hover:text-white
                       hover:border-indigo-500/60 transition-all"
          >
            <RefreshIcon /> Re-run AI
          </button>
        </div>
      </div>

      <div className="flex gap-5 items-start" style={{ flexWrap: 'wrap' }}>

        {/* ── Canvas ──────────────────────────────────────────────── */}
        <div className="flex-1 min-w-0">
          <div
            className="glass rounded-2xl overflow-hidden relative"
            style={{ lineHeight: 0 }}
          >
            {!imgLoaded && (
              <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
                Loading image…
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={canvasSize.w}
              height={canvasSize.h}
              style={{
                display:    imgLoaded ? 'block' : 'none',
                width:      '100%',
                cursor:     drawMode ? 'crosshair' : 'default',
                userSelect: 'none',
              }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-red-600/70 border border-red-500/60" />
              AI detected
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-[#7B1C1C]/80 border border-red-600/60" />
              Custom terms
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-slate-700 border border-slate-600" />
              Your additions
            </span>
          </div>

          {/* Keyboard shortcut hint bar */}
          <div style={{
            display: 'flex',
            gap: '16px',
            marginTop: '10px',
            fontSize: '11px',
            color: 'var(--color-text-tertiary, #64748b)',
            flexWrap: 'wrap',
          }}>
            <span>
              <kbd style={{
                background: 'rgba(51,65,85,0.8)',
                border: '0.5px solid rgba(100,116,139,0.5)',
                borderRadius: '4px',
                padding: '1px 5px',
                fontFamily: 'monospace',
                fontSize: '10px',
                color: '#94a3b8',
              }}>R</kbd>{' '}draw mode
            </span>
            <span>
              <kbd style={{
                background: 'rgba(51,65,85,0.8)',
                border: '0.5px solid rgba(100,116,139,0.5)',
                borderRadius: '4px',
                padding: '1px 5px',
                fontFamily: 'monospace',
                fontSize: '10px',
                color: '#94a3b8',
              }}>Del</kbd>{' '}delete selected
            </span>
            <span>
              <kbd style={{
                background: 'rgba(51,65,85,0.8)',
                border: '0.5px solid rgba(100,116,139,0.5)',
                borderRadius: '4px',
                padding: '1px 5px',
                fontFamily: 'monospace',
                fontSize: '10px',
                color: '#94a3b8',
              }}>Ctrl+Z</kbd>{' '}undo
            </span>
            <span>
              <kbd style={{
                background: 'rgba(51,65,85,0.8)',
                border: '0.5px solid rgba(100,116,139,0.5)',
                borderRadius: '4px',
                padding: '1px 5px',
                fontFamily: 'monospace',
                fontSize: '10px',
                color: '#94a3b8',
              }}>Esc</kbd>{' '}deselect
            </span>
          </div>
        </div>

        {/* ── Controls ────────────────────────────────────────────── */}
        <div className="w-64 flex-shrink-0 space-y-4">

          {/* Count badges */}
          <div className="flex gap-2">
            <span className="flex-1 text-center text-xs font-semibold px-3 py-2 rounded-xl
                             bg-red-500/15 text-red-400 border border-red-500/25">
              AI: {activeAiBoxes.length} item{activeAiBoxes.length !== 1 ? 's' : ''}
            </span>
            <span className="flex-1 text-center text-xs font-semibold px-3 py-2 rounded-xl
                             bg-blue-500/15 text-blue-400 border border-blue-500/25">
              You: {userBoxes.length} item{userBoxes.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Box list */}
          <div className="glass rounded-xl overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-800">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                All Boxes
              </p>
            </div>
            <div className="max-h-72 overflow-y-auto">
              {activeAiBoxes.length === 0 && userBoxes.length === 0 && (
                <p className="text-slate-600 text-xs text-center py-6">
                  No boxes yet
                </p>
              )}
              {activeAiBoxes.map((b, i) => (
                <div
                  key={`ai-${i}`}
                  onClick={() => setSelectedBox({ type: 'ai', index: i })}
                  className={`flex items-center gap-2 px-3 py-2 text-xs cursor-pointer transition-colors
                              border-b border-slate-800/60 hover:bg-slate-800/40
                              ${selectedBox?.type === 'ai' && selectedBox?.index === i
                                ? 'bg-slate-800/60' : ''}`}
                >
                  <div className={`w-2.5 h-2.5 rounded-sm flex-shrink-0 ${b.source === 'custom' ? 'bg-red-900' : 'bg-red-500/80'}`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-300 truncate">{b.label || 'ai'}</p>
                    {b.value && (
                      <p className="text-slate-600 font-mono">{maskValue(b.value)}</p>
                    )}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeBox({ type: 'ai', index: i }); }}
                    className="text-slate-600 hover:text-red-400 transition-colors p-0.5 flex-shrink-0"
                  >
                    <XIcon />
                  </button>
                </div>
              ))}
              {userBoxes.map((b, i) => (
                <div
                  key={`user-${i}`}
                  onClick={() => setSelectedBox({ type: 'user', index: i })}
                  className={`flex items-center gap-2 px-3 py-2 text-xs cursor-pointer transition-colors
                              border-b border-slate-800/60 hover:bg-slate-800/40
                              ${selectedBox?.type === 'user' && selectedBox?.index === i
                                ? 'bg-slate-800/60' : ''}`}
                >
                  <div className="w-2.5 h-2.5 rounded-sm flex-shrink-0 bg-slate-700" />
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-300">Manual #{i + 1}</p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeBox({ type: 'user', index: i }); }}
                    className="text-slate-600 hover:text-red-400 transition-colors p-0.5 flex-shrink-0"
                  >
                    <XIcon />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Confirm button */}
          <button
            onClick={handleFinalise}
            disabled={isFinalising || (activeAiBoxes.length === 0 && userBoxes.length === 0)}
            className="w-full py-3 rounded-full font-semibold text-white text-sm
                       flex items-center justify-center gap-2 transition-all
                       bg-gradient-to-r from-indigo-600 to-cyan-600
                       hover:from-indigo-500 hover:to-cyan-500
                       shadow-lg shadow-indigo-500/20
                       disabled:opacity-40 disabled:cursor-not-allowed
                       disabled:shadow-none disabled:transform-none"
          >
            {isFinalising ? (
              <><SpinnerIcon /> Finalising…</>
            ) : (
              <><CheckIcon /> Confirm &amp; Finalise</>
            )}
          </button>

          <p className="text-slate-600 text-xs text-center">
            {activeAiBoxes.length + userBoxes.length} box
            {activeAiBoxes.length + userBoxes.length !== 1 ? 'es' : ''} will be applied
          </p>
        </div>
      </div>
    </div>
  );
}

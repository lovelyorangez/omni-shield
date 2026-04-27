import { useState, useRef, useEffect, useCallback } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// PDF.js renders at this scale; we divide by it when sending coords to backend.
const PDF_SCALE = 1.5;

/* ── Icons ─────────────────────────────────────────────────────────────── */
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
const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
  </svg>
);
const ChevronIcon = ({ dir }) => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d={dir === 'left' ? 'M15.75 19.5L8.25 12l7.5-7.5' : 'M8.25 4.5l7.5 7.5-7.5 7.5'} />
  </svg>
);
const PencilIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487z" />
  </svg>
);

/* ── Helpers ────────────────────────────────────────────────────────────── */
const maskValue = (val) => {
  const s = String(val ?? '');
  return s.length <= 2 ? `${s}***` : `${s.slice(0, 2)}***`;
};

/* ── Main Component ─────────────────────────────────────────────────────── */
export default function PDFEditor({ filename, fileHash, redactedFile, aiRedactions, ownerAddress, onFinalise, onSkip }) {
  const canvasRef   = useRef(null);
  const pageImgRef  = useRef(null);   // native Image rendered from current PDF page

  const [pdfDoc,      setPdfDoc]      = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages,    setNumPages]    = useState(0);
  const [imgLoaded,   setImgLoaded]   = useState(false);
  const [canvasSize,  setCanvasSize]  = useState({ w: 600, h: 800 });

  // pageBoxes: { pageNum: [{xmin,ymin,xmax,ymax,label}] }
  const [pageBoxes,   setPageBoxes]   = useState({});
  const [history,     setHistory]     = useState([]); // stack of full pageBoxes snapshots
  const [selectedBox, setSelectedBox] = useState(null); // { page, index } | null
  const [drawMode,    setDrawMode]    = useState(false);

  const [loading,       setLoading]       = useState(true);
  const [rendering,     setRendering]     = useState(false);
  const [error,         setError]         = useState(null);
  const [isFinalising,  setIsFinalising]  = useState(false);

  // Drawing state
  const drawingRef   = useRef(false);
  const drawStartRef = useRef(null);
  const [drawPreview, setDrawPreview] = useState(null);

  /* ── Load PDF via PDF.js ──────────────────────────────────────────────── */
  useEffect(() => {
    const lib = window.pdfjsLib;
    if (!lib) {
      setError('PDF.js not loaded — check that the CDN scripts in index.html are reachable.');
      setLoading(false);
      return;
    }
    const url = `${API_BASE_URL}/download/${filename}`;
    lib.getDocument(url).promise
      .then((pdf) => {
        setPdfDoc(pdf);
        setNumPages(pdf.numPages);
      })
      .catch((err) => setError(`Could not load PDF: ${err.message}`))
      .finally(() => setLoading(false));
  }, [filename]);

  /* ── Render current page → offscreen canvas → Image object ──────────── */
  useEffect(() => {
    if (!pdfDoc) return;
    setImgLoaded(false);
    setRendering(true);

    pdfDoc.getPage(currentPage)
      .then((page) => {
        const viewport  = page.getViewport({ scale: PDF_SCALE });
        const offscreen = document.createElement('canvas');
        offscreen.width  = viewport.width;
        offscreen.height = viewport.height;

        return page.render({
          canvasContext: offscreen.getContext('2d'),
          viewport,
        }).promise.then(() => {
          const dataUrl = offscreen.toDataURL('image/png');
          const img = new Image();
          img.onload = () => {
            pageImgRef.current = img;
            setCanvasSize({ w: viewport.width, h: viewport.height });
            setRendering(false);
            setImgLoaded(true);
          };
          img.src = dataUrl;
        });
      })
      .catch((err) => {
        console.error('[PDFEditor] page render error:', err);
        setRendering(false);
      });
  }, [pdfDoc, currentPage]);

  /* ── Draw: page image + AI boxes + user boxes + preview ─────────────── */
  const redraw = useCallback(() => {
    if (!imgLoaded || !canvasRef.current || !pageImgRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const { w, h } = canvasSize;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(pageImgRef.current, 0, 0, w, h);

    // Layer 1: AI-detected boxes for this page (fixed, scaled from PDF user-space)
    (aiRedactions || [])
      .filter((r) => r.page === currentPage && r.x0 != null)
      .forEach((r) => {
        const x  = r.x0 * PDF_SCALE;
        const y  = r.y0 * PDF_SCALE;
        const bw = (r.x1 - r.x0) * PDF_SCALE;
        const bh = (r.y1 - r.y0) * PDF_SCALE;
        if (r.source === 'custom') {
          ctx.fillStyle   = 'rgba(123,28,28,0.85)';
          ctx.strokeStyle = '#DC2626';
          ctx.lineWidth   = 1;
          ctx.fillRect(x, y, bw, bh);
          ctx.setLineDash([]);
          ctx.strokeRect(x, y, bw, bh);
        } else {
          ctx.fillStyle = 'rgba(0,0,0,0.9)';
          ctx.fillRect(x, y, bw, bh);
        }
      });

    // Layer 2: user-drawn boxes for this page
    ctx.setLineDash([]);
    (pageBoxes[currentPage] || []).forEach((b, i) => {
      const bw = b.xmax - b.xmin;
      const bh = b.ymax - b.ymin;
      ctx.fillStyle = 'rgba(0,0,0,0.9)';
      ctx.fillRect(b.xmin, b.ymin, bw, bh);

      const isSelected = selectedBox?.page === currentPage && selectedBox?.index === i;
      ctx.strokeStyle = isSelected ? '#60a5fa' : '#3b82f6';
      ctx.lineWidth   = isSelected ? 2.5 : 2;
      ctx.strokeRect(
        isSelected ? b.xmin - 1 : b.xmin,
        isSelected ? b.ymin - 1 : b.ymin,
        isSelected ? bw + 2 : bw,
        isSelected ? bh + 2 : bh,
      );
    });

    // Layer 3: live draw preview
    if (drawPreview && drawPreview.w > 0 && drawPreview.h > 0) {
      ctx.fillStyle   = 'rgba(0,0,0,0.28)';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.fillRect(drawPreview.x, drawPreview.y, drawPreview.w, drawPreview.h);
      ctx.strokeRect(drawPreview.x, drawPreview.y, drawPreview.w, drawPreview.h);
      ctx.setLineDash([]);
    }
  }, [imgLoaded, canvasSize, aiRedactions, pageBoxes, currentPage, drawPreview, selectedBox]);

  useEffect(() => { redraw(); }, [redraw]);

  /* ── Canvas coordinate helpers ───────────────────────────────────────── */
  const getCanvasPos = (e) => {
    const rect   = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width  / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
  };

  // Hit-test against user-drawn boxes on the current page (coords are canvas pixels)
  const hitTest = (pos) => {
    const boxes = pageBoxes[currentPage] || [];
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (pos.x >= b.xmin && pos.x <= b.xmax &&
          pos.y >= b.ymin && pos.y <= b.ymax) {
        return { page: currentPage, index: i };
      }
    }
    return null;
  };

  /* ── Mouse handlers ──────────────────────────────────────────────────── */
  const handleMouseDown = (e) => {
    if (!imgLoaded || rendering) return;
    const pos = getCanvasPos(e);

    if (drawMode) {
      // Draw mode: always start a potential draw, deselect any selection
      setSelectedBox(null);
      drawingRef.current   = true;
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
    const pos = getCanvasPos(e);
    setDrawPreview({
      x: Math.min(drawStartRef.current.x, pos.x),
      y: Math.min(drawStartRef.current.y, pos.y),
      w: Math.abs(pos.x - drawStartRef.current.x),
      h: Math.abs(pos.y - drawStartRef.current.y),
    });
  };

  const handleMouseUp = (e) => {
    if (!drawingRef.current || !drawStartRef.current) return;
    const pos = getCanvasPos(e);
    const dw  = Math.abs(pos.x - drawStartRef.current.x);
    const dh  = Math.abs(pos.y - drawStartRef.current.y);

    if (dw >= 8 && dh >= 8) {
      const xmin = Math.min(drawStartRef.current.x, pos.x);
      const ymin = Math.min(drawStartRef.current.y, pos.y);
      const xmax = Math.max(drawStartRef.current.x, pos.x);
      const ymax = Math.max(drawStartRef.current.y, pos.y);
      // Save snapshot before mutating (cap at 20 entries)
      setHistory(prev => [...prev.slice(-19), pageBoxes]);
      setPageBoxes((prev) => ({
        ...prev,
        [currentPage]: [...(prev[currentPage] || []), { xmin, ymin, xmax, ymax, label: 'manual' }],
      }));
    }

    drawingRef.current   = false;
    drawStartRef.current = null;
    setDrawPreview(null);
  };

  const removeBox = (pageNum, idx) => {
    setPageBoxes((prev) => ({
      ...prev,
      [pageNum]: (prev[pageNum] || []).filter((_, i) => i !== idx),
    }));
    if (selectedBox?.page === pageNum && selectedBox?.index === idx) {
      setSelectedBox(null);
    }
  };

  /* ── Keyboard shortcuts ────────────────────────────────────────────────── */
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      // Ctrl+Z — undo last user-drawn box (restores full pageBoxes snapshot)
      if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        if (history.length === 0) return;
        const prev = history[history.length - 1];
        setHistory(h => h.slice(0, -1));
        setPageBoxes(prev);
        setSelectedBox(null);
        return;
      }

      // Del / Backspace — delete the selected box
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (!selectedBox) return;
        e.preventDefault();
        setHistory(prev => [...prev.slice(-19), pageBoxes]);
        setPageBoxes(prev => ({
          ...prev,
          [selectedBox.page]: (prev[selectedBox.page] || [])
            .filter((_, i) => i !== selectedBox.index),
        }));
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
  }, [history, pageBoxes, selectedBox, drawMode]);

  /* ── Derived counts ──────────────────────────────────────────────────── */
  const currentPageBoxCount = (pageBoxes[currentPage] || []).length;
  const totalBoxCount = Object.values(pageBoxes).reduce((s, bs) => s + bs.length, 0);

  /* ── Finalise ────────────────────────────────────────────────────────── */
  const handleFinalise = async () => {
    setIsFinalising(true);
    try {
      // Convert canvas coords (1.5x scale) back to PDF user-space points
      const pages = {};
      Object.entries(pageBoxes).forEach(([pageNum, boxes]) => {
        if (boxes.length > 0) {
          pages[pageNum] = boxes.map((b) => ({
            xmin: b.xmin / PDF_SCALE,
            ymin: b.ymin / PDF_SCALE,
            xmax: b.xmax / PDF_SCALE,
            ymax: b.ymax / PDF_SCALE,
          }));
        }
      });

      const res = await fetch(`${API_BASE_URL}/redact/finalise/pdf`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ file_hash: fileHash, redacted_file: redactedFile, pages, owner_address: ownerAddress || '' }),
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

  /* ── Render ──────────────────────────────────────────────────────────── */
  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400 gap-3">
        <SpinnerIcon /> Loading PDF…
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-6 border border-red-500/25">
        <p className="text-red-400 text-sm font-medium">{error}</p>
        <button onClick={onSkip}
          className="mt-4 px-5 py-2 rounded-full text-sm font-medium border
                     border-slate-600/70 text-slate-300 hover:text-white transition-all">
          Skip Editor
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-white">PDF Redaction Editor</h2>
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
          <button onClick={onSkip}
            className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium
                       border border-slate-600/70 text-slate-400 hover:text-white
                       hover:border-indigo-500/60 transition-all">
            Skip Editor
          </button>
        </div>
      </div>

      <div className="flex gap-5 items-start" style={{ flexWrap: 'wrap' }}>

        {/* ── Canvas area ─────────────────────────────────────────── */}
        <div className="flex-1 min-w-0 space-y-3">
          {/* Page navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium
                         border border-slate-600/70 text-slate-300 transition-all
                         hover:border-indigo-500/60 hover:text-white
                         disabled:opacity-30 disabled:cursor-not-allowed">
              <ChevronIcon dir="left" /> Previous
            </button>
            <span className="text-slate-400 text-sm font-medium">
              Page {currentPage} of {numPages}
              {currentPageBoxCount > 0 && (
                <span className="ml-2 text-xs text-indigo-400">
                  ({currentPageBoxCount} box{currentPageBoxCount !== 1 ? 'es' : ''})
                </span>
              )}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
              disabled={currentPage === numPages}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium
                         border border-slate-600/70 text-slate-300 transition-all
                         hover:border-indigo-500/60 hover:text-white
                         disabled:opacity-30 disabled:cursor-not-allowed">
              Next <ChevronIcon dir="right" />
            </button>
          </div>

          {/* Canvas */}
          <div className="glass rounded-2xl overflow-hidden relative" style={{ lineHeight: 0 }}>
            {(!imgLoaded || rendering) && (
              <div className="absolute inset-0 flex items-center justify-center
                              bg-slate-900/60 z-10 text-slate-400 gap-2 text-sm">
                <SpinnerIcon /> Rendering page…
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={canvasSize.w}
              height={canvasSize.h}
              style={{
                display: 'block',
                width: '100%',
                cursor: drawMode ? 'crosshair' : 'default',
                userSelect: 'none',
              }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-black/90 border border-slate-700" />
              AI detected
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-[#7B1C1C]/85 border border-red-600" />
              Custom terms
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-black/90 border border-blue-500" />
              Your additions
            </span>
          </div>

          {/* Keyboard shortcut hint bar */}
          <div style={{
            display: 'flex',
            gap: '16px',
            marginTop: '2px',
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

        {/* ── Side panel ──────────────────────────────────────────── */}
        <div className="w-64 flex-shrink-0 space-y-4">

          {/* Total count */}
          <div className="flex gap-2">
            <span className="flex-1 text-center text-xs font-semibold px-3 py-2 rounded-xl
                             bg-slate-800/60 text-slate-300 border border-slate-700/60">
              Total: {totalBoxCount} box{totalBoxCount !== 1 ? 'es' : ''}
            </span>
            <span className="flex-1 text-center text-xs font-semibold px-3 py-2 rounded-xl
                             bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
              Page {currentPage}: {currentPageBoxCount}
            </span>
          </div>

          {/* Boxes on this page */}
          <div className="glass rounded-xl overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-800">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                Page {currentPage} Boxes
              </p>
            </div>
            <div className="max-h-44 overflow-y-auto">
              {currentPageBoxCount === 0 ? (
                <p className="text-slate-600 text-xs text-center py-5">
                  No boxes on this page yet
                </p>
              ) : (
                (pageBoxes[currentPage] || []).map((b, i) => (
                  <div key={i}
                    onClick={() => setSelectedBox({ page: currentPage, index: i })}
                    className={`flex items-center gap-2 px-3 py-2 text-xs cursor-pointer
                                border-b border-slate-800/60 hover:bg-slate-800/40 transition-colors
                                ${selectedBox?.page === currentPage && selectedBox?.index === i
                                  ? 'bg-slate-800/60' : ''}`}>
                    <div className={`w-2.5 h-2.5 rounded-sm flex-shrink-0
                                     ${selectedBox?.page === currentPage && selectedBox?.index === i
                                       ? 'bg-blue-500/80' : 'bg-slate-700'}`} />
                    <span className="flex-1 text-slate-300">Box #{i + 1}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeBox(currentPage, i); }}
                      className="text-slate-600 hover:text-red-400 transition-colors p-0.5">
                      <XIcon />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* AI suggestions (reference only) */}
          {aiRedactions?.length > 0 && (
            <div className="glass rounded-xl overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-800">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                  AI Detected PII
                </p>
              </div>
              <div className="max-h-44 overflow-y-auto">
                {aiRedactions.map((r, i) => (
                  <div key={i}
                    className="flex items-center justify-between px-3 py-2 text-xs
                               border-b border-slate-800/60">
                    <span className="text-slate-400 truncate">{String(r.label ?? r)}</span>
                    <span className="text-slate-600 font-mono ml-2 flex-shrink-0">
                      {maskValue(r.value ?? r)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Confirm button */}
          <button
            onClick={handleFinalise}
            disabled={isFinalising || totalBoxCount === 0}
            className="w-full py-3 rounded-full font-semibold text-white text-sm
                       flex items-center justify-center gap-2 transition-all
                       bg-gradient-to-r from-indigo-600 to-cyan-600
                       hover:from-indigo-500 hover:to-cyan-500
                       shadow-lg shadow-indigo-500/20
                       disabled:opacity-40 disabled:cursor-not-allowed
                       disabled:shadow-none disabled:transform-none">
            {isFinalising ? <><SpinnerIcon /> Finalising…</> : <><CheckIcon /> Confirm &amp; Finalise</>}
          </button>

          <p className="text-slate-600 text-xs text-center">
            {totalBoxCount} box{totalBoxCount !== 1 ? 'es' : ''} across {
              Object.values(pageBoxes).filter((bs) => bs.length > 0).length
            } page{Object.values(pageBoxes).filter((bs) => bs.length > 0).length !== 1 ? 's' : ''}
          </p>
        </div>
      </div>
    </div>
  );
}

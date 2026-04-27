import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWallet } from '../context/WalletContext.jsx';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import RedactionEditor from '../components/RedactionEditor.jsx';
import PDFEditor from '../components/PDFEditor.jsx';
import RedactionControls from '../components/RedactionControls.jsx';
import PipelineTimingChart from '../components/PipelineTimingChart.jsx';
import TextRedactionEditor from '../components/TextRedactionEditor.jsx';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const TABS = ['Image', 'PDF', 'Audio', 'Text', 'Word', 'Slides'];
const ACCEPTS = {
  Image:  'image/jpeg,image/png,image/webp',
  PDF:    'application/pdf',
  Audio:  'audio/mpeg,audio/wav,audio/mp4,audio/x-m4a',
  Text:   'text/plain,.txt',
  Word:   '.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  Slides: '.pptx,application/vnd.openxmlformats-officedocument.presentationml.presentation',
};
const TAB_ENDPOINT = {
  Image:  'image',
  PDF:    'pdf',
  Audio:  'audio',
  Text:   'text',
  Word:   'docx',
  Slides: 'pptx',
};
const STATUS_STEPS = [
  'Routing document...',
  'Running AI agents...',
  'Refining bounding boxes...',
  'Anchoring to blockchain...',
];

/* ── Inline SVG icons ──────────────────────────────────────────────────── */
const UploadCloudIcon = () => (
  <svg className="w-7 h-7 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775
         5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0
         0118 19.5H6.75z" />
  </svg>
);
const FileIcon = () => (
  <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125
         0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0
         12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125
         1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0
         1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
  </svg>
);
const CheckIcon = () => (
  <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
  </svg>
);
const DownloadIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
  </svg>
);
const DocumentIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424
         48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75
         0-.231-.035-.454-.1-.664M6.75 7.5H4.875c-.621 0-1.125.504-1.125 1.125v12c0 .621.504 1.125 1.125
         1.125h9.75c.621 0 1.125-.504 1.125-1.125V16.5a9 9 0 00-9-9z" />
  </svg>
);
const VerifyLinkIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0
         01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5
         0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
  </svg>
);
const EyeIcon = () => (
  <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638
         0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64
         19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);
const ErrorIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73
         0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898
         0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
  </svg>
);
const SpinnerIcon = () => (
  <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);
const SpinnerSmIcon = () => (
  <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);

/* ── Helpers ────────────────────────────────────────────────────────────── */
const formatBytes = (b) => {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
};

const maskValue = (val) => {
  const s = String(val ?? '');
  return s.length <= 2 ? `${s}***` : `${s.slice(0, 2)}***`;
};

/* ── Component ──────────────────────────────────────────────────────────── */
export default function AppPage() {
  const { walletAddress, connectWallet } = useWallet();
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const progressTimerRef = useRef(null);

  const [activeTab,      setActiveTab]      = useState('Image');
  const [file,           setFile]           = useState(null);
  const [originalUrl,    setOriginalUrl]    = useState(null);
  const [ownerAddress,   setOwnerAddress]   = useState('');
  const [isDragging,     setIsDragging]     = useState(false);
  const [isProcessing,   setIsProcessing]   = useState(false);
  const [statusStep,     setStatusStep]     = useState(0);
  const [progress,       setProgress]       = useState(0);
  const [result,         setResult]         = useState(null);
  const [viewMode,       setViewMode]       = useState('side-by-side');
  // Editor phase: null | 'editing' | 'done'
  const [editorPhase,    setEditorPhase]    = useState(null);
  const [finalResult,    setFinalResult]    = useState(null);
  const [showWalletModal, setShowWalletModal] = useState(false);

  const [hasProcessed,      setHasProcessed]      = useState(false);
  const [dryRun,            setDryRun]            = useState(false);

  const [redactionControls, setRedactionControls] = useState({
    pii_filter: {
      names: true, faces: true, signatures: true, phones: true, emails: true,
      dob: true, id_numbers: true, addresses: true, org_names: true, dates: true,
    },
    custom_terms: [],
    custom_patterns: [],
  });

  // Auto-fill wallet address
  useEffect(() => {
    if (walletAddress) setOwnerAddress(walletAddress);
  }, [walletAddress]);

  // Cleanup object URLs on unmount
  useEffect(() => () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
  }, []);

  const handleFileSelect = useCallback((f) => {
    if (!f) return;
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    setFile(f);
    setOriginalUrl(URL.createObjectURL(f));
    setResult(null);
    setEditorPhase(null);
    setFinalResult(null);
    setHasProcessed(false);
  }, [originalUrl]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileSelect(dropped);
  }, [handleFileSelect]);

  const switchTab = (tab) => {
    setActiveTab(tab);
    setFile(null);
    setResult(null);
    setEditorPhase(null);
    setFinalResult(null);
    if (originalUrl) { URL.revokeObjectURL(originalUrl); setOriginalUrl(null); }
  };

  /* Simulate pipeline progress over ~36 s */
  const startProgress = () => {
    setStatusStep(0);
    setProgress(0);
    let prog = 0;
    progressTimerRef.current = setInterval(() => {
      prog = Math.min(prog + 100 / 36, 94);
      setProgress(prog);
      setStatusStep(Math.min(Math.floor(prog / 25), STATUS_STEPS.length - 1));
    }, 1000);
  };

  const stopProgress = () => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setProgress(100);
    setStatusStep(STATUS_STEPS.length - 1);
  };

  const handleRedact = async () => {
    if (!file) return;
    if (!walletAddress) { setShowWalletModal(true); return; }
    const type = TAB_ENDPOINT[activeTab] || activeTab.toLowerCase();
    const formData = new FormData();
    formData.append('file', file);
    formData.append('owner_address', walletAddress || ownerAddress || 'anonymous');
    formData.append('pii_filter',       JSON.stringify(redactionControls.pii_filter));
    formData.append('custom_terms',     JSON.stringify(redactionControls.custom_terms));
    formData.append('custom_patterns',  JSON.stringify(redactionControls.custom_patterns));

    setIsProcessing(true);
    setResult(null);
    startProgress();

    try {
      const url = `${API_BASE_URL}/redact/${type}${dryRun ? '?dry_run=true' : ''}`;
      const res = await fetch(url, { method: 'POST', body: formData });
      if (!res.ok) {
        stopProgress();
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      let data = await res.json();

      // ── Async queue path: poll until done ──────────────────────────────────
      if (data.job_id && !data.file_hash && !data.dry_run) {
        const jobId = data.job_id;
        setStatusStep(1); // "Running AI agents..."
        const MAX_ATTEMPTS = 100; // 100 × 3 s = 300 s timeout (covers model reload after non-image jobs)
        let done = false;
        for (let i = 0; i < MAX_ATTEMPTS; i++) {
          await new Promise(r => setTimeout(r, 3000));
          // Advance the status indicator every ~15 s
          if (i === 5)  setStatusStep(2); // "Refining bounding boxes..."
          if (i === 10) setStatusStep(3); // "Anchoring to blockchain..."
          const pollRes = await fetch(`${API_BASE_URL}/api/job/${jobId}`, {
            headers: { Accept: 'application/json' },
          });
          if (!pollRes.ok) continue;
          const pollData = await pollRes.json();
          if (pollData.status === 'done') {
            let r = pollData.result ?? pollData;
            // Normalize queued job result → same shape as sync response so
            // the editor check (data.file_hash) and aiBoxes (data.redactions)
            // both work correctly.
            if (r.output_path && !r.file_hash) {
              r = {
                ...r,
                // file_hash must be truthy for the editor gate below
                file_hash:  r.output_path,
                // redactions live inside audit{} in the worker result;
                // promote them to the top level that RedactionEditor expects
                redactions: r.audit?.redactions ?? [],
              };
            }
            // Use server-side original (persisted after cleanup of /tmp upload).
            // Relative URL so nginx/Cloudflare routes /api/ to the backend
            // regardless of what API_BASE_URL resolves to in the browser.
            if (r.original_file) {
              setOriginalUrl(`/api/output/${r.original_file}`);
            }
            data = r;
            done = true;
            break;
          }
          if (pollData.status === 'failed') {
            throw new Error('Image processing failed: ' + (pollData.error ?? '').slice(0, 120));
          }
        }
        if (!done) throw new Error('Timeout — image job took longer than 240 s');
      }

      stopProgress();
      setHasProcessed(true);
      // Signal dashboard to refresh immediately after any non-dry-run redaction
      if (!data.dry_run) {
        localStorage.setItem('omni_last_redaction', Date.now().toString());
      }
      if (data.dry_run) {
        // Dry run: show would_redact list, skip editor
        setResult({ ...data, type });
        setEditorPhase('dry-run');
      } else if (type === 'image' && data.file_hash) {
        // Image path: go into editor review before finalising
        console.log('agent_timings:', data.agent_timings);
        setResult({ ...data, type });
        setEditorPhase('editing');
      } else if (type === 'pdf') {
        // PDF path: open manual box editor first
        setResult({ ...data, type });
        setEditorPhase('pdf-editing');
      } else if (type === 'docx' || type === 'pptx') {
        // Word/Slides: no editor, straight to download result
        setResult({
          ...data,
          type,
          sha256_hex: data.file_hash,
          original_filename: file.name,
          timestamp: Math.floor(Date.now() / 1000),
        });
        setEditorPhase(null);
      } else {
        setResult({ ...data, type });
        setEditorPhase(null);
      }
    } catch (err) {
      stopProgress();
      setResult({ error: err.message });
      setEditorPhase(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadFile = async (filename) => {
    const res = await fetch(`${API_BASE_URL}/download/${filename}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  const generateAuditPDF = async (overrideData) => {
    const record = overrideData || result;
    if (!record) return;

    // ── Normalise: try fetching full audit JSON from backend, fall back to record ──
    const rawHash = record.sha256_hex || record.file_hash || '';
    let auditData = record;
    if (rawHash && rawHash !== 'N/A') {
      try {
        const res = await fetch(`${API_BASE_URL}/api/audit/${rawHash}`);
        if (res.ok) {
          const fetched = await res.json();
          console.log('auditData from backend:', fetched);
          auditData = { ...record, ...fetched };
        }
      } catch (_) {}
    }

    // ── Canonical fields ──────────────────────────────────────────────────────
    const fileHash   = auditData.sha256_hex || auditData.file_hash || '';
    const rawDocType = auditData.doc_type || record.doc_type || 'unknown';
    const docTypeLabel = rawDocType === 'unknown'
      ? 'Not detected'
      : rawDocType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    const coverageSuffix = {
      id_card:          '% of image area redacted',
      image:            '% of image area redacted',
      pdf:              '% of document text redacted',
      audio_recording:  '% of audio duration redacted',
      text:             '% of text characters redacted',
      docx:             '% of document text redacted',
      pptx:             '% of slide text redacted',
    }[rawDocType] || '% of content redacted';
    const coveragePct = auditData.coverage_pct ?? record.coverage_pct ?? null;
    const coverageStr = coveragePct != null
      ? `${coveragePct}% ${coverageSuffix}`
      : 'Not available';

    const blockchainStatus =
      auditData.verified      ? 'Anchored' :
      auditData.blockchain_status ||
      (fileHash               ? 'Anchored' : 'Pending');

    const ownerAddr = auditData.owner_address
      || auditData.owner
      || auditData.walletAddress
      || auditData.record_owner
      || auditData.ownerAddress
      || 'Not recorded';

    const rawTs = auditData.timestamp ?? record.timestamp;
    const blockTs = rawTs
      ? new Date(Number(rawTs) * 1000).toLocaleString()
      : auditData.block_timestamp || auditData.blockTimestamp || 'Not recorded';

    const origFilename = auditData.filename
      || auditData.original_filename
      || record.filename
      || record.original_filename
      || file?.name
      || 'N/A';

    // ── Unified items list (objects or plain strings) ─────────────────────────
    const maskVal = (v) => {
      if (!v) return '—';
      const s = String(v);
      if (s.length <= 2) return '*'.repeat(s.length);
      return s[0] + '*'.repeat(Math.max(s.length - 2, 3)) + s[s.length - 1];
    };

    const rawItems = auditData.redactions
      || auditData.redacted_items
      || record.redactions
      || record.redacted_items
      || [];
    const tableRows = rawItems.map((item, i) => {
      if (typeof item === 'string') {
        return [i + 1, 'Text', maskVal(item), 'AI', 'N/A'];
      }
      const box = item.pixel_box
        ? `[${item.pixel_box.map(n => Math.round(n)).join(', ')}]`
        : item.box && item.box !== null
          ? `[${[].concat(item.box).map(n => Math.round(n)).join(', ')}]`
          : 'N/A';
      return [
        i + 1,
        String(item.label ?? 'Text'),
        maskVal(item.value ?? item),
        item.source === 'custom' ? 'Custom' : 'AI',
        box,
      ];
    });

    // ── Build PDF ─────────────────────────────────────────────────────────────
    const doc = new jsPDF();
    const now = new Date().toLocaleString();
    const M   = 20;
    let y     = 20;

    // 1. Dark background
    doc.setFillColor(15, 15, 25);
    doc.rect(0, 0, 210, 297, 'F');

    // 2. Header
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(15);
    doc.setFont(undefined, 'bold');
    doc.text('OMNI-SHIELD REDACTION AUDIT REPORT', M, y);
    y += 11;
    doc.setFontSize(9);
    doc.setFont(undefined, 'normal');
    doc.setTextColor(150, 150, 180);
    doc.text('All processing performed locally on edge hardware', M, y);
    y += 11;
    doc.setDrawColor(99, 102, 241);
    doc.setLineWidth(0.5);
    doc.line(M, y, 190, y);
    y += 10;

    // 4. Metadata
    const meta = [
      ['Date & Time',       now],
      ['Original File',     origFilename],
      ['Document Type',     docTypeLabel],
      ['SHA-256 Hash',      fileHash || 'N/A'],
      ['Items Redacted',    String(auditData.redaction_count ?? record.redaction_count ?? 0)],
      ['Coverage',          coverageStr],
      ['Blockchain Status', blockchainStatus],
      ['Wallet Address',    ownerAddr],
      ['Network',           'Ganache (Local Testnet)'],
      ['Block Timestamp',   blockTs],
    ];
    doc.setFontSize(9);
    meta.forEach(([label, value]) => {
      doc.setFont(undefined, 'bold');
      doc.setTextColor(130, 140, 255);
      doc.text(`${label}:`, M, y);
      doc.setFont(undefined, 'normal');
      doc.setTextColor(200, 200, 220);
      const split = doc.splitTextToSize(String(value), 118);
      doc.text(split, M + 50, y);
      y += split.length > 1 ? 12 : 7;
    });

    // 5. Redacted fields table
    y += 4;
    doc.setFontSize(11);
    doc.setFont(undefined, 'bold');
    doc.setTextColor(160, 165, 255);
    doc.text('Redacted Fields', M, y);
    y += 4;

    if (tableRows.length > 0) {
      autoTable(doc, {
        startY: y,
        head: [['#', 'Label', 'Masked Value', 'Source', 'Bounding Box']],
        body: tableRows,
        headStyles:         { fillColor: [30, 30, 50], textColor: [160, 165, 255], fontSize: 8, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [22, 22, 35] },
        bodyStyles:         { fillColor: [18, 18, 28], textColor: [200, 200, 220], fontSize: 8 },
        styles:             { cellPadding: 3 },
        columnStyles:       { 0: { cellWidth: 10 }, 3: { cellWidth: 20 }, 4: { cellWidth: 40 } },
        margin:             { left: M, right: M },
      });
      y = doc.lastAutoTable.finalY + 10;
    } else {
      doc.setFontSize(9);
      doc.setFont(undefined, 'normal');
      doc.setTextColor(120, 120, 140);
      doc.text('No detailed redaction log available for this record.', M + 4, y + 4);
      y += 14;
    }

    // 6. PII categories — two side-by-side compact tables
    const piiFilter = auditData.pii_filter || auditData.redaction_config?.pii_filter;
    if (piiFilter && Object.keys(piiFilter).length > 0) {
      if (y > 230) { doc.addPage(); doc.setFillColor(15,15,25); doc.rect(0,0,210,297,'F'); y = 20; }
      doc.setFontSize(10);
      doc.setFont(undefined, 'bold');
      doc.setTextColor(160, 165, 255);
      doc.text('Redaction Categories', M, y);
      y += 4;
      const catY = y;
      const catStyles = {
        styles:             { fontSize: 7, cellPadding: 2 },
        headStyles:         { fillColor: [30, 30, 50], textColor: 255, fontSize: 7, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [22, 22, 35] },
        bodyStyles:         { fillColor: [18, 18, 28], textColor: [200, 200, 220] },
        columnStyles:       { 0: { cellWidth: 52 }, 1: { cellWidth: 28 } },
      };
      autoTable(doc, {
        startY: catY, tableWidth: 80, margin: { left: 14 },
        head: [['Category', 'Status']],
        body: [
          ['Names',           piiFilter.names      !== false ? 'Enabled' : 'Off'],
          ['Faces',           piiFilter.faces      !== false ? 'Enabled' : 'Off'],
          ['Signatures',      piiFilter.signatures !== false ? 'Enabled' : 'Off'],
          ['Phone Numbers',   piiFilter.phones     !== false ? 'Enabled' : 'Off'],
          ['Email Addresses', piiFilter.emails     !== false ? 'Enabled' : 'Off'],
        ],
        ...catStyles,
      });
      const leftBottom = doc.lastAutoTable.finalY;
      autoTable(doc, {
        startY: catY, tableWidth: 80, margin: { left: 108 },
        head: [['Category', 'Status']],
        body: [
          ['Dates of Birth',  piiFilter.dob        !== false ? 'Enabled' : 'Off'],
          ['ID Numbers',      piiFilter.id_numbers !== false ? 'Enabled' : 'Off'],
          ['Addresses',       piiFilter.addresses  !== false ? 'Enabled' : 'Off'],
          ['Org Names',       piiFilter.org_names  !== false ? 'Enabled' : 'Off'],
          ['Dates (General)', piiFilter.dates      !== false ? 'Enabled' : 'Off'],
        ],
        ...catStyles,
      });
      y = Math.max(leftBottom, doc.lastAutoTable.finalY) + 6;
    }

    // 7. Custom terms
    const customMatched = auditData.custom_terms_matched
      || auditData.redaction_config?.custom_terms_matched
      || [];
    const customPatterns = auditData.custom_patterns_matched
      || auditData.redaction_config?.custom_patterns_matched
      || [];
    if (customMatched.length > 0 || customPatterns.length > 0) {
      if (y > 260) { doc.addPage(); doc.setFillColor(15,15,25); doc.rect(0,0,210,297,'F'); y = 20; }
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      doc.setTextColor(160, 165, 255);
      doc.text('Custom Terms Matched', M, y);
      y += 7;
      doc.setFontSize(9);
      doc.setFont(undefined, 'normal');
      doc.setTextColor(200, 200, 220);
      customMatched.forEach((t) => { doc.text(`● "${t}" — found and redacted`, M + 4, y); y += 6; });
      customPatterns.forEach((p) => { doc.text(`● /${p}/ (regex) — found and redacted`, M + 4, y); y += 6; });
    }

    // 8. Page numbers
    const pageCount = doc.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(80, 80, 100);
      doc.text('Generated by Omni-Shield — Local-first AI Redaction', 105, 285, { align: 'center' });
      doc.text(`Page ${i} of ${pageCount}`, 105, 290, { align: 'center' });
    }

    const shortHash = fileHash ? fileHash.slice(0, 8) : Date.now();
    const dateStr   = new Date().toISOString().slice(0, 10);
    doc.save(`omni-shield-audit-${shortHash}-${dateStr}.pdf`);
  };

  /* ── Derived state ─────────────────────────────────────────────────────── */
  const redactedFilename = result?.output_path
    ? result.output_path.split('/').pop()
    : result?.redacted_file ?? null;

  /* ── Render ───────────────────────────────────────────────────────────── */
  return (
    <>
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6"
      style={{ background: '#0a0a0f' }}>
      <div className="max-w-6xl mx-auto">

        {/* Page header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-1">Redact Document</h1>
          <p className="text-slate-400 text-sm">
            Upload a document and let the 8-agent AI pipeline detect and remove PII.
          </p>
        </div>

        {/* ── DRY-RUN RESULT ───────────────────────────────────── */}
        {editorPhase === 'dry-run' && result?.dry_run && (
          <div className="glass rounded-2xl p-6 mb-8 space-y-4 border border-amber-500/20">
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold
                               bg-amber-500/15 text-amber-400 border border-amber-500/30">
                DRY RUN
              </span>
              <h2 className="text-lg font-bold text-white">Preview — no files written</h2>
            </div>
            <p className="text-slate-400 text-sm">{result.message}</p>
            <div className="flex items-center gap-4">
              <span className="text-3xl font-bold text-white">{result.redaction_count ?? 0}</span>
              <span className="text-slate-400 text-sm">items would be redacted</span>
              {result.doc_type && result.doc_type !== 'unknown' && (
                <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase
                                 tracking-wide bg-indigo-500/15 text-indigo-300">
                  {result.doc_type.replace(/_/g, ' ')}
                </span>
              )}
            </div>
            {result.would_redact?.length > 0 && (
              <div className="glass rounded-xl overflow-hidden">
                <div className="px-4 py-2 border-b border-slate-800">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                    Would Redact
                  </p>
                </div>
                <div className="max-h-64 overflow-y-auto divide-y divide-slate-800/60">
                  {result.would_redact.map((r, i) => (
                    <div key={i} className="flex items-center gap-3 px-4 py-2.5 text-xs">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium flex-shrink-0
                                        ${r.label === 'Custom' || r.label === 'custom'
                                          ? 'bg-amber-500/15 text-amber-400'
                                          : 'bg-red-500/15 text-red-400'}`}>
                        {r.label || 'ai'}
                      </span>
                      <span className="text-slate-300 font-mono truncate flex-1">
                        {r.value || '—'}
                      </span>
                      {r.box && (
                        <span className="text-slate-600 font-mono flex-shrink-0">
                          [{r.box.slice(0, 4).map(n => Math.round(n)).join(', ')}]
                        </span>
                      )}
                      {r.page && (
                        <span className="text-slate-600 flex-shrink-0">p{r.page}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {activeTab === 'Image' &&
              result?.agent_timings &&
              Object.keys(result.agent_timings).length > 0 && (
                <PipelineTimingChart timings={result.agent_timings} />
            )}
            <button
              onClick={() => { setResult(null); setEditorPhase(null); }}
              className="px-5 py-2 rounded-full text-sm font-medium border
                         border-slate-600/70 text-slate-300 hover:text-white transition-all">
              Clear
            </button>
          </div>
        )}

        {/* ── EDITOR PHASE (image review) ───────────────────────── */}
        {editorPhase === 'editing' && result && originalUrl && (
          <div className="glass rounded-2xl p-6 mb-8">
            <RedactionEditor
              originalUrl={originalUrl}
              aiBoxes={(result.redactions || []).map(r => ({
                ...r,
                // Bug 1: backend returns pixel_box, editor expects .box
                box: r.pixel_box || r.box,
              }))}
              fileHash={result.file_hash}
              originalFile={result.original_file}
              ownerAddress={walletAddress || ownerAddress}
              onFinalise={(fr) => {
                setFinalResult(fr);
                setEditorPhase('done');
              }}
              onReRun={() => {
                setResult(null);
                setEditorPhase(null);
              }}
            />
            {activeTab === 'Image' &&
              result?.agent_timings &&
              Object.keys(result.agent_timings).length > 0 && (
                <PipelineTimingChart timings={result.agent_timings} />
            )}
          </div>
        )}

        {/* ── FINALISED RESULT (image editor done) ─────────────── */}
        {editorPhase === 'done' && finalResult && (
          <div className="glass rounded-2xl p-6 mb-8 space-y-5">
            <div>
              <h2 className="text-lg font-bold text-white mb-1">Redaction Complete</h2>
              <p className="text-slate-400 text-sm">
                Your edited document has been finalised and encrypted.
              </p>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-4xl font-bold text-white">
                {finalResult.redaction_count ?? 0}
              </span>
              <span className="text-slate-400 text-sm">boxes applied</span>
              {result?.doc_type && result.doc_type !== 'unknown' && (
                <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase
                                 tracking-wide bg-indigo-500/15 text-indigo-300">
                  {result.doc_type.replace(/_/g, ' ')}
                </span>
              )}
            </div>
            {activeTab === 'Image' &&
              result?.agent_timings &&
              Object.keys(result.agent_timings).length > 0 && (
                <PipelineTimingChart timings={result.agent_timings} />
            )}
            {/* Bug 4: show all three action buttons in the done panel */}
            <div className="flex flex-wrap gap-3">
              {finalResult.finalised_file && (
                <button
                  onClick={() => downloadFile(finalResult.finalised_file)}
                  className="flex-1 btn-primary py-3 rounded-full text-white font-medium text-sm
                             flex items-center justify-center gap-2 min-w-[160px]"
                >
                  <DownloadIcon /> Download Redacted
                </button>
              )}
              <button
                onClick={() => generateAuditPDF({
                  ...result,
                  file_hash: finalResult.file_hash,
                  redaction_count: finalResult.redaction_count,
                })}
                className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                           border border-slate-600/70 hover:border-indigo-500/60
                           hover:text-indigo-300 transition-all flex items-center justify-center gap-2"
              >
                <DocumentIcon /> Audit Report
              </button>
              {finalResult.file_hash && (
                <button
                  onClick={() => navigate(`/verify?hash=${finalResult.file_hash}`)}
                  className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                             border border-slate-600/70 hover:border-cyan-500/50
                             hover:text-cyan-300 transition-all flex items-center justify-center gap-2"
                >
                  <VerifyLinkIcon /> Verify on Chain
                </button>
              )}
              <button
                onClick={() => {
                  setResult(null);
                  setFinalResult(null);
                  setEditorPhase(null);
                }}
                className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                           border border-slate-600/70 hover:border-indigo-500/60
                           hover:text-indigo-300 transition-all flex items-center justify-center gap-2"
              >
                Redact Another
              </button>
            </div>
          </div>
        )}

        {/* ── PDF EDITOR (manual box drawing) ───────────────────── */}
        {editorPhase === 'pdf-editing' && result && (
          <div className="glass rounded-2xl p-6 mb-8">
            <PDFEditor
              filename={result.redacted_file}
              fileHash={result.file_hash}
              redactedFile={result.redacted_file}
              aiRedactions={result.redactions || []}
              ownerAddress={walletAddress || ownerAddress}
              onFinalise={(fr) => {
                setFinalResult(fr);
                setEditorPhase('pdf-done');
              }}
              onSkip={() => setEditorPhase('pdf-done')}
            />
          </div>
        )}

        {/* ── PDF REVIEW STEP ───────────────────────────────────── */}
        {editorPhase === 'pdf-done' && result && (
          <div className="glass rounded-2xl p-6 mb-8 space-y-5">
            <div>
              <h2 className="text-lg font-bold text-white mb-1">PDF Redaction Complete</h2>
              <p className="text-slate-400 text-sm">
                {finalResult
                  ? 'Manual redactions applied. Download the finalised PDF below.'
                  : 'Review what was redacted, then download the protected PDF.'}
              </p>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-4xl font-bold text-white">
                {(finalResult?.redaction_count ?? result.redaction_count) ?? 0}
              </span>
              <span className="text-slate-400 text-sm">
                {finalResult ? 'manual boxes applied' : 'items redacted'}
              </span>
              {result.doc_type && (
                <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase
                                 tracking-wide bg-indigo-500/15 text-indigo-300">
                  {result.doc_type.replace(/_/g, ' ')}
                </span>
              )}
            </div>

            {/* Show finalised PDF if editor was used, otherwise the AI-redacted one */}
            {(finalResult?.finalised_file || result.redacted_file) && (
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
                  {finalResult ? 'Finalised PDF Preview' : 'Redacted PDF Preview'}
                </p>
                <iframe
                  src={`${API_BASE_URL}/download/${finalResult?.finalised_file ?? result.redacted_file}`}
                  title="Redacted PDF"
                  className="w-full rounded-xl border border-slate-700/60"
                  style={{ height: '480px' }}
                />
              </div>
            )}

            {result.redactions?.length > 0 && (
              <div className="border border-slate-800 rounded-xl overflow-hidden">
                {result.redactions.map((r, i) => (
                  <div key={i}
                    className={`flex items-center justify-between px-4 py-2.5 text-sm
                                ${i % 2 === 0 ? 'bg-slate-900/40' : 'bg-slate-800/20'}`}>
                    <span className="text-slate-400">{String(r.label ?? r)}</span>
                    <span className="text-slate-300 font-mono">{maskValue(r.value ?? r)}</span>
                  </div>
                ))}
              </div>
            )}
            {result.file_hash && (
              <div className="flex items-center gap-3 p-3 rounded-xl
                              bg-green-500/8 border border-green-500/20">
                <div className="w-6 h-6 rounded-full bg-green-500/15 flex items-center justify-center flex-shrink-0">
                  <CheckIcon />
                </div>
                <div>
                  <p className="text-green-400 text-sm font-medium">Anchored to blockchain</p>
                  <p className="text-slate-500 text-xs font-mono mt-0.5">
                    {result.file_hash.slice(0, 24)}…
                  </p>
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-3">
              {(finalResult?.finalised_file || result.redacted_file) && (
                <button
                  onClick={() => downloadFile(finalResult?.finalised_file ?? result.redacted_file)}
                  className="flex-1 btn-primary py-3 rounded-full text-white font-medium text-sm
                             flex items-center justify-center gap-2 min-w-[160px]"
                >
                  <DownloadIcon /> Download Redacted PDF
                </button>
              )}
              <button
                onClick={() => generateAuditPDF({
                  ...result,
                  redaction_count: finalResult?.redaction_count ?? result.redaction_count,
                })}
                className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                           border border-slate-600/70 hover:border-indigo-500/60
                           hover:text-indigo-300 transition-all flex items-center justify-center gap-2"
              >
                <DocumentIcon /> Audit Report
              </button>
              {result.file_hash && (
                <button
                  onClick={() => navigate(`/verify?hash=${result.file_hash}`)}
                  className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                             border border-slate-600/70 hover:border-cyan-500/50
                             hover:text-cyan-300 transition-all flex items-center justify-center gap-2"
                >
                  <VerifyLinkIcon /> Verify on Chain
                </button>
              )}
              <button
                onClick={() => {
                  setResult(null);
                  setFinalResult(null);
                  setEditorPhase(null);
                }}
                className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                           border border-slate-600/70 hover:border-indigo-500/60
                           hover:text-indigo-300 transition-all flex items-center justify-center gap-2"
              >
                Redact Another
              </button>
            </div>
          </div>
        )}

        {/* ── UPLOAD / RESULTS (hidden during editing / done) ──── */}
        {!editorPhase && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* ── LEFT COLUMN ─────────────────────────────────────────────── */}
          <div className="space-y-5">

            {/* Type tabs + drop zone */}
            <div className="glass rounded-2xl p-6">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
                Document Type
              </p>

              {/* Tabs */}
              <div className="flex flex-wrap gap-2 mb-6">
                {TABS.map((tab) => (
                  <button key={tab} onClick={() => switchTab(tab)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                      activeTab === tab
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
                        : 'bg-slate-800/80 text-slate-400 hover:text-white'
                    }`}>
                    {tab}
                  </button>
                ))}
              </div>

              {/* Drop zone */}
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center p-10 rounded-xl
                            cursor-pointer transition-all duration-200
                            border-2 border-dashed
                            ${isDragging
                              ? 'drop-zone-active'
                              : 'border-indigo-500/25 hover:border-indigo-500/50 hover:bg-indigo-500/[0.03]'
                            }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept={ACCEPTS[activeTab]}
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                />
                {file ? (
                  <div className="text-center w-full">
                    {activeTab === 'Image' && originalUrl ? (
                      <img
                        src={originalUrl}
                        alt="Preview"
                        className="max-h-40 max-w-full mx-auto rounded-lg object-contain mb-3"
                      />
                    ) : (
                      <div className="w-12 h-12 rounded-xl bg-indigo-500/15 border border-indigo-500/25
                                      flex items-center justify-center mx-auto mb-3">
                        <FileIcon />
                      </div>
                    )}
                    <p className="text-white font-medium text-sm">{file.name}</p>
                    <p className="text-slate-500 text-xs mt-1">{formatBytes(file.size)}</p>
                    <p className="text-indigo-400/70 text-xs mt-2">Click to replace</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="w-12 h-12 rounded-xl bg-slate-800 flex items-center
                                    justify-center mx-auto mb-3">
                      <UploadCloudIcon />
                    </div>
                    <p className="text-white font-medium">Drop your document here</p>
                    <p className="text-slate-500 text-sm mt-1">or click to browse</p>
                    <p className="text-slate-600 text-xs mt-2">
                      {activeTab === 'Image'  ? 'JPG · PNG · WEBP'
                        : activeTab === 'PDF'    ? 'PDF'
                        : activeTab === 'Text'   ? 'TXT'
                        : activeTab === 'Word'   ? 'DOCX'
                        : activeTab === 'Slides' ? 'PPTX'
                        : 'MP3 · WAV · M4A'}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Wallet address */}
            <div className="glass rounded-2xl p-6">
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
                Wallet address for audit record
              </label>
              <input
                type="text"
                value={ownerAddress}
                onChange={(e) => setOwnerAddress(e.target.value)}
                placeholder="0x… (auto-filled when wallet connected)"
                className="w-full bg-slate-900/60 border border-slate-700/60 rounded-xl px-4 py-3
                           text-white font-mono text-sm placeholder:text-slate-600 placeholder:font-sans
                           focus:outline-none focus:border-indigo-500/70 transition-colors"
              />
            </div>

            {/* Redaction controls — shown once a file is selected */}
            {file && (
              <RedactionControls onChange={setRedactionControls} />
            )}

            {/* Redact button */}
            <button
              onClick={handleRedact}
              disabled={!file || isProcessing || !walletAddress}
              title={!walletAddress ? 'Connect wallet to enable redaction' : undefined}
              className="w-full py-4 rounded-full font-semibold text-white text-base
                         flex items-center justify-center gap-3
                         btn-primary disabled:opacity-40 disabled:cursor-not-allowed
                         disabled:transform-none disabled:shadow-none"
            >
              {isProcessing ? (
                <><SpinnerIcon />{STATUS_STEPS[statusStep]}</>
              ) : 'Redact Document'}
            </button>

            {/* Dry-run toggle — dev only */}
            {import.meta.env.DEV && (
              <label className="flex items-center gap-2.5 cursor-pointer select-none w-fit">
                <input
                  type="checkbox"
                  checked={dryRun}
                  onChange={(e) => setDryRun(e.target.checked)}
                  className="w-3.5 h-3.5 rounded accent-amber-500"
                />
                <span className="text-xs text-amber-400/80 font-medium">
                  Dry run — preview detections, skip file write &amp; blockchain
                </span>
              </label>
            )}

            {/* Processing card */}
            {isProcessing && (
              <div className="glass rounded-2xl p-6">
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-slate-400">{STATUS_STEPS[statusStep]}</span>
                  <span className="text-indigo-400 font-mono">{Math.round(progress)}%</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-5">
                  <div className="h-full rounded-full transition-all duration-1000
                                  bg-gradient-to-r from-indigo-500 to-cyan-500"
                    style={{ width: `${progress}%` }} />
                </div>
                <div className="space-y-2.5">
                  {STATUS_STEPS.map((step, i) => (
                    <div key={i}
                      className={`flex items-center gap-3 text-sm transition-opacity duration-300 ${
                        i <= statusStep ? 'opacity-100' : 'opacity-25'
                      }`}>
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${
                        i < statusStep  ? 'bg-green-500/20'
                        : i === statusStep ? 'bg-indigo-500/20'
                        : 'bg-slate-800'
                      }`}>
                        {i < statusStep  && <CheckIcon />}
                        {i === statusStep && <SpinnerSmIcon />}
                      </div>
                      <span className={i <= statusStep ? 'text-slate-300' : 'text-slate-600'}>
                        {step}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── RIGHT COLUMN ──────────────────────────────────────────── */}
          <div className="space-y-5">

            {/* Empty state */}
            {!result && !isProcessing && (
              <div className="glass rounded-2xl p-10 flex flex-col items-center
                              justify-center text-center min-h-[300px]">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/60 flex items-center
                                justify-center mb-4">
                  <EyeIcon />
                </div>
                <p className="text-slate-400 font-medium">Results will appear here</p>
                <p className="text-slate-600 text-sm mt-1">
                  Upload a document and click Redact to begin
                </p>
              </div>
            )}

            {/* Error state */}
            {result?.error && (
              <div className="glass rounded-2xl p-6 border border-red-500/25">
                <div className="flex items-start gap-3 text-red-400">
                  <div className="mt-0.5 flex-shrink-0"><ErrorIcon /></div>
                  <div>
                    <p className="font-semibold">Processing failed</p>
                    <p className="text-sm text-red-400/80 mt-1">{result.error}</p>
                  </div>
                </div>
              </div>
            )}

            {result && !result.error && (
              <>
                {/* Before / After preview (image only) */}
                {result.type === 'image' && redactedFilename && (
                  <div className="glass rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                        Preview
                      </p>
                      <div className="flex gap-2">
                        {['side-by-side', 'overlay'].map((mode) => (
                          <button key={mode} onClick={() => setViewMode(mode)}
                            className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                              viewMode === mode
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-800 text-slate-400 hover:text-white'
                            }`}>
                            {mode === 'side-by-side' ? 'Side by Side' : 'Overlay'}
                          </button>
                        ))}
                      </div>
                    </div>

                    {viewMode === 'side-by-side' ? (
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <p className="text-xs text-slate-500 text-center mb-2">Original</p>
                          {originalUrl && (
                            <div className="relative">
                              <img src={originalUrl} alt="Original"
                                className="w-full rounded-lg opacity-50" />
                              <div className="absolute inset-0 flex items-center justify-center">
                                <span className="text-white/70 font-bold text-xs
                                                 bg-black/50 px-2 py-1 rounded tracking-widest">
                                  ORIGINAL
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                        <div>
                          <p className="text-xs text-slate-500 text-center mb-2">Redacted</p>
                          <img
                            src={`${API_BASE_URL}/download/${redactedFilename}`}
                            alt="Redacted"
                            className="w-full rounded-lg"
                          />
                        </div>
                      </div>
                    ) : (
                      <img
                        src={`${API_BASE_URL}/download/${redactedFilename}`}
                        alt="Redacted"
                        className="w-full rounded-lg"
                      />
                    )}
                  </div>
                )}

                {/* Text / Word / Slides — manual review editor */}
                {(result.type === 'text' || result.type === 'docx' || result.type === 'pptx') && (
                  <TextRedactionEditor
                    result={result}
                    docType={result.type}
                    onDownload={downloadFile}
                  />
                )}

                {/* Summary card */}
                <div className="glass rounded-2xl p-6">
                  <div className="flex flex-wrap items-center gap-3 mb-4">
                    {result.doc_type && (
                      <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase
                                       tracking-wide bg-indigo-500/15 text-indigo-300">
                        {result.doc_type.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <div style={{display:'flex', gap:24, marginBottom:12}}>
                    <div>
                      <div style={{fontSize:11, color:'var(--color-text-tertiary)',
                                   textTransform:'uppercase', letterSpacing:'0.06em',
                                   marginBottom:2}}>
                        Items Redacted
                      </div>
                      <div style={{fontSize:28, fontWeight:700, color:'#fff', lineHeight:1}}>
                        {result.redaction_count ?? 0}
                      </div>
                    </div>
                    {result.coverage_pct != null && (
                      <div>
                        <div style={{fontSize:11, color:'var(--color-text-tertiary)',
                                     textTransform:'uppercase', letterSpacing:'0.06em',
                                     marginBottom:2}}>
                          Coverage
                        </div>
                        <div style={{fontSize:28, fontWeight:700, color:'#fff', lineHeight:1}}>
                          {result.coverage_pct ?? 0}%
                        </div>
                      </div>
                    )}
                  </div>
                  {result.coverage_pct != null && (
                    <div style={{marginBottom:16}}>
                      <div style={{height:4, borderRadius:2,
                                   background:'var(--color-border-tertiary)'}}>
                        <div style={{
                          height:'100%', borderRadius:2,
                          width:`${Math.min(result.coverage_pct ?? 0, 100)}%`,
                          background:'#7F77DD',
                          transition:'width 1s ease'
                        }}/>
                      </div>
                      <div style={{fontSize:11, color:'var(--color-text-tertiary)', marginTop:4}}>
                        of document area redacted
                      </div>
                    </div>
                  )}
                  <div className="mb-5" />

                  {/* Redacted fields list */}
                  {(result.redactions?.length > 0 || result.redacted_items?.length > 0) && (
                    <div className="space-y-0 mb-5 border border-slate-800 rounded-xl overflow-hidden">
                      {(result.redactions || result.redacted_items || []).map((r, i) => {
                        const label  = typeof r === 'object' ? (r.label ?? 'PII') : 'PII';
                        const value  = typeof r === 'object' ? (r.value ?? r) : r;
                        return (
                          <div key={i}
                            className={`flex items-center justify-between px-4 py-2.5 text-sm
                                        ${i % 2 === 0 ? 'bg-slate-900/40' : 'bg-slate-800/20'}`}>
                            <span className="text-slate-400">{String(label)}</span>
                            <span className="text-slate-300 font-mono">{maskValue(value)}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Blockchain anchor badge */}
                  <div className="flex items-center gap-3 p-3 rounded-xl
                                  bg-green-500/8 border border-green-500/20">
                    <div className="w-6 h-6 rounded-full bg-green-500/15
                                    flex items-center justify-center flex-shrink-0">
                      <CheckIcon />
                    </div>
                    <div>
                      <p className="text-green-400 text-sm font-medium">Anchored to blockchain</p>
                      {result.file_hash && (
                        <p className="text-slate-500 text-xs font-mono mt-0.5">
                          {result.file_hash.slice(0, 24)}…
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex flex-wrap gap-3">
                  {redactedFilename && (
                    <button
                      onClick={() => downloadFile(redactedFilename)}
                      className="flex-1 btn-primary py-3 rounded-full text-white font-medium text-sm
                                 flex items-center justify-center gap-2 min-w-[140px]"
                    >
                      <DownloadIcon /> Download Redacted
                    </button>
                  )}
                  <button
                    onClick={generateAuditPDF}
                    className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                               border border-slate-600/70 hover:border-indigo-500/60
                               hover:text-indigo-300 transition-all flex items-center justify-center gap-2"
                  >
                    <DocumentIcon /> Audit Report
                  </button>
                  {result.file_hash && (
                    <button
                      onClick={() => navigate(`/verify?hash=${result.file_hash}`)}
                      className="flex-1 py-3 rounded-full text-white font-medium text-sm min-w-[140px]
                                 border border-slate-600/70 hover:border-cyan-500/50
                                 hover:text-cyan-300 transition-all flex items-center justify-center gap-2"
                    >
                      <VerifyLinkIcon /> Verify on Chain
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
        )}
      </div>
    </div>

    {/* ── Wallet required modal ─────────────────────────────────────── */}
    {showWalletModal && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
        onClick={() => setShowWalletModal(false)}>
        <div className="glass rounded-3xl p-8 max-w-sm w-full mx-4 text-center"
          onClick={(e) => e.stopPropagation()}>
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/15 border border-indigo-500/20
                          flex items-center justify-center mx-auto mb-5">
            <svg width="32" height="32" viewBox="0 0 24 24"
              fill="none" stroke="#6366f1" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <h2 className="text-xl font-bold text-white mb-3">Wallet Required</h2>
          <p className="text-slate-400 text-sm mb-8 leading-relaxed">
            Please connect your MetaMask wallet before redacting documents.
            Your wallet address is used to anchor the audit record to the blockchain.
          </p>
          <div className="flex flex-col gap-3">
            <button
              onClick={() => { connectWallet(); setShowWalletModal(false); }}
              className="w-full py-3 rounded-full font-semibold text-white text-sm btn-primary">
              Connect Wallet
            </button>
            <button
              onClick={() => setShowWalletModal(false)}
              className="w-full py-3 rounded-full text-sm font-medium
                         border border-slate-600/70 text-slate-400
                         hover:text-white hover:border-slate-500 transition-all">
              Cancel
            </button>
          </div>
        </div>
      </div>
    )}
    </>
  );
}

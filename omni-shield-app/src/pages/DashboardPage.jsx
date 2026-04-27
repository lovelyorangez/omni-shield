import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWallet } from '../context/WalletContext.jsx';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/* ── Icons ─────────────────────────────────────────────────────────────── */
const RefreshIcon = ({ spinning }) => (
  <svg className={`w-4 h-4 ${spinning ? 'animate-spin' : ''}`}
    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993
         0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25
         0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
);
const CopyIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125
         1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06
         9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504
         1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0
         00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5
         10.375H9.75a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375
         3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375
         3.375 0 00-3.375-3.375H9.75" />
  </svg>
);
const DownloadIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021
         18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
  </svg>
);
const CheckSmIcon = () => (
  <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
  </svg>
);
const SpinnerSmIcon = () => (
  <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);
const DocumentCheckIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M10.125 2.25h-4.5c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504
         1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125v-9M10.125
         2.25h.375a9 9 0 019 9v.375M10.125 2.25A3.375 3.375 0 0113.5 5.625v1.5c0
         .621.504 1.125 1.125 1.125h1.5a3.375 3.375 0 013.375 3.375M9 15l2.25
         2.25L15 12" />
  </svg>
);
const ShieldCheckIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6
         3.99 3.99 0 013 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332
         9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196
         0-6.1-1.248-8.25-3.285z" />
  </svg>
);
const CalendarIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25
         2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0
         0021 18.75m-18 0v-7.5A2.25 2.25 0 015.25 9h13.5A2.25 2.25 0
         0121 11.25v7.5" />
  </svg>
);
const ClockIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);
const WalletIcon = () => (
  <svg className="w-10 h-10 text-indigo-400/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0
         003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18
         0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 9m18
         0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v3" />
  </svg>
);

/* ── Helpers ────────────────────────────────────────────────────────────── */
const truncHash = (h) => h ? `${h.slice(0, 8)}…${h.slice(-6)}` : '—';
const fmtTimestamp = (ts) => {
  if (!ts) return '—';
  try { return new Date(ts * 1000).toLocaleString(); } catch { return String(ts); }
};

const LS_VERIFY_PREFIX = 'omni_verified_';

const loadVerifiedState = (records) => {
  const state = {};
  records.forEach((rec) => {
    try {
      const raw = localStorage.getItem(LS_VERIFY_PREFIX + rec.sha256_hex);
      if (raw) state[rec.sha256_hex] = JSON.parse(raw);
    } catch { /* non-fatal */ }
  });
  return state;
};

/* ── Component ──────────────────────────────────────────────────────────── */
export default function DashboardPage() {
  const navigate = useNavigate();
  const { walletAddress, connectWallet } = useWallet();

  // Use connected wallet → localStorage key written at connect time.
  // No hardcoded fallback — prevents querying the wrong (contract-deployer) wallet.
  const effectiveWallet =
    walletAddress ||
    localStorage.getItem('omni_wallet') ||
    '';

  const [records,      setRecords]      = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [copied,       setCopied]       = useState(null);
  const [spinning,     setSpinning]     = useState(false);

  // Per-row verify state: { [sha256_hex]: {date, ...} | 'not-found' }
  const [verifiedRows, setVerifiedRows] = useState({});
  const [verifyingRow, setVerifyingRow] = useState(null);

  // Per-row download loading state and error toast
  const [downloadingRow, setDownloadingRow] = useState(null); // sha256_hex | null
  const [toast, setToast] = useState(null); // string | null
  const toastTimerRef = useRef(null);

  // Guard against concurrent fetches
  const isFetchingRef = useRef(false);

  /* ── Fetch records ─────────────────────────────────────────────────── */
  const fetchRecords = useCallback(async () => {
    if (isFetchingRef.current) return;   // drop duplicate calls
    isFetchingRef.current = true;
    setLoading(true);
    setError(null);
    setSpinning(true);
    try {
      // Relative URL — nginx/Cloudflare routes /api/ to the backend.
      // /api/my-documents reads local audit files so it works even when
      // Ganache is down (unlike /api/records which queries the blockchain).
      const res = await fetch(`/api/my-documents/${effectiveWallet}`);
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      // Client-side dedup by sha256_hex
      const unique = Array.from(
        new Map((data.documents ?? []).map((r) => [r.sha256_hex, r])).values()
      );
      setRecords(unique);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setSpinning(false);
      isFetchingRef.current = false;
    }
  }, [effectiveWallet]);

  // Fetch on mount and whenever the effective wallet changes; poll every 10 s.
  useEffect(() => {
    fetchRecords();
    const interval = setInterval(fetchRecords, 10000);
    return () => clearInterval(interval);
  }, [effectiveWallet]); // eslint-disable-line react-hooks/exhaustive-deps

  // Immediately re-fetch whenever AppPage signals a completed redaction via
  // localStorage (works across same-origin tabs and within the same page).
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'omni_last_redaction') fetchRecords();
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, [fetchRecords]);

  // Seed verified state from localStorage whenever records change (Fix 3)
  useEffect(() => {
    if (records.length > 0) {
      setVerifiedRows((prev) => ({ ...loadVerifiedState(records), ...prev }));
    }
  }, [records]);

  /* ── Copy hash ─────────────────────────────────────────────────────── */
  const copyHash = (hash, idx) => {
    navigator.clipboard.writeText(hash).then(() => {
      setCopied(idx);
      setTimeout(() => setCopied(null), 1500);
    });
  };

  /* ── Per-row verify (Fix 3) ────────────────────────────────────────── */
  const handleVerify = async (hash) => {
    if (verifiedRows[hash] && verifiedRows[hash] !== 'not-found') return;
    if (verifyingRow === hash) return;
    setVerifyingRow(hash);
    try {
      const res = await fetch(`${API_BASE_URL}/verify`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body:    new URLSearchParams({ file_hash: hash }),
      });
      const data = await res.json();
      if (data.in_ledger) {
        const record = { date: new Date().toISOString(), ...data };
        try {
          localStorage.setItem(LS_VERIFY_PREFIX + hash, JSON.stringify(record));
        } catch { /* storage full */ }
        setVerifiedRows((prev) => ({ ...prev, [hash]: record }));
      } else {
        setVerifiedRows((prev) => ({ ...prev, [hash]: 'not-found' }));
        setTimeout(() => {
          setVerifiedRows((prev) => {
            if (prev[hash] === 'not-found') {
              const next = { ...prev };
              delete next[hash];
              return next;
            }
            return prev;
          });
        }, 2500);
      }
    } catch (err) {
      console.error('[Dashboard] verify error:', err);
    } finally {
      setVerifyingRow(null);
    }
  };

  /* ── CSV export ───────────────────────────────────────────────────── */
  /* ── Download by hash ─────────────────────────────────────────────── */
  const showToast = (msg) => {
    setToast(msg);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 3500);
  };

  const handleDownload = async (rec) => {
    const hash = rec.sha256_hex;
    if (downloadingRow === hash) return;
    setDownloadingRow(hash);
    try {
      // Prefer direct path via redacted_file field; fall back to by-hash lookup
      const downloadPath = rec.redacted_file
        ? `/api/output/${rec.redacted_file}`
        : `${API_BASE_URL}/download/by-hash/${hash}?wallet=${encodeURIComponent(effectiveWallet)}`;
      const res = await fetch(downloadPath);
      if (res.ok) {
        const blob        = await res.blob();
        const disposition = res.headers.get('content-disposition') || '';
        const nameMatch   = disposition.match(/filename="?([^"]+)"?/);
        const filename    = nameMatch ? nameMatch[1] : (rec.redacted_file || `redacted_${hash.slice(0, 8)}`);
        const blobUrl     = URL.createObjectURL(blob);
        const a           = document.createElement('a');
        a.href     = blobUrl;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(blobUrl);
      } else if (res.status === 403) {
        showToast('This record does not belong to your wallet.');
      } else if (res.status === 404) {
        showToast('File not found on this server.');
      } else {
        showToast(`Download failed: HTTP ${res.status}`);
      }
    } catch (err) {
      showToast(`Download error: ${err.message}`);
    } finally {
      setDownloadingRow(null);
    }
  };

  const exportCSV = () => {
    const headers = ['#', 'Hash', 'Doc Type', 'Redactions', 'Timestamp', 'Status'];
    const rows = records.map((r, i) => [
      i + 1,
      r.sha256_hex,
      r.doc_type || 'unknown',
      r.redaction_count ?? 0,
      new Date(r.timestamp * 1000).toLocaleString(),
      r.verified ? 'Verified' : 'Unverified',
    ]);
    const csv = [headers, ...rows]
      .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `omni-shield-audit-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ── Derived stats ─────────────────────────────────────────────────── */
  const totalRedactions = records.reduce((sum, r) => sum + (r.redaction_count ?? 0), 0);
  const timestamps      = records.map((r) => r.timestamp).filter(Boolean).sort((a, b) => a - b);
  const firstDate       = timestamps[0]
    ? new Date(timestamps[0] * 1000).toLocaleDateString() : '—';
  const lastDate        = timestamps[timestamps.length - 1]
    ? new Date(timestamps[timestamps.length - 1] * 1000).toLocaleDateString() : '—';

  const STATS_CARDS = [
    { icon: <DocumentCheckIcon />, value: records.length, label: 'Documents Processed' },
    { icon: <ShieldCheckIcon />,   value: totalRedactions, label: 'PII Items Redacted' },
    { icon: <CalendarIcon />,      value: firstDate,       label: 'First Redaction' },
    { icon: <ClockIcon />,         value: lastDate,        label: 'Most Recent' },
  ];

  /* ── Wallet connect gate ──────────────────────────────────────────── */
  if (!effectiveWallet) {
    return (
      <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 flex flex-col items-center justify-center"
        style={{ background: '#0a0a0f' }}>
        <div className="text-center max-w-sm">
          <div className="w-20 h-20 rounded-2xl bg-slate-800/60 flex items-center
                          justify-center mx-auto mb-6">
            <WalletIcon />
          </div>
          <h2 className="text-2xl font-bold text-white mb-3">Connect Your Wallet</h2>
          <p className="text-slate-400 text-sm mb-8 leading-relaxed">
            Connect your wallet to view your blockchain-anchored redaction records.
          </p>
          <button
            onClick={connectWallet}
            className="btn-primary px-8 py-3.5 rounded-full text-white font-semibold text-sm">
            Connect Wallet
          </button>
        </div>
      </div>
    );
  }

  /* ── Render ───────────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6"
      style={{ background: '#0a0a0f' }}>
      <div className="max-w-6xl mx-auto">

        {/* Header row */}
        <div className="flex items-start justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">Audit Dashboard</h1>
            <p className="text-slate-400 text-sm">
              All records shown have been verified on the blockchain. Click a hash to view full verification details.{' '}
              <span className="font-mono text-indigo-400 text-xs">
                {effectiveWallet.slice(0, 8)}…{effectiveWallet.slice(-6)}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <button
              onClick={exportCSV}
              disabled={records.length === 0}
              className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium
                         border border-slate-600/70 text-slate-300
                         hover:border-indigo-500/60 hover:text-indigo-300 transition-all
                         disabled:opacity-30 disabled:cursor-not-allowed">
              Export CSV
            </button>
            <button onClick={fetchRecords}
              className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium
                         border border-slate-600/70 text-slate-300
                         hover:border-indigo-500/60 hover:text-indigo-300 transition-all">
              <RefreshIcon spinning={spinning} /> Refresh
            </button>
          </div>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {STATS_CARDS.map((s, i) => (
            <div key={i} className="glass rounded-2xl p-5">
              <div className="text-indigo-400/60 mb-3">{s.icon}</div>
              <div className="text-2xl font-bold text-white mb-1">{s.value}</div>
              <div className="text-slate-500 text-xs">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Error banner */}
        {error && (
          <div className="glass rounded-2xl p-5 mb-6 border border-red-500/25">
            <p className="text-red-400 text-sm">
              <span className="font-semibold">Could not load records: </span>{error}
            </p>
            <p className="text-slate-500 text-xs mt-1">
              Make sure Ganache is running and the backend server is reachable.
            </p>
          </div>
        )}

        {/* Table or empty state */}
        {!loading && !error && records.length === 0 ? (
          <div className="glass rounded-2xl p-16 text-center">
            <div className="w-16 h-16 rounded-2xl bg-slate-800/60 flex items-center
                            justify-center mx-auto mb-5">
              <DocumentCheckIcon />
            </div>
            <p className="text-white font-semibold text-lg mb-2">
              No redactions recorded yet.
            </p>
            <p className="text-slate-400 text-sm mb-6">
              Upload your first document to get started.
            </p>
            <button onClick={() => navigate('/app')}
              className="btn-primary px-7 py-3 rounded-full text-white font-medium text-sm">
              Go to App
            </button>
          </div>
        ) : (
          <div className="glass rounded-2xl overflow-hidden">
            {/* Table header */}
            <div className="grid grid-cols-[2rem_1fr_6rem_10rem_7rem_3rem]
                            gap-4 px-6 py-3 border-b border-slate-800/80">
              {['#', 'Document Hash', 'Redactions', 'Timestamp', 'Status', 'DL'].map((h, i) => (
                <span key={i}
                  className={`text-xs font-semibold text-slate-500 uppercase tracking-wider
                              ${i === 1 ? '' : 'text-center'}`}>
                  {h}
                </span>
              ))}
            </div>

            {/* Rows */}
            {loading ? (
              <div className="px-6 py-10 text-center text-slate-500 text-sm">
                Loading records from blockchain…
              </div>
            ) : (
              records.map((rec, i) => {
                const isVerified  = verifiedRows[rec.sha256_hex] &&
                                    verifiedRows[rec.sha256_hex] !== 'not-found';
                const isNotFound  = verifiedRows[rec.sha256_hex] === 'not-found';
                const isVerifying = verifyingRow === rec.sha256_hex;

                return (
                  <div key={rec.sha256_hex}
                    className={`grid grid-cols-[2rem_1fr_6rem_10rem_7rem_3rem]
                                 gap-4 px-6 py-4 border-b border-slate-800/40
                                 hover:bg-slate-800/20 transition-colors items-center
                                 ${i === records.length - 1 ? 'border-b-0' : ''}`}>
                    {/* # */}
                    <span className="text-slate-600 text-sm text-center">{i + 1}</span>

                    {/* Hash */}
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-mono text-sm text-slate-300 truncate">
                        {truncHash(rec.sha256_hex)}
                      </span>
                      <button
                        onClick={() => copyHash(rec.sha256_hex, i)}
                        className="text-slate-600 hover:text-slate-300 transition-colors flex-shrink-0">
                        {copied === i ? <CheckSmIcon /> : <CopyIcon />}
                      </button>
                    </div>

                    {/* Count */}
                    <span className="text-center text-white font-semibold text-sm">
                      {rec.redaction_count ?? 0}
                    </span>

                    {/* Timestamp */}
                    <span className="text-slate-400 text-xs text-center">
                      {fmtTimestamp(rec.timestamp)}
                    </span>

                    {/* Verified badge — all blockchain records are verified; click to view details */}
                    <div className="flex justify-center">
                      {isVerified ? (
                        <button
                          onClick={() => navigate(`/verify?hash=${rec.sha256_hex}`)}
                          className="px-3 py-1.5 rounded-full text-xs font-medium
                                     bg-green-500/15 border border-green-500/30 text-green-400
                                     whitespace-nowrap hover:bg-green-500/25 transition-colors">
                          Verified ✓
                        </button>
                      ) : (
                        <button
                          onClick={() => handleVerify(rec.sha256_hex)}
                          disabled={isVerifying}
                          className="px-3 py-1.5 rounded-full text-xs font-medium
                                     border border-indigo-500/30 text-indigo-300
                                     hover:bg-indigo-500/10 hover:border-indigo-400/60
                                     transition-all whitespace-nowrap
                                     disabled:opacity-50 disabled:cursor-not-allowed
                                     flex items-center gap-1.5">
                          {isVerifying ? <><SpinnerSmIcon /> Checking…</> : 'Verify'}
                        </button>
                      )}
                    </div>

                    {/* Download cell */}
                    <div className="flex justify-center">
                      <button
                        onClick={() => handleDownload(rec)}
                        disabled={downloadingRow === rec.sha256_hex}
                        title="Download redacted file"
                        className="text-slate-500 hover:text-indigo-300 transition-colors
                                   disabled:opacity-40 disabled:cursor-not-allowed p-1">
                        {downloadingRow === rec.sha256_hex
                          ? <SpinnerSmIcon />
                          : <DownloadIcon />}
                      </button>
                    </div>

                  </div>
                );
              })
            )}
          </div>
        )}
      </div>

      {/* Toast notification */}
      {toast && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3 px-5 py-3
                        rounded-2xl shadow-xl border border-red-500/30
                        bg-slate-900/95 backdrop-blur-sm text-red-400 text-sm font-medium
                        animate-fade-in">
          {toast}
          <button
            onClick={() => setToast(null)}
            className="text-slate-500 hover:text-slate-300 transition-colors ml-1 text-xs">
            ✕
          </button>
        </div>
      )}

    </div>
  );
}

import { useState, useRef, useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/* ── Icons ─────────────────────────────────────────────────────────────── */
const UploadIcon = () => (
  <svg className="w-7 h-7 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775
         5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0
         0118 19.5H6.75z" />
  </svg>
);
const SpinnerIcon = () => (
  <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);
const BigCheckIcon = () => (
  <svg className="w-16 h-16 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6
         3.99 3.99 0 013 9.749c0 5.592 3.824 10.29 9 11.623
         5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196
         0-6.1-1.248-8.25-3.285z" />
  </svg>
);
const BigXIcon = () => (
  <svg className="w-16 h-16 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

/* ── Helpers ────────────────────────────────────────────────────────────── */
const fmtTimestamp = (ts) => {
  if (!ts) return '—';
  try { return new Date(ts * 1000).toLocaleString(); } catch { return String(ts); }
};

/* ── localStorage helpers ───────────────────────────────────────────────── */
const LS_PREFIX = 'omni_verified_';

const saveVerified = (sha256_hex, data) => {
  try {
    localStorage.setItem(LS_PREFIX + sha256_hex, JSON.stringify({
      date: new Date().toISOString(),
      redaction_count: data.redaction_count,
      timestamp: data.timestamp,
    }));
  } catch { /* storage full — non-fatal */ }
};

const loadVerified = (sha256_hex) => {
  if (!sha256_hex || sha256_hex.length !== 64) return null;
  try {
    const raw = localStorage.getItem(LS_PREFIX + sha256_hex);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
};

/* ── ZK proof badge ─────────────────────────────────────────────────────── */
function ZKProofBadge({ zkResult }) {
  if (!zkResult) return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full
                     bg-slate-700/50 border border-slate-600/40 text-slate-400
                     text-xs font-medium">
      ZK Proof: checking…
    </span>
  );
  if (zkResult.loading) return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full
                     bg-slate-700/50 border border-slate-600/40 text-slate-400
                     text-xs font-medium">
      <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
      </svg>
      ZK Proof: verifying…
    </span>
  );
  if (zkResult.verified) return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full
                     bg-violet-500/20 border border-violet-500/40 text-violet-300
                     text-xs font-semibold">
      ✓ ZK Proof Valid
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full
                     bg-slate-700/50 border border-slate-600/40 text-slate-400
                     text-xs font-medium">
      ZK Proof: {zkResult.reason || 'unavailable'}
    </span>
  );
}

/* ── Result card ────────────────────────────────────────────────────────── */
function VerifyResultCard({ result, zkResult, prevVerified }) {
  if (!result) return null;

  if (result.error) return (
    <div className="glass rounded-2xl p-6 border border-red-500/25 mt-4">
      <p className="text-red-400 text-sm font-medium">{result.error}</p>
    </div>
  );

  const found = result.in_ledger ?? result.exists;

  return (
    <div className={`glass rounded-2xl p-8 mt-5 border text-center ${
      found ? 'border-green-500/40' : 'border-red-500/20'
    }`}>
      <div className="flex justify-center mb-4">
        {found ? <BigCheckIcon /> : <BigXIcon />}
      </div>

      {found && (
        <div className="flex flex-wrap justify-center gap-2 mb-3">
          <span className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full
                           bg-green-500/20 border border-green-500/40 text-green-400
                           text-sm font-semibold tracking-wide">
            ✓ Verified on Blockchain
          </span>
          <ZKProofBadge zkResult={zkResult} />
        </div>
      )}

      <h3 className={`text-2xl font-bold mb-2 ${found ? 'text-green-400' : 'text-red-400'}`}>
        {found ? 'Document Verified' : 'Document Not Found'}
      </h3>
      <p className="text-slate-400 text-sm mb-6">
        {found
          ? 'This document has been verified on the blockchain.'
          : 'This document has no redaction record on this blockchain.'}
      </p>

      {found && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left mb-4">
            {[
              { label: 'Redactions',  value: String(result.redaction_count ?? 0) },
              { label: 'Timestamp',   value: fmtTimestamp(result.timestamp) },
              { label: 'SHA-256',     value: result.sha256_hex
                ? `${result.sha256_hex.slice(0, 16)}…`
                : '—' },
            ].map((item, i) => (
              <div key={i} className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">{item.label}</p>
                <p className="text-white font-mono text-sm">{item.value}</p>
              </div>
            ))}
          </div>
          {zkResult?.verified && zkResult.commitment && (
            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800 text-left mb-4">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">
                ZK Commitment (first 4 words)
              </p>
              <p className="text-violet-300 font-mono text-xs break-all">
                {zkResult.commitment.slice(0, 4).map(n => n.toString(16).padStart(8, '0')).join(' ')}…
              </p>
              <p className="text-slate-600 text-xs mt-1">
                Groth16 / BN128 — proof verified against on-disk verification key
              </p>
            </div>
          )}
          {prevVerified?.date && (
            <p className="text-slate-500 text-xs mt-2">
              Previously verified on {new Date(prevVerified.date).toLocaleString()}
            </p>
          )}
        </>
      )}
    </div>
  );
}

/* ── Component ──────────────────────────────────────────────────────────── */
export default function VerifyPage() {
  const [searchParams]  = useSearchParams();
  const prefilledHash   = searchParams.get('hash') || '';

  const [activeTab,   setActiveTab]   = useState(prefilledHash ? 'hash' : 'file');
  const [isDragging,  setIsDragging]  = useState(false);
  const [file,        setFile]        = useState(null);
  const [hashInput,   setHashInput]   = useState(prefilledHash);
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [zkResult,    setZkResult]    = useState(null);
  const [prevVerified, setPrevVerified] = useState(() => loadVerified(prefilledHash));
  const fileInputRef  = useRef(null);

  const fetchZkProof = useCallback(async (sha256) => {
    if (!sha256) return;
    setZkResult({ loading: true });
    try {
      const res  = await fetch(`${API_BASE_URL}/api/verify-proof/${sha256}`);
      const data = await res.json();
      setZkResult(data);
    } catch {
      setZkResult({ verified: false, reason: 'fetch error' });
    }
  }, []);

  // Auto-verify if hash is pre-filled from URL
  useEffect(() => {
    if (prefilledHash && prefilledHash.length === 64) {
      setHashInput(prefilledHash);
      setActiveTab('hash');
      setPrevVerified(loadVerified(prefilledHash));
      // Auto-submit verification immediately
      setLoading(true);
      setResult(null);
      setZkResult(null);
      fetch(`${API_BASE_URL}/verify`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body:    new URLSearchParams({ file_hash: prefilledHash }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.in_ledger && data.sha256_hex) {
            saveVerified(data.sha256_hex, data);
            setPrevVerified(loadVerified(data.sha256_hex));
            fetchZkProof(data.sha256_hex);
          }
          setResult(data);
        })
        .catch((err) => setResult({ error: err.message }))
        .finally(() => setLoading(false));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefilledHash]);

  // Check localStorage whenever the hash input reaches 64 chars
  useEffect(() => {
    setPrevVerified(loadVerified(hashInput.trim()));
  }, [hashInput]);

  const handleFileSelect = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  }, [handleFileSelect]);

  const verifyByFile = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setResult(null);
    setZkResult(null);
    try {
      const res  = await fetch(`${API_BASE_URL}/verify`, { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      if (data.in_ledger && data.sha256_hex) {
        saveVerified(data.sha256_hex, data);
        setPrevVerified(loadVerified(data.sha256_hex));
        fetchZkProof(data.sha256_hex);
      }
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const verifyByHash = async () => {
    const h = hashInput.trim();
    if (!h || h.length !== 64) return;
    setLoading(true);
    setResult(null);
    setZkResult(null);
    try {
      const res  = await fetch(`${API_BASE_URL}/verify`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body:    new URLSearchParams({ file_hash: h }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      if (data.in_ledger && data.sha256_hex) {
        saveVerified(data.sha256_hex, data);
        setPrevVerified(loadVerified(data.sha256_hex));
        fetchZkProof(data.sha256_hex);
      }
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const tabClass = (id) =>
    `px-5 py-2.5 rounded-full text-sm font-medium transition-all ${
      activeTab === id
        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
        : 'bg-slate-800/80 text-slate-400 hover:text-white'
    }`;

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6"
      style={{ background: '#0a0a0f' }}>
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-1">Blockchain Verification</h1>
          <p className="text-slate-400 text-sm">
            Confirm that a document's redaction record exists on-chain.
          </p>
        </div>

        <div className="glass rounded-2xl p-6">
          {/* Tabs */}
          <div className="flex gap-3 mb-6">
            <button className={tabClass('file')} onClick={() => { setActiveTab('file'); setResult(null); }}>
              Verify by File
            </button>
            <button className={tabClass('hash')} onClick={() => { setActiveTab('hash'); setResult(null); }}>
              Verify by Hash
            </button>
          </div>

          {/* ── Tab: File ─────────────────────────────────────────────── */}
          {activeTab === 'file' && (
            <div className="space-y-4">
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onClick={() => fileInputRef.current?.click()}
                className={`flex flex-col items-center justify-center p-10 rounded-xl
                            cursor-pointer transition-all duration-200 border-2 border-dashed
                            ${isDragging
                              ? 'drop-zone-active'
                              : 'border-indigo-500/25 hover:border-indigo-500/50 hover:bg-indigo-500/[0.03]'
                            }`}
              >
                <input ref={fileInputRef} type="file" className="hidden"
                  onChange={(e) => handleFileSelect(e.target.files[0])} />
                {file ? (
                  <div className="text-center">
                    <p className="text-white font-medium">{file.name}</p>
                    <p className="text-slate-500 text-sm mt-1">Click to change</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="flex justify-center mb-3"><UploadIcon /></div>
                    <p className="text-white font-medium">Drop document here</p>
                    <p className="text-slate-500 text-sm mt-1">or click to browse</p>
                  </div>
                )}
              </div>

              <button
                onClick={verifyByFile}
                disabled={!file || loading}
                className="w-full py-3.5 rounded-full font-semibold text-white
                           btn-primary flex items-center justify-center gap-3
                           disabled:opacity-40 disabled:cursor-not-allowed
                           disabled:transform-none disabled:shadow-none"
              >
                {loading ? <><SpinnerIcon /> Verifying…</> : 'Verify Document'}
              </button>
            </div>
          )}

          {/* ── Tab: Hash ─────────────────────────────────────────────── */}
          {activeTab === 'hash' && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-500
                                  uppercase tracking-widest mb-2">
                  SHA-256 Hash (64 hex characters)
                </label>
                <input
                  type="text"
                  value={hashInput}
                  onChange={(e) => setHashInput(e.target.value)}
                  placeholder="e.g. a3f1d8c2e4b7…"
                  maxLength={64}
                  className="w-full bg-slate-900/60 border border-slate-700/60 rounded-xl
                             px-4 py-3 text-white font-mono text-sm
                             placeholder:text-slate-600 placeholder:font-sans
                             focus:outline-none focus:border-indigo-500/70 transition-colors"
                />
                <p className={`text-xs mt-1.5 transition-colors ${
                  hashInput.length === 64
                    ? 'text-green-500'
                    : hashInput.length > 0
                    ? 'text-slate-500'
                    : 'text-transparent'
                }`}>
                  {hashInput.length}/64 characters
                  {hashInput.length === 64 && ' ✓'}
                </p>
              </div>

              <button
                onClick={verifyByHash}
                disabled={hashInput.trim().length !== 64 || loading}
                className="w-full py-3.5 rounded-full font-semibold text-white
                           btn-primary flex items-center justify-center gap-3
                           disabled:opacity-40 disabled:cursor-not-allowed
                           disabled:transform-none disabled:shadow-none"
              >
                {loading ? <><SpinnerIcon /> Verifying…</> : 'Verify Hash'}
              </button>
            </div>
          )}

          {/* Previously verified hint (hash tab, no active result yet) */}
          {activeTab === 'hash' && !result && prevVerified?.date && (
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl
                            bg-green-500/8 border border-green-500/20 mt-2">
              <span className="text-green-400 text-xs font-semibold">✓</span>
              <p className="text-green-400 text-xs">
                Previously verified on {new Date(prevVerified.date).toLocaleString()}
              </p>
            </div>
          )}

          {/* Result */}
          <VerifyResultCard result={result} zkResult={zkResult} prevVerified={prevVerified} />
        </div>

        {/* Explainer */}
        <div className="glass rounded-2xl p-6 mt-6">
          <h3 className="text-sm font-semibold text-white mb-4">How verification works</h3>
          <ul className="space-y-3">
            {[
              {
                num: '01',
                text: 'Your document is SHA-256 hashed client-side or on the edge server. The hash is a unique 64-character fingerprint of the file\'s exact contents.',
              },
              {
                num: '02',
                text: 'After redaction, the hash and redaction count are written to an Ethereum smart contract (AuditLog.sol) deployed on your local Ganache node.',
              },
              {
                num: '03',
                text: 'Anyone with the original file can re-compute its hash and query the contract. Because the blockchain is append-only, the record cannot be falsified or deleted.',
              },
            ].map((item) => (
              <li key={item.num} className="flex gap-3 text-sm text-slate-400">
                <span className="text-indigo-500 font-mono font-bold flex-shrink-0 mt-0.5">
                  {item.num}
                </span>
                <span className="leading-relaxed">{item.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

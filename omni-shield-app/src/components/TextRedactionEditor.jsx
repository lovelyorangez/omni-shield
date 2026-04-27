import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const SpinnerIcon = () => (
  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
  </svg>
);

const DOC_TYPE_LABELS = { text: 'Text', docx: 'Word', pptx: 'Slides' };

/**
 * TextRedactionEditor — shown after text/docx/pptx redaction completes.
 *
 * Props:
 *   result        — the API response object (file_hash, redactions, redacted_file, doc_type…)
 *   docType       — 'text' | 'docx' | 'pptx'
 *   onDownload    — called with (filename) to trigger download via the parent's downloadFile()
 */
export default function TextRedactionEditor({ result, docType, onDownload }) {
  const redactionObjects = (result.redactions || result.redacted_items || []).map((r) =>
    typeof r === 'object' ? r : { label: 'PII', value: r, source: 'ai' }
  );

  // De-duplicate by value
  const seen = new Set();
  const unique = redactionObjects.filter((r) => {
    if (!r.value || seen.has(r.value)) return false;
    seen.add(r.value);
    return true;
  });

  const [checked, setChecked]   = useState(() => Object.fromEntries(unique.map((_, i) => [i, true])));
  const [customInput, setCustomInput] = useState('');
  const [customTerms, setCustomTerms] = useState([]);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState(null);

  const toggleAll = (val) =>
    setChecked(Object.fromEntries(unique.map((_, i) => [i, val])));

  const addCustom = () => {
    const t = customInput.trim();
    if (t && !customTerms.includes(t)) {
      setCustomTerms((prev) => [...prev, t]);
    }
    setCustomInput('');
  };

  const removeCustom = (term) =>
    setCustomTerms((prev) => prev.filter((t) => t !== term));

  const handleApplyDownload = async () => {
    setApplying(true);
    setApplyError(null);
    try {
      if (docType === 'text') {
        // Re-apply confirmed items via backend
        const confirmed = unique.filter((_, i) => checked[i]).map((r) => r.value);
        const res = await fetch(`${API_BASE_URL}/redact/text/apply`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            file_hash:             result.file_hash,
            confirmed_redactions:  confirmed,
            custom_terms:          customTerms,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        onDownload(data.redacted_file);
      } else {
        // docx / pptx: file already generated, just download it
        onDownload(result.redacted_file);
      }
    } catch (err) {
      setApplyError(err.message);
    } finally {
      setApplying(false);
    }
  };

  const checkedCount = Object.values(checked).filter(Boolean).length + customTerms.length;
  const typeLabel = DOC_TYPE_LABELS[docType] || docType;

  return (
    <div className="glass rounded-2xl p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Review Redactions</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            {typeLabel} · {unique.length} item{unique.length !== 1 ? 's' : ''} detected
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => toggleAll(true)}
            className="px-3 py-1.5 rounded-full text-xs font-medium border border-slate-600/70
                       text-slate-400 hover:text-white hover:border-indigo-500/60 transition-all">
            All on
          </button>
          <button
            onClick={() => toggleAll(false)}
            className="px-3 py-1.5 rounded-full text-xs font-medium border border-slate-600/70
                       text-slate-400 hover:text-white hover:border-slate-500 transition-all">
            All off
          </button>
        </div>
      </div>

      {/* Redaction list */}
      <div className="border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-4 py-2 border-b border-slate-800 bg-slate-900/40">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
            AI Detected Redactions
          </p>
        </div>
        {unique.length === 0 ? (
          <p className="text-slate-600 text-sm text-center py-6">No items detected</p>
        ) : (
          <div className="max-h-72 overflow-y-auto divide-y divide-slate-800/60">
            {unique.map((r, i) => (
              <label
                key={i}
                className={`flex items-center gap-3 px-4 py-2.5 cursor-pointer
                            hover:bg-slate-800/30 transition-colors
                            ${i % 2 === 0 ? 'bg-slate-900/30' : 'bg-slate-800/10'}`}
              >
                <input
                  type="checkbox"
                  checked={checked[i] ?? true}
                  onChange={(e) => setChecked((prev) => ({ ...prev, [i]: e.target.checked }))}
                  className="w-3.5 h-3.5 rounded accent-indigo-500 flex-shrink-0"
                />
                <span className="flex-1 text-slate-300 text-sm font-mono truncate">
                  {r.value}
                </span>
                {r.label && r.label !== 'text' && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-medium
                                   bg-indigo-500/15 text-indigo-300 border border-indigo-500/20
                                   flex-shrink-0 uppercase tracking-wide">
                    {r.label}
                  </span>
                )}
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Custom terms */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
          Add manual term
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addCustom(); } }}
            placeholder="Type a term to redact…"
            className="flex-1 bg-slate-900/60 border border-slate-700/60 rounded-xl
                       px-3 py-2 text-sm text-white placeholder:text-slate-600
                       focus:outline-none focus:border-indigo-500/70 transition-colors"
          />
          <button
            onClick={addCustom}
            disabled={!customInput.trim()}
            className="px-4 py-2 rounded-xl text-sm font-medium border border-indigo-500/40
                       text-indigo-300 hover:bg-indigo-500/10 transition-all
                       disabled:opacity-30 disabled:cursor-not-allowed">
            +
          </button>
        </div>
        {customTerms.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {customTerms.map((t) => (
              <span key={t}
                className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs
                           bg-amber-500/15 border border-amber-500/30 text-amber-300">
                {t}
                <button
                  onClick={() => removeCustom(t)}
                  className="text-amber-400/60 hover:text-amber-300 transition-colors">
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {applyError && (
        <p className="text-red-400 text-sm">{applyError}</p>
      )}

      {/* Apply button */}
      <button
        onClick={handleApplyDownload}
        disabled={applying || checkedCount === 0}
        className="w-full py-3 rounded-full font-semibold text-white text-sm
                   flex items-center justify-center gap-2 transition-all
                   bg-gradient-to-r from-indigo-600 to-cyan-600
                   hover:from-indigo-500 hover:to-cyan-500
                   shadow-lg shadow-indigo-500/20
                   disabled:opacity-40 disabled:cursor-not-allowed
                   disabled:shadow-none">
        {applying ? (
          <><SpinnerIcon /> Applying…</>
        ) : (
          <>Apply &amp; Download ({checkedCount} item{checkedCount !== 1 ? 's' : ''})</>
        )}
      </button>
    </div>
  );
}

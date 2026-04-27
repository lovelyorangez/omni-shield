import { useState, useCallback } from 'react';

/* ── Presets ──────────────────────────────────────────────────────────────── */
const FULL_SCAN = {
  names: true, faces: true, signatures: true, phones: true, emails: true,
  dob: true, id_numbers: true, addresses: true, org_names: true, dates: true,
};
const PRESETS = {
  'Full Scan': FULL_SCAN,
  Legal: {
    names: true, faces: false, signatures: true, phones: false, emails: false,
    dob: false, id_numbers: true, addresses: true, org_names: false, dates: true,
  },
  Medical: {
    names: true, faces: false, signatures: false, phones: false, emails: false,
    dob: true, id_numbers: false, addresses: true, org_names: false, dates: true,
  },
  Minimal: {
    names: true, faces: true, signatures: false, phones: false, emails: false,
    dob: false, id_numbers: false, addresses: false, org_names: false, dates: false,
  },
};

/* ── Toggle config ────────────────────────────────────────────────────────── */
const TOGGLES = [
  // [key, icon, label]         — two columns, left first then right
  ['names',      '👤', 'Names (people)'],
  ['faces',      '🖼', 'Faces (photos)'],
  ['phones',     '📞', 'Phone numbers'],
  ['signatures', '✍', 'Signatures'],
  ['emails',     '📧', 'Email addresses'],
  ['id_numbers', '🪪', 'ID numbers'],
  ['dob',        '📅', 'Dates of birth'],
  ['org_names',  '🏢', 'Organisation names'],
  ['addresses',  '📍', 'Addresses'],
  ['dates',      '📆', 'Dates (general)'],
];

/* ── Toggle switch component ─────────────────────────────────────────────── */
function Toggle({ checked, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full
                  transition-colors duration-200 focus:outline-none
                  ${checked ? 'bg-indigo-600' : 'bg-slate-700'}`}
    >
      <span
        className={`inline-block h-4 w-4 mt-0.5 rounded-full bg-white shadow
                    transform transition-transform duration-200
                    ${checked ? 'translate-x-4' : 'translate-x-0.5'}`}
      />
    </button>
  );
}

/* ── Main component ───────────────────────────────────────────────────────── */
export default function RedactionControls({ onChange }) {
  const [filter,       setFilter]       = useState(FULL_SCAN);
  const [activePreset, setActivePreset] = useState('Full Scan');
  const [customInput,  setCustomInput]  = useState('');
  const [customTags,   setCustomTags]   = useState([]);   // {text, isRegex}

  /* Emit combined state up whenever anything changes */
  const emit = useCallback((nextFilter, nextTags) => {
    const terms    = nextTags.filter((t) => !t.isRegex).map((t) => t.text);
    const patterns = nextTags.filter((t) =>  t.isRegex).map((t) => t.text);
    onChange({ pii_filter: nextFilter, custom_terms: terms, custom_patterns: patterns });
  }, [onChange]);

  const applyPreset = (name) => {
    setActivePreset(name);
    setFilter(PRESETS[name]);
    emit(PRESETS[name], customTags);
  };

  const toggleKey = (key) => {
    const next = { ...filter, [key]: !filter[key] };
    setFilter(next);
    setActivePreset(null);   // custom — deselect preset
    emit(next, customTags);
  };

  const addTerm = () => {
    const raw = customInput.trim();
    if (!raw || customTags.length >= 20) return;
    const isRegex = raw.startsWith('/') && raw.endsWith('/') && raw.length > 2;
    const text    = isRegex ? raw.slice(1, -1) : raw;
    const next    = [...customTags, { text, isRegex }];
    setCustomTags(next);
    setCustomInput('');
    emit(filter, next);
  };

  const removeTerm = (i) => {
    const next = customTags.filter((_, idx) => idx !== i);
    setCustomTags(next);
    emit(filter, next);
  };

  return (
    <div className="glass rounded-2xl p-5 space-y-5">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
        Redaction Controls
      </p>

      {/* ── Section A: Preset profiles ─────────────────────────────── */}
      <div className="flex flex-wrap gap-2">
        {Object.keys(PRESETS).map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => applyPreset(name)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all
                        ${activePreset === name
                          ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/60'
                          : 'bg-slate-800/80 text-slate-400 border border-transparent hover:text-white hover:border-slate-600'
                        }`}
          >
            {name}
          </button>
        ))}
        {!activePreset && (
          <span className="px-3 py-1.5 rounded-full text-xs font-medium text-slate-600
                           border border-slate-800/60 self-center">
            Custom
          </span>
        )}
      </div>

      {/* ── Section B: Category toggles ─────────────────────────────── */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
        {TOGGLES.map(([key, icon, label]) => (
          <label key={key}
            className="flex items-center justify-between gap-2 cursor-pointer
                       py-1 hover:opacity-80 transition-opacity">
            <span className="flex items-center gap-2 text-sm text-slate-300">
              <span className="text-base leading-none">{icon}</span>
              {label}
            </span>
            <Toggle checked={!!filter[key]} onChange={() => toggleKey(key)} />
          </label>
        ))}
      </div>

      {/* ── Section C: Custom terms ──────────────────────────────────── */}
      <div className="space-y-2.5">
        <div className="flex gap-2">
          <input
            type="text"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addTerm()}
            placeholder='Add word, phrase, or /regex/'
            maxLength={120}
            className="flex-1 bg-slate-900/60 border border-slate-700/60 rounded-xl
                       px-3 py-2 text-white text-sm placeholder:text-slate-600
                       focus:outline-none focus:border-indigo-500/70 transition-colors"
          />
          <button
            type="button"
            onClick={addTerm}
            disabled={!customInput.trim() || customTags.length >= 20}
            className="px-4 py-2 rounded-xl text-sm font-medium transition-all
                       bg-indigo-600/20 text-indigo-300 border border-indigo-500/30
                       hover:bg-indigo-600/35 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            + Add
          </button>
        </div>

        {customTags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {customTags.map((t, i) => (
              <span key={i}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
                           font-medium border
                           bg-indigo-500/10 text-indigo-300 border-indigo-500/25">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                {t.isRegex ? (
                  <>
                    <span className="font-mono">{t.text}</span>
                    <span className="text-indigo-500 text-[10px] font-semibold uppercase
                                     tracking-wide bg-indigo-500/15 px-1 rounded">
                      regex
                    </span>
                  </>
                ) : (
                  <span>"{t.text}"</span>
                )}
                <button
                  type="button"
                  onClick={() => removeTerm(i)}
                  className="ml-0.5 text-indigo-500 hover:text-red-400 transition-colors
                             leading-none flex-shrink-0"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        {customTags.length >= 20 && (
          <p className="text-xs text-slate-600">Maximum 20 custom terms reached.</p>
        )}
      </div>
    </div>
  );
}

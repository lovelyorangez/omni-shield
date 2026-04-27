import { useNavigate } from 'react-router-dom';

/* ── Icons ─────────────────────────────────────────────────────────────── */
const ShieldBgIcon = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg"
    className="w-full h-full">
    <path d="M100 10 L180 40 L180 100 C180 145 145 175 100 190 C55 175 20 145 20 100 L20 40 Z"
      stroke="currentColor" strokeWidth="3" fill="none" />
    <path d="M70 100 L88 118 L130 76"
      stroke="currentColor" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const LockIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75
         0h10.5a.75.75 0 01.75.75v7.5a.75.75 0
         01-.75.75H6.75a.75.75 0 01-.75-.75v-7.5a.75.75 0 01.75-.75z" />
  </svg>
);

const ChainIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5
         0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5
         4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
  </svg>
);

const BuildingIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75
         6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75
         3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621
         0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3.75h.008v.008h-.008v-.008zm0
         3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z" />
  </svg>
);

/* ── Data ───────────────────────────────────────────────────────────────── */
const FEATURES = [
  {
    icon: <LockIcon />,
    title: '100% Local Processing',
    body: 'Your documents never leave your hardware. AI inference runs on your GPU using 4-bit quantized models.',
  },
  {
    icon: <ChainIcon />,
    title: 'Cryptographic Audit Trail',
    body: 'Every redaction is SHA-256 hashed and anchored to an Ethereum smart contract. Tamper-proof compliance records.',
  },
  {
    icon: <BuildingIcon />,
    title: 'Multi-Format Support',
    body: 'Images, PDFs, and audio. Supports ID cards from USA, India, UK, Germany, Australia, and Canada.',
  },
];

const STATS = [
  { value: '36s',     label: 'Average processing time' },
  { value: '8',       label: 'AI pipeline stages' },
  { value: '6',       label: 'Countries supported' },
  { value: '100%',    label: 'Local execution' },
];

/* ── Component ──────────────────────────────────────────────────────────── */
export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen animated-gradient-bg">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <div className="relative flex flex-col min-h-screen overflow-hidden">

        {/* Background shield */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none">
          <div className="text-indigo-500/[0.04] shield-float w-[600px] h-[600px] max-w-full">
            <ShieldBgIcon />
          </div>
        </div>

        {/* Radial glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-[700px] h-[700px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 70%)',
          }} />

        {/* Content — flex-1 so it fills the space above the scroll indicator */}
        <div className="relative z-10 flex-1 flex items-center justify-center px-6 text-center">
          <div className="max-w-4xl mx-auto py-16">

            {/* Brand mark */}
            <div className="flex items-center justify-center gap-2 mb-8">
              <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955
                     11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824
                     10.29 9 11.622 5.176-1.332 9-6.03 9-11.622
                     0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span className="text-white font-bold tracking-tight">OMNI-SHIELD</span>
            </div>

            {/* Status badge */}
            <div className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full
                            border border-indigo-500/25 bg-indigo-500/5
                            text-indigo-300 text-sm mb-10">
              <span className="w-2 h-2 rounded-full bg-emerald-400"
                style={{ animation: 'pulse-ring 2.5s ease-in-out infinite' }} />
              All inference runs locally on your hardware
            </div>

            {/* Headline */}
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-white leading-tight mb-4">
              Enterprise PII Redaction.
            </h1>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight mb-8 gradient-text">
              Zero Data Leaves Your Machine.
            </h1>

            {/* Subheadline */}
            <p className="text-lg sm:text-xl text-slate-400 max-w-2xl mx-auto mb-12 leading-relaxed">
              AI-powered document redaction with cryptographic audit trails. Built for
              compliance teams who cannot afford to trust the cloud.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => navigate('/app')}
                className="btn-primary px-9 py-4 rounded-full text-white font-semibold
                           text-lg shadow-xl w-full sm:w-auto"
              >
                Start Redacting
              </button>
              <button
                onClick={() => navigate('/dashboard')}
                className="px-9 py-4 rounded-full text-white font-medium text-lg w-full sm:w-auto
                           border border-slate-600/70 hover:border-indigo-500/60
                           hover:text-indigo-300 transition-all duration-200"
              >
                View Audit Dashboard
              </button>
            </div>
          </div>
        </div>

        {/* Scroll indicator — sits at the bottom of the hero as a flex item, never duplicated */}
        <div className="relative z-10 flex justify-center pb-8 pt-2">
          <span className="text-slate-600 animate-bounce text-xs tracking-widest uppercase">
            scroll
          </span>
        </div>
      </div>

      {/* ── Feature cards ────────────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto px-6 py-24">
        <p className="text-center text-sm font-semibold text-indigo-400 uppercase tracking-widest mb-4">
          Why Omni-Shield
        </p>
        <h2 className="text-3xl sm:text-4xl font-bold text-white text-center mb-14">
          Built for the privacy-first enterprise
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-20">
          {FEATURES.map((f, i) => (
            <div key={i} className="glass rounded-2xl p-8 hover:border-indigo-500/30 transition-colors duration-300">
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20
                              flex items-center justify-center text-indigo-400 mb-6"
                style={{ boxShadow: '0 0 20px rgba(99,102,241,0.3)' }}>
                {f.icon}
              </div>
              <h3 className="text-lg font-semibold text-white mb-3">{f.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>

        {/* ── Stats ─────────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-20">
          {STATS.map((s, i) => (
            <div key={i} className="glass rounded-2xl p-6 text-center">
              <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">{s.value}</div>
              <div className="text-slate-500 text-sm">{s.label}</div>
            </div>
          ))}
        </div>

        {/* ── Footer ────────────────────────────────────────────────────── */}
        <p className="text-center text-slate-600 text-sm">
          Built with PyTorch · Ethereum · ZK-SNARKs · Runs on NVIDIA RTX
        </p>
      </div>
    </div>
  );
}

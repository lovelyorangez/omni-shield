import { NavLink, Link } from 'react-router-dom';
import { useWallet } from '../context/WalletContext.jsx';

const ShieldIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955
         11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824
         10.29 9 11.622 5.176-1.332 9-6.03 9-11.622
         0-1.042-.133-2.052-.382-3.016z" />
  </svg>
);

const WalletIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round"
      d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0
         00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
  </svg>
);

export default function Navbar() {
  const { walletAddress, connectWallet } = useWallet();

  const linkClass = ({ isActive }) =>
    `text-sm font-medium transition-colors duration-150 ${
      isActive
        ? 'text-indigo-400'
        : 'text-slate-400 hover:text-white'
    }`;

  return (
    <nav className="glass-navbar fixed top-0 left-0 right-0 z-50">
      <div className="max-w-6xl mx-auto px-6 py-3.5 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="text-indigo-400 group-hover:text-indigo-300 transition-colors">
            <ShieldIcon />
          </span>
          <span className="font-bold text-white tracking-tight text-base">
            OMNI-SHIELD
          </span>
        </Link>

        <div className="flex items-center gap-7">
          {/* Nav links */}
          <div className="hidden sm:flex items-center gap-6">
            <NavLink to="/app"       className={linkClass}>App</NavLink>
            <NavLink to="/dashboard" className={linkClass}>Dashboard</NavLink>
            <NavLink to="/verify"    className={linkClass}>Verify</NavLink>
          </div>

          {/* Wallet button — pulses when no wallet connected */}
          <button
            onClick={connectWallet}
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium
                       border border-indigo-500/30 text-indigo-300
                       hover:bg-indigo-500/10 hover:border-indigo-400/60
                       transition-all duration-200
                       ${!walletAddress ? 'animate-pulse' : ''}`}
          >
            <WalletIcon />
            {walletAddress
              ? `${walletAddress.slice(0, 6)}…${walletAddress.slice(-4)}`
              : 'Connect Wallet'}
          </button>
        </div>
      </div>
    </nav>
  );
}

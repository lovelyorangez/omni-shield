import { Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar.jsx';
import LandingPage from './pages/LandingPage.jsx';
import AppPage from './pages/AppPage.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import VerifyPage from './pages/VerifyPage.jsx';
import SystemStatsPanel from './components/SystemStatsPanel.jsx';

export default function App() {
  const location = useLocation();
  const showNav = location.pathname !== '/';

  return (
    <>
      <SystemStatsPanel />
      {showNav && <Navbar />}
      <Routes>
        <Route path="/"          element={<LandingPage />} />
        <Route path="/app"       element={<AppPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/verify"    element={<VerifyPage />} />
      </Routes>
    </>
  );
}

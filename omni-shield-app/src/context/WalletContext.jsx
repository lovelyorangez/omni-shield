import { createContext, useContext, useState } from 'react';

export const WalletContext = createContext(null);

export function WalletProvider({ children }) {
  const [walletAddress, setWalletAddress] = useState('');

  const connectWallet = async () => {
    if (!window.ethereum) {
      alert('MetaMask is not installed. Please install it to use wallet features.');
      return;
    }
    try {
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
      if (accounts?.length > 0) {
        setWalletAddress(accounts[0]);
        localStorage.setItem('omni_wallet', accounts[0]);
      }
    } catch {
      // user rejected
    }
  };

  return (
    <WalletContext.Provider value={{ walletAddress, setWalletAddress, connectWallet }}>
      {children}
    </WalletContext.Provider>
  );
}

export const useWallet = () => useContext(WalletContext);

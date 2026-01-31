const { app, BrowserWindow } = require('electron');
    const path = require('path');

    function createWindow() {
      const win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
          nodeIntegration: true,
          contextIsolation: false,
        },
      });

      // KEY FIX: In dev mode, load from localhost. In prod, load file.
      // We check if the app is packaged or if we are running via 'npm run dev'
      const isDev = !app.isPackaged; 
      const startUrl = isDev 
        ? 'http://localhost:5173' 
        : `file://${path.join(__dirname, '../dist/index.html')}`;

      console.log(`Loading URL: ${startUrl}`); // Debug print
      win.loadURL(startUrl);
    }

    app.whenReady().then(createWindow);

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') app.quit();
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
    

import React, { useEffect } from 'react';
import { useThemeStore } from './store/useThemeStore';
import './styles/main.scss';
import AppRouter from './router';

function App() {
  const { theme } = useThemeStore();

  useEffect(() => {
    // Limpia las clases de tema anteriores y añade la actual al body
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${theme}-theme`);
  }, [theme]);

  return (
    <div className="app-container">
      <AppRouter />
    </div>
  );
}

export default App;

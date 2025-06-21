import React from 'react';
import { useThemeStore } from '../../store/useThemeStore';
import './ThemeSwitcher.scss';

export const ThemeSwitcher: React.FC = () => {
  const { theme, toggleTheme } = useThemeStore();

  return (
    <button className="theme-switcher" onClick={toggleTheme}>
      {theme === 'light' ? '🌙' : '☀️'}
    </button>
  );
}; 
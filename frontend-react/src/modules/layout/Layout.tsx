import React, { useState, useEffect } from 'react';
import { ThemeToggle } from '../ui/ThemeToggle';

export const Layout: React.FC<React.PropsWithChildren> = ({ children }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div className="min-h-screen flex flex-col">
      <Header onMenu={()=>setMenuOpen(o=>!o)} />
      <div className="flex-1 flex flex-row overflow-hidden">
        <aside className={`w-64 bg-panelAlt border-r border-slate-800/70 hidden md:block`}></aside>
        <main className="flex-1 overflow-auto p-6 md:p-8 space-y-8">{children}</main>
      </div>
      <footer className="text-center text-xs text-slate-500 py-6">© 2025 PPE Vision Console</footer>
    </div>
  );
};

const Header: React.FC<{onMenu:()=>void}> = ({ onMenu }) => {
  return (
    <header className="h-16 backdrop-blur bg-bg/60 border-b border-slate-800/70 flex items-center px-4 md:px-8 gap-6">
      <button onClick={onMenu} className="md:hidden btn-outline px-2 py-1 rounded-md">☰</button>
      <h1 className="text-base font-semibold tracking-wide bg-gradient-to-r from-accent to-accentB text-transparent bg-clip-text flex items-center gap-2">
        PPE Vision Console <span className="text-[11px] font-medium bg-panelAlt border border-slate-700 px-2 py-0.5 rounded-full">YOLOv8</span>
      </h1>
      <div className="flex-1" />
      <nav className="hidden md:flex items-center gap-6 text-sm text-slate-300">
        <a href="#" className="hover:text-white transition-colors">Settings</a>
        <a href="#" className="hover:text-white transition-colors">Help</a>
        <a href="#" className="hover:text-white transition-colors">Logout</a>
      </nav>
      <ThemeToggle />
    </header>
  );
};

import React, { useEffect, useState } from 'react';

export const ThemeToggle: React.FC = () => {
  const [dark, setDark] = useState(true);
  useEffect(()=>{
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('theme:isDark', dark? '1':'0');
  },[dark]);
  useEffect(()=>{ const saved = localStorage.getItem('theme:isDark'); if(saved!==null) setDark(saved==='1'); },[]);
  return (
    <button onClick={()=>setDark(d=>!d)} className="btn-outline text-xs px-3 py-1.5 rounded-md">
      {dark? '🌙 Dark':'☀️ Light'}
    </button>
  );
};

// Central API base configuration. Vite exposes import.meta.env.VITE_API_BASE
export const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000';

export const apiUrl = (path: string) => {
  if(path.startsWith('http')) return path;
  return API_BASE.replace(/\/$/, '') + path;
};
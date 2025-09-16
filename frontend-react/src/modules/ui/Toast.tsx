import React, { createContext, useContext, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';

export interface Toast { id:string; message:string; type:'success'|'error'|'info'; }
interface ToastCtx { push(t:Omit<Toast,'id'>):void }
const Ctx = createContext<ToastCtx>({ push: ()=>{} });

export const ToastHost: React.FC = () => {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const remove = useCallback((id:string)=> setToasts(t=> t.filter(x=>x.id!==id)),[]);
  const push = useCallback((t:Omit<Toast,'id'>)=>{
    const id = Math.random().toString(36).slice(2);
    setToasts(ts=> [...ts, { ...t, id }]);
    setTimeout(()=> remove(id), 4000);
  },[remove]);
  return createPortal(
    <Ctx.Provider value={{ push }}>
      <div className="fixed top-4 right-4 space-y-3 z-50 w-72">
        {toasts.map(t=> <ToastItem key={t.id} toast={t} onClose={()=>remove(t.id)} />)}
      </div>
    </Ctx.Provider>,
    document.body
  );
};

export function useToasts(){ return useContext(Ctx); }

const colorMap: Record<Toast['type'], string> = {
  success: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
  error: 'border-danger/50 bg-danger/10 text-danger',
  info: 'border-slate-500/40 bg-slate-500/10 text-slate-200'
};

const ToastItem: React.FC<{toast:Toast; onClose:()=>void}> = ({ toast, onClose }) => (
  <div className={`rounded-lg border px-4 py-3 text-sm shadow-md backdrop-blur flex items-start gap-2 ${colorMap[toast.type]}`}>
    <div className="flex-1 leading-snug">{toast.message}</div>
    <button onClick={onClose} className="text-xs opacity-70 hover:opacity-100">✕</button>
  </div>
);

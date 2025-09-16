import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useToasts } from '../ui/Toast';
import { useApp } from '../../context/AppContext';
import { apiUrl } from '../../config';

export const StreamPanel: React.FC = () => {
  const { streamOn, startStream, stopStream, config, stats, captureFrame, latencyMs, addZone } = useApp();
  const imgRef = useRef<HTMLImageElement|null>(null);
  const containerRef = useRef<HTMLDivElement|null>(null);
  const [isFs, setIsFs] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [points, setPoints] = useState<[number,number][]>([]);
  const [zoneName, setZoneName] = useState('');
  const { push } = useToasts();

  // When stream starts (or config changes camera/conf) update img src
  useEffect(()=>{
    if(streamOn && config){
  const url = apiUrl(`/stream?camera=${config.camera_index}&conf=${config.default_conf}&_=${Date.now()}`);
      if(imgRef.current) imgRef.current.src = url;
    } else if(!streamOn && imgRef.current){
      imgRef.current.src='';
    }
  },[streamOn, config?.camera_index, config?.default_conf]);

  const onStart = useCallback(()=>{ if(!config){ push({ type:'error', message:'Config not loaded yet'}); return;} startStream(); push({ type:'success', message:'Stream started'}); },[config,startStream,push]);
  const onStop = useCallback(()=>{ stopStream(); push({ type:'info', message:'Stream stopped'}); },[stopStream,push]);

  const onCapture = useCallback(async ()=>{
    const blob = await captureFrame(imgRef.current); if(!blob){ push({ type:'error', message:'Capture failed'}); return; }
    const a=document.createElement('a'); const ts=new Date().toISOString().replace(/[:.]/g,'-'); a.download=`frame-${ts}.png`; a.href=URL.createObjectURL(blob); a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000);
    push({ type:'success', message:'Frame captured'});
  },[captureFrame,push]);

  const toggleFs = useCallback(()=>{
    const el = containerRef.current; if(!el) return;
    if(!document.fullscreenElement){ el.requestFullscreen?.(); }
    else { document.exitFullscreen?.(); }
  },[]);

  useEffect(()=>{
    const handler = ()=> setIsFs(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handler);
    return ()=> document.removeEventListener('fullscreenchange', handler);
  },[]);

  const resetDrawing = ()=>{ setDrawing(false); setPoints([]); setZoneName(''); };

  const onCanvasClick = useCallback((e: React.MouseEvent<HTMLDivElement>)=>{
    if(!drawing || !imgRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * (imgRef.current.naturalWidth || imgRef.current.width);
    const y = ((e.clientY - rect.top) / rect.height) * (imgRef.current.naturalHeight || imgRef.current.height);
    setPoints(p=> [...p, [Math.round(x), Math.round(y)]]);
  },[drawing]);

  const saveZone = useCallback(async ()=>{
    if(points.length < 3){ return; }
    await addZone({ name: zoneName || `zone_${Date.now()}`, polygon: points as any, alert_on: ['NO-Hardhat','NO-Safety Vest','NO-Mask'] });
    resetDrawing();
  },[points, zoneName, addZone]);

  return (
    <div className="card p-5 space-y-5" ref={containerRef}>
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Live Stream</h2>
        <div className="flex flex-col items-end gap-2 text-xs">
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border ${streamOn? 'border-emerald-500/50 text-ok bg-emerald-500/10':'border-slate-600 text-slate-400'}`}>{streamOn? 'LIVE':'OFF'}</span>
            <button onClick={toggleFs} className="btn-outline text-xs" title="Fullscreen">{isFs? 'Exit FS':'Fullscreen'}</button>
            <button onClick={()=> setDrawing(d=> !d)} className={`btn-outline text-xs ${drawing? 'border-amber-400 text-amber-300':'text-xs'}`} title="Draw Zone">{drawing? 'Cancel Draw':'Draw Zone'}</button>
            <button onClick={onCapture} className="btn-outline text-xs">Capture</button>
            <button onClick={streamOn? onStop : onStart} className="btn-accent text-xs px-3">{streamOn? 'Stop':'Start'}</button>
          </div>
          <div className="flex gap-4 text-[10px] text-slate-500">
            <span>Frames: {stats?.frames ?? 0}</span>
            <span>Latency: {latencyMs? latencyMs.toFixed(0)+' ms':'-'}</span>
          </div>
        </div>
      </div>
      <div className="aspect-video w-full rounded-xl overflow-hidden border border-slate-700 bg-black flex items-center justify-center relative select-none" onClick={onCanvasClick}>
        <img ref={imgRef} alt="Live stream" className="w-full h-full object-contain pointer-events-none" />
        {drawing && (
          <>
            <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
              {points.length >= 2 && (
                <polyline points={points.map(p=> p.join(',')).join(' ')} fill="none" stroke="#fbbf24" strokeWidth={2} />
              )}
              {points.length >=3 && (
                <polygon points={points.map(p=> p.join(',')).join(' ')} fill="rgba(251,191,36,0.15)" stroke="#fbbf24" strokeWidth={2} />
              )}
              {points.map((p,i)=>(
                <circle key={i} cx={p[0]} cy={p[1]} r={4} fill="#fbbf24" stroke="#78350f" strokeWidth={1} />
              ))}
            </svg>
            <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur px-3 py-2 rounded border border-amber-500/40 flex flex-col gap-2 text-[10px] w-56">
              <div className="flex items-center justify-between">
                <span className="text-amber-300 font-semibold">Drawing Zone</span>
                <button onClick={resetDrawing} className="text-red-300 hover:text-red-200">✕</button>
              </div>
              <input value={zoneName} onChange={e=> setZoneName(e.target.value)} placeholder="Zone name" className="bg-slate-800/60 rounded px-2 py-1 text-xs focus:outline-none focus:ring focus:ring-amber-400/40" />
              <div className="flex items-center justify-between">
                <span>{points.length} pts</span>
                <div className="flex gap-2">
                  <button disabled={points.length<3} onClick={saveZone} className="btn-accent disabled:opacity-40 disabled:cursor-not-allowed text-[10px] px-2">Save</button>
                </div>
              </div>
              <p className="text-[10px] text-slate-400">Click on the video to add points. Need 3+ points.</p>
            </div>
          </>
        )}
        {!streamOn && <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-sm">Stream Off</div>}
      </div>
    </div>
  );
};

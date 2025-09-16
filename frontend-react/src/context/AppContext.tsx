import React, { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef } from 'react';
import { useToasts } from '../modules/ui/Toast';
import { apiUrl } from '../config';

// Types
export interface CameraInfo { index:number; width:number; height:number }
export interface Zone { name:string; polygon:[number,number][]; alert_on:string[]; min_conf:number }
export interface Detection { bbox:[number,number,number,number]; cls:number; cls_name:string; conf:number; track_id?:number }
export interface Alert { zone:string; cls_name:string; conf:number; track_id?:number }
export interface LiveStats { fps:number; detections:number; non_compliant:number; frames:number; alerts_last:Alert[]; recent_detections:Detection[]; last_frame_ts?:number|null; server_time?:number }
export interface Config { tracking:boolean; draw_zones:boolean; beep:boolean; default_conf:number; camera_index:number }

interface State {
  config: Config | null;
  cameras: CameraInfo[];
  zones: Zone[];
  stats: LiveStats | null;
  streamOn: boolean;
  loading: Record<string, boolean>;
  error?: string | null;
  fpsHistory: number[];
  detHistory: number[];
  nonCompHistory: number[];
  detectionHistory: Detection[];
}

const initialState: State = {
  config: null,
  cameras: [],
  zones: [],
  stats: null,
  streamOn: false,
  loading: {},
  error: null,
  fpsHistory: [],
  detHistory: [],
  nonCompHistory: [],
  detectionHistory: []
};

type Action =
  | { type:'SET_CONFIG'; config: Config }
  | { type:'SET_CAMERAS'; cameras: CameraInfo[] }
  | { type:'SET_ZONES'; zones: Zone[] }
  | { type:'SET_STATS'; stats: LiveStats }
  | { type:'SET_STREAM'; on: boolean }
  | { type:'SET_LOADING'; key:string; value:boolean }
  | { type:'SET_ERROR'; error:string|null }
  | { type:'APPEND_HISTORY'; fps:number; det:number; nonc:number; detections:Detection[] };

function reducer(state:State, action:Action): State {
  switch(action.type){
    case 'SET_CONFIG': return { ...state, config: action.config };
    case 'SET_CAMERAS': return { ...state, cameras: action.cameras };
    case 'SET_ZONES': return { ...state, zones: action.zones };
    case 'SET_STATS': return { ...state, stats: action.stats };
    case 'SET_STREAM': return { ...state, streamOn: action.on };
    case 'SET_LOADING': return { ...state, loading: { ...state.loading, [action.key]: action.value } };
    case 'SET_ERROR': return { ...state, error: action.error };
    case 'APPEND_HISTORY': {
      const maxPts = 120;
      const fpsHistory = [...state.fpsHistory, action.fps].slice(-maxPts);
      const detHistory = [...state.detHistory, action.det].slice(-maxPts);
      const nonCompHistory = [...state.nonCompHistory, action.nonc].slice(-maxPts);
      const detectionHistory = [...state.detectionHistory, ...action.detections].slice(-500);
      return { ...state, fpsHistory, detHistory, nonCompHistory, detectionHistory };
    }
    default: return state;
  }
}

interface Ctx extends State {
  refreshConfig():Promise<void>;
  refreshCameras():Promise<void>;
  refreshZones():Promise<void>;
  applyConfig(p: Partial<Config>):Promise<void>;
  addZone(z: Omit<Zone,'min_conf'> & { min_conf?:number }):Promise<void>;
  deleteZone(name:string):Promise<void>;
  startStream():void;
  stopStream():void;
  captureFrame(imgEl: HTMLImageElement | null):Promise<Blob|null>;
  latencyMs: number | null;
  exportDetectionsCSV(): void;
  exportDetectionsNDJSON(): void;
}

const AppContext = createContext<Ctx | undefined>(undefined);

export const AppProvider: React.FC<{children:React.ReactNode}> = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { push } = useToasts();
  const pollingRef = useRef<number | null>(null);
  const lastAlertIdsRef = useRef<string[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(()=>{ // simple tiny beep
    audioRef.current = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=");
  },[]);

  const setLoading = useCallback((key:string, value:boolean)=> dispatch({ type:'SET_LOADING', key, value }),[]);
  const setError = useCallback((error:string|null)=> dispatch({ type:'SET_ERROR', error }),[]);

  const refreshConfig = useCallback(async ()=>{
    setLoading('config', true); try {
  const r = await fetch(apiUrl('/api/config')); if(!r.ok) throw new Error('config'); dispatch({ type:'SET_CONFIG', config: await r.json() });
    } catch(e:any){ setError('Config load failed'); push({ type:'error', message:'Config load failed'}); } finally { setLoading('config', false);} },[setLoading,setError,push]);

  const refreshCameras = useCallback(async ()=>{
    setLoading('cameras', true); try {
  const r = await fetch(apiUrl('/api/cameras')); if(!r.ok) throw new Error('cameras'); const data = await r.json(); dispatch({ type:'SET_CAMERAS', cameras: data.cameras||[] });
    } catch(e:any){ setError('Camera load failed'); push({ type:'error', message:'Camera list failed'}); } finally { setLoading('cameras', false);} },[setLoading,setError,push]);

  const refreshZones = useCallback( async ()=>{
    setLoading('zones', true); try {
  const r = await fetch(apiUrl('/api/zones')); if(!r.ok) throw new Error('zones'); dispatch({ type:'SET_ZONES', zones: await r.json() });
    } catch(e:any){ setError('Zone load failed'); push({ type:'error', message:'Zones load failed'}); } finally { setLoading('zones', false);} },[setLoading,setError,push]);

  const applyConfig = useCallback(async (p: Partial<Config>)=>{
    setLoading('applyConfig', true); try {
  const r = await fetch(apiUrl('/api/config'),{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p) }); if(!r.ok) throw new Error('apply'); const cfg = await r.json(); dispatch({ type:'SET_CONFIG', config: cfg }); push({ type:'success', message:'Config updated'});
    } catch(e:any){ push({ type:'error', message:'Update failed'});} finally { setLoading('applyConfig', false);} },[push]);

  const addZone = useCallback(async (z: Omit<Zone,'min_conf'> & { min_conf?:number })=>{
    setLoading('addZone', true); try {
  const r = await fetch(apiUrl('/api/zones'),{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(z) }); if(!r.ok) throw new Error('add zone'); await refreshZones(); push({ type:'success', message:'Zone added'});
    } catch(e:any){ push({ type:'error', message:'Add zone failed'});} finally { setLoading('addZone', false);} },[refreshZones,push]);

  const deleteZone = useCallback(async (name:string)=>{
    setLoading('deleteZone', true); try {
  const r = await fetch(apiUrl(`/api/zones/${encodeURIComponent(name)}`),{ method:'DELETE' }); if(!r.ok) throw new Error('delete zone'); await refreshZones(); push({ type:'success', message:'Zone deleted'});
    } catch(e:any){ push({ type:'error', message:'Delete zone failed'});} finally { setLoading('deleteZone', false);} },[refreshZones,push]);

  const startStream = useCallback(()=>{ if(!state.config) return; dispatch({ type:'SET_STREAM', on:true }); },[state.config]);
  const stopStream = useCallback(()=>{ dispatch({ type:'SET_STREAM', on:false }); },[]);

  // Poll stats while streamOn
  useEffect(()=>{
    if(state.streamOn){
      const poll = async () => {
        try { const r = await fetch(apiUrl('/api/live_stats')); if(r.ok){ const data: LiveStats = await r.json();
            dispatch({ type:'SET_STATS', stats: data });
            dispatch({ type:'APPEND_HISTORY', fps: data.fps||0, det: data.detections||0, nonc: data.non_compliant||0, detections: data.recent_detections||[] });
            if(state.config?.beep && data.alerts_last?.length){
              const ids = data.alerts_last.map(a=>`${a.zone}:${a.cls_name}:${a.track_id??''}`);
              if(ids.some(id=> !lastAlertIdsRef.current.includes(id)) && audioRef.current){ audioRef.current.currentTime=0; audioRef.current.play().catch(()=>{}); }
              lastAlertIdsRef.current = ids.slice(-50);
            }
          } } catch { /* ignore */ }
      }; poll(); pollingRef.current = window.setInterval(poll, 750) as unknown as number; return ()=>{ if(pollingRef.current) window.clearInterval(pollingRef.current); };
    } else { if(pollingRef.current) window.clearInterval(pollingRef.current); }
  },[state.streamOn, state.config?.beep]);

  // Initial loads
  useEffect(()=>{ refreshConfig(); refreshCameras(); refreshZones(); },[refreshConfig,refreshCameras,refreshZones]);

  const captureFrame = useCallback(async (imgEl: HTMLImageElement | null)=>{
    if(!imgEl) return null; const canvas = document.createElement('canvas'); canvas.width = imgEl.naturalWidth||imgEl.width; canvas.height = imgEl.naturalHeight||imgEl.height; const ctx = canvas.getContext('2d'); if(!ctx) return null; ctx.drawImage(imgEl,0,0); return await new Promise<Blob|null>(res=> canvas.toBlob(res,'image/png')); },[]);

  const latencyMs = useMemo(()=>{
    if(!state.stats?.last_frame_ts || !state.stats?.server_time) return null; const ms = (state.stats.server_time - state.stats.last_frame_ts) * 1000; return ms < 0 ? null : ms; },[state.stats]);

  const exportDetectionsCSV = useCallback(()=>{
    const rows = [['cls_name','conf','x1','y1','x2','y2','track_id']];
    state.detectionHistory.forEach(d=> rows.push([d.cls_name, d.conf.toFixed(4), ...d.bbox.map(n=> String(Math.round(n))), d.track_id? String(d.track_id):''] as any));
    const csv = rows.map(r=> r.join(',')).join('\n');
    const blob = new Blob([csv], { type:'text/csv' }); const a=document.createElement('a'); a.download='detections.csv'; a.href=URL.createObjectURL(blob); a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),3000);
  },[state.detectionHistory]);

  const exportDetectionsNDJSON = useCallback(()=>{
    const lines = state.detectionHistory.map(d=> JSON.stringify(d));
    const blob = new Blob([lines.join('\n')+'\n'], { type:'application/x-ndjson' }); const a=document.createElement('a'); a.download='detections.ndjson'; a.href=URL.createObjectURL(blob); a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),3000);
  },[state.detectionHistory]);

  const value: Ctx = {
    ...state,
    refreshConfig,
    refreshCameras,
    refreshZones,
    applyConfig,
    addZone,
    deleteZone,
    startStream,
    stopStream,
    captureFrame,
    latencyMs,
    exportDetectionsCSV,
    exportDetectionsNDJSON
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export function useApp(){
  const ctx = useContext(AppContext); if(!ctx) throw new Error('AppContext missing'); return ctx;
}

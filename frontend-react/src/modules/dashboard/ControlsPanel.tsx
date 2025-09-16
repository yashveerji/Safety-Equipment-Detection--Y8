import React, { useEffect, useRef, useState } from 'react';
import { useToasts } from '../ui/Toast';
import { useApp } from '../../context/AppContext';
import { apiUrl } from '../../config';

export const ControlsPanel: React.FC = () => {
  const { cameras, refreshCameras, config, applyConfig } = useApp();
  const [localConf, setLocalConf] = useState({
    camera_index: config?.camera_index ?? 0,
    default_conf: config?.default_conf ?? 0.5,
    tracking: config?.tracking ?? true,
    draw_zones: config?.draw_zones ?? true,
    beep: config?.beep ?? false
  });
  const fileRef = useRef<HTMLInputElement|null>(null);
  const { push } = useToasts();

  useEffect(()=>{
    if(config){
      setLocalConf({
        camera_index: config.camera_index,
        default_conf: config.default_conf,
        tracking: config.tracking,
        draw_zones: config.draw_zones,
        beep: config.beep
      });
    }
  },[config]);

  function update<K extends keyof typeof localConf>(k:K, v:any){ setLocalConf(prev=> ({ ...prev, [k]: v })); }

  async function onApply(){ await applyConfig(localConf); }

  function onZoneFileChange(e:React.ChangeEvent<HTMLInputElement>){
    const f = e.target.files?.[0]; if(!f) return; const reader = new FileReader();
    reader.onload = async () => {
      try {
        const text = String(reader.result||'');
        const arr = JSON.parse(text);
        if(!Array.isArray(arr)) throw new Error('Root must be array');
        for(const z of arr){
          if(!z.name || !Array.isArray(z.points)) throw new Error('Invalid zone object');
          await fetch(apiUrl('/api/zones'),{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(z)});
        }
        push({ type:'success', message:`Uploaded ${arr.length} zones` });
      } catch(err:any){ push({ type:'error', message:'Zone upload error: '+err.message }); }
    };
    reader.readAsText(f);
  }

  return (
    <div className="card p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Controls</h2>
        <button onClick={refreshCameras} className="btn-outline text-xs">Reload Cams</button>
      </div>
      <div className="space-y-4">
        <div>
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 flex justify-between mb-1">Camera</label>
          <select value={localConf.camera_index} onChange={e=>update('camera_index', parseInt(e.target.value))} className="w-full bg-panelAlt border border-slate-700 rounded-lg px-3 py-2 text-sm">
            {cameras.map(c=> <option key={c.index} value={c.index}>Camera {c.index} ({c.width}x{c.height})</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 flex justify-between mb-1">Confidence <span className="text-accent font-semibold">{localConf.default_conf.toFixed(2)}</span></label>
          <input type="range" min={0.1} max={0.95} step={0.05} value={localConf.default_conf} onChange={e=>update('default_conf', parseFloat(e.target.value))} className="w-full" />
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <label className="flex items-center gap-2"><input type="checkbox" checked={localConf.tracking} onChange={e=>update('tracking', e.target.checked)} /> Tracking</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={localConf.draw_zones} onChange={e=>update('draw_zones', e.target.checked)} /> Draw Zones</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={localConf.beep} onChange={e=>update('beep', e.target.checked)} /> Beep Alerts</label>
        </div>
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-400 flex justify-between">Zone JSON <span className="text-[10px] text-slate-500">Array of objects</span></label>
          <input ref={fileRef} type="file" accept="application/json" onChange={onZoneFileChange} className="block w-full text-xs" />
          <pre className="text-[10px] bg-panelAlt p-2 rounded border border-slate-700 overflow-auto max-h-32">{`[
  {"name":"Zone A","points":[[100,120],[180,130],[140,200]],"alert_on":["NO-Hardhat"]}
]`}</pre>
        </div>
        <div className="flex flex-wrap gap-3">
          <button onClick={onApply} className="btn-accent">Apply</button>
        </div>
      </div>
    </div>
  );
};

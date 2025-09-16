import React from 'react';
import { useApp } from '../../context/AppContext';
import { apiUrl } from '../../config';
import { useToasts } from '../ui/Toast';

export const ZonesPanel: React.FC = () => {
  const { zones, deleteZone, refreshZones } = useApp();
  const { push } = useToasts();

  return (
    <div className="card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Zones</h2>
        <div className="flex gap-2">
          <button onClick={refreshZones} className="btn-outline text-xs">Refresh</button>
        </div>
      </div>
      <div className="max-h-56 overflow-auto space-y-2 text-xs">
        {zones.length===0 && <div className="text-slate-600">No zones</div>}
        {zones.map(z=> (
          <div key={z.name} className="flex items-center justify-between bg-panelAlt border border-slate-700/60 rounded px-2 py-1 gap-2">
            <div className="flex-1 truncate">
              <span className="font-semibold text-accent mr-2">{z.name}</span>
              <span className="text-slate-500">{z.polygon.length} pts</span>
            </div>
            <button onClick={async ()=>{ await deleteZone(z.name); push({ type:'info', message:`Deleted ${z.name}`}); }} className="text-xs text-red-400 hover:text-red-300">Delete</button>
          </div>
        ))}
      </div>
      <details className="text-[10px] text-slate-500">
        <summary className="cursor-pointer mb-2 text-slate-400">Zone JSON format</summary>
        <pre className="bg-panelAlt p-2 rounded border border-slate-700 overflow-auto">{`[
  {"name":"Zone A","polygon":[[x,y],[x,y],[x,y]],"alert_on":["NO-Hardhat"],"min_conf":0.4}
]`}</pre>
      </details>
    </div>
  );
};
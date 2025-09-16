import React from 'react';
import { useApp } from '../../context/AppContext';

export const AlertsList: React.FC = () => {
  const { stats } = useApp();
  const alerts = stats?.alerts_last || [];
  return (
    <div className="card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Recent Alerts</h2>
        <span className="text-xs text-slate-500">{alerts.length}</span>
      </div>
      <div className="max-h-56 overflow-auto space-y-2 text-xs">
        {alerts.length === 0 && <div className="text-slate-600">No alerts</div>}
        {alerts.map((a,i)=>(
          <div key={i} className="flex justify-between items-center bg-panelAlt rounded px-2 py-1 border border-slate-700/60">
            <span className="font-semibold text-accent">{a.cls_name}</span>
            <span className="text-slate-400">{a.zone}</span>
            <span className="text-slate-500">{(a.conf*100).toFixed(0)}%</span>
            {a.track_id && <span className="text-[10px] text-slate-500">ID {a.track_id}</span>}
          </div>
        ))}
      </div>
    </div>
  );
};
import React from 'react';
import { useApp } from '../../context/AppContext';
import { Sparkline } from '../ui/Sparkline';

export const StatsPanel: React.FC = () => {
  const { stats, latencyMs, fpsHistory, detHistory, nonCompHistory, exportDetectionsCSV, exportDetectionsNDJSON } = useApp();
  return (
    <div className="card p-5 space-y-4">
      <h2 className="panel-title">Live Stats</h2>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <Metric label="FPS" value={stats?.fps ?? 0} />
        <Metric label="Detections" value={stats?.detections ?? 0} />
        <Metric label="Non-Compliant" value={stats?.non_compliant ?? 0} />
        <Metric label="Frames" value={stats?.frames ?? 0} />
        <Metric label="Latency ms" value={latencyMs? latencyMs.toFixed(0):'-'} />
      </div>
      <div className="space-y-3">
  <div className="flex items-center justify-between text-[11px] text-slate-400"><span>FPS Trend</span><span>{fpsHistory.length ? fpsHistory[fpsHistory.length-1].toFixed(1) : '-'}</span></div>
        <Sparkline data={fpsHistory} />
  <div className="flex items-center justify-between text-[11px] text-slate-400"><span>Detections Trend</span><span>{detHistory.length ? detHistory[detHistory.length-1] : '-'}</span></div>
        <Sparkline data={detHistory} stroke="#6366f1" fill="rgba(99,102,241,0.15)" />
  <div className="flex items-center justify-between text-[11px] text-slate-400"><span>Non-Compliance Trend</span><span>{nonCompHistory.length ? nonCompHistory[nonCompHistory.length-1] : '-'}</span></div>
        <Sparkline data={nonCompHistory} stroke="#ef4444" fill="rgba(239,68,68,0.15)" />
      </div>
      <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-700/50">
        <button onClick={exportDetectionsCSV} className="btn-outline text-xs">Export CSV</button>
        <button onClick={exportDetectionsNDJSON} className="btn-outline text-xs">Export NDJSON</button>
      </div>
    </div>
  );
};

const Metric: React.FC<{label:string; value:number|string}> = ({ label, value }) => (
  <div className="metric">
    <h4>{label}</h4>
    <span>{value}</span>
  </div>
);

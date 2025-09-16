import React, { useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';

export const DetectionTable: React.FC = () => {
  const { stats } = useApp();
  const [query, setQuery] = useState('');
  const [nonCompOnly, setNonCompOnly] = useState(false);
  const detsRaw = stats?.recent_detections || [];
  const dets = useMemo(()=>{
    const q = query.trim().toLowerCase();
    return detsRaw.filter(d=>{
      if(q && !d.cls_name.toLowerCase().includes(q)) return false;
      if(nonCompOnly && !d.cls_name.startsWith('NO-')) return false;
      return true;
    });
  },[detsRaw, query, nonCompOnly]);
  return (
    <div className="card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Recent Detections</h2>
        <div className="flex items-center gap-2">
          <input value={query} onChange={e=> setQuery(e.target.value)} placeholder="Filter" className="bg-panelAlt border border-slate-700 text-xs rounded px-2 py-1 w-28 focus:outline-none focus:ring-1 focus:ring-accent" />
          <label className="flex items-center gap-1 text-[10px] select-none cursor-pointer text-slate-400">
            <input type="checkbox" checked={nonCompOnly} onChange={e=> setNonCompOnly(e.target.checked)} className="accent-accent" /> Non-Comp
          </label>
          <span className="text-xs text-slate-500">{dets.length}</span>
        </div>
      </div>
      <div className="overflow-auto max-h-64">
        <table className="w-full text-xs">
          <thead className="text-slate-400">
            <tr>
              <th className="text-left font-semibold py-1 pr-2">Class</th>
              <th className="text-left font-semibold py-1 pr-2">Conf</th>
              <th className="text-left font-semibold py-1 pr-2">BBox</th>
              <th className="text-left font-semibold py-1 pr-2">Track</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {dets.length===0 && <tr><td colSpan={4} className="py-2 text-slate-600">No detections</td></tr>}
            {dets.map((d,i)=>(
              <tr key={i} className="hover:bg-panelAlt/60">
                <td className="py-1 pr-2 font-medium text-slate-200">{d.cls_name}</td>
                <td className="py-1 pr-2 text-slate-400">{(d.conf*100).toFixed(1)}%</td>
                <td className="py-1 pr-2 text-slate-500">{d.bbox.map((n:number)=>Math.round(n)).join(',')}</td>
                <td className="py-1 pr-2 text-slate-500">{d.track_id ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
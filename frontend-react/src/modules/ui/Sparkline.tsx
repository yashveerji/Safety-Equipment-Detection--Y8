import React, { useMemo } from 'react';

interface SparklineProps { data:number[]; width?:number; height?:number; stroke?:string; fill?:string; }
export const Sparkline: React.FC<SparklineProps> = ({ data, width=120, height=28, stroke='#10b981', fill='rgba(16,185,129,0.15)' }) => {
  const path = useMemo(()=>{
    if(!data.length) return '';
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = max - min || 1;
    const step = width / Math.max(data.length-1,1);
    const pts = data.map((v,i)=> [i*step, height - ((v-min)/range)*height]);
    const d = pts.map((p,i)=> (i? 'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
    // area fill
    const area = d + ` L ${pts[pts.length-1][0].toFixed(1)},${height} L 0,${height} Z`;
    return { line:d, area };
  },[data,height,width]);
  return (
    <svg width={width} height={height} className="overflow-visible">
      {path && <path d={path.area} fill={fill} stroke="none"/>}
      {path && <path d={path.line} fill="none" stroke={stroke} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />}
    </svg>
  );
};
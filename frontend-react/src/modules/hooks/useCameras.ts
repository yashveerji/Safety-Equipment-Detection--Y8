import { useCallback, useEffect, useState } from 'react';

interface CameraInfo { index:number; width:number; height:number }

export function useCameras(){
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const refresh = useCallback(()=>{ fetch('/api/cameras').then(r=>r.ok?r.json():null).then(d=>{ if(d?.cameras) setCameras(d.cameras); }).catch(()=>{}); },[]);
  useEffect(()=>{ refresh(); },[refresh]);
  return { cameras, refresh };
}

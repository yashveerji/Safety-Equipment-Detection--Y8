import { useEffect, useRef } from 'react';

export function useInterval(cb:()=>void, ms:number){
  const ref = useRef(cb);
  ref.current = cb;
  useEffect(()=>{
    if(ms==null) return; let id = setInterval(()=> ref.current(), ms); return ()=> clearInterval(id);
  },[ms]);
}

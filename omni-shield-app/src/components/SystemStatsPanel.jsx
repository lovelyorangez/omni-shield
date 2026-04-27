import { useState, useEffect, useRef, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''

let _pollingActive = false

const Gauge = ({ label, pct, used, color }) => (
  <div style={{width:'100%'}}>
    <div style={{
      display:'flex', justifyContent:'space-between',
      alignItems:'baseline', marginBottom:4
    }}>
      <span style={{
        fontSize:11, fontWeight:500,
        color:'var(--color-text-secondary)',
        textTransform:'uppercase', letterSpacing:'0.05em'
      }}>
        {label}
      </span>
      <span style={{fontSize:11, color:'var(--color-text-tertiary)'}}>
        {used || ''}
      </span>
    </div>
    <div style={{height:5, borderRadius:3,
                 background:'var(--color-border-tertiary)',
                 overflow:'hidden', marginBottom:2}}>
      <div style={{
        height:'100%', borderRadius:3,
        width:`${Math.min(pct ?? 0, 100)}%`,
        background: (pct ?? 0) > 90 ? '#E24B4A' : color,
        transition:'width 0.7s ease'
      }}/>
    </div>
    <div style={{fontSize:12, fontWeight:600,
                 color:'var(--color-text-primary)'}}>
      {pct ?? 0}%
    </div>
  </div>
)

export default function SystemStatsPanel() {
  const [stats, setStats] = useState(null)
  const idRef = useRef(null)

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/system/stats`)
      if (!res.ok) return
      const data = await res.json()
      setStats(data)
    } catch(_) {}
  }, [])

  useEffect(() => {
    if (_pollingActive) return
    _pollingActive = true
    poll()
    idRef.current = setInterval(poll, 2000)
    return () => {
      clearInterval(idRef.current)
      idRef.current = null
      _pollingActive = false
    }
  }, [poll])

  if (!stats) return null

  return (
    <div style={{
      position:'fixed',
      top:'60px',
      right:'16px',
      width:'220px',
      zIndex:100,
      background:'var(--color-background-secondary)',
      border:'0.5px solid var(--color-border-tertiary)',
      borderRadius:12,
      padding:'12px 14px',
    }}>
      <div style={{
        fontSize:10, fontWeight:600,
        letterSpacing:'0.07em', textTransform:'uppercase',
        color:'var(--color-text-tertiary)',
        marginBottom:10
      }}>
        System resources
      </div>
      <div style={{display:'flex', flexDirection:'column', gap:10}}>
        <Gauge
          label={stats.gpu_name && stats.gpu_name !== 'N/A' ? stats.gpu_name : 'GPU VRAM'}
          pct={stats.gpu_pct}
          used={stats.gpu_total_mb > 0
            ? `${(stats.gpu_used_mb/1024).toFixed(1)}/${(stats.gpu_total_mb/1024).toFixed(1)}GB`
            : null}
          color="#7F77DD"
        />
        <Gauge
          label="CPU"
          pct={stats.cpu_pct}
          color="#1D9E75"
        />
        <Gauge
          label="RAM"
          pct={stats.ram_pct}
          used={stats.ram_used_gb != null
            ? `${stats.ram_used_gb}/${stats.ram_total_gb}GB`
            : null}
          color="#BA7517"
        />
      </div>
      <div style={{
        fontSize:9, color:'var(--color-text-tertiary)',
        textAlign:'right', marginTop:8
      }}>
        {new Date().toLocaleTimeString()}
      </div>
    </div>
  )
}

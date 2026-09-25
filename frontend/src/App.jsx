import { useEffect, useRef, useState } from 'react'
import { Activity, AlertTriangle, ArrowUpRight, Check, ChevronRight, CircleHelp, CloudUpload, Compass, Gauge, HardDrive, Moon, Play, RotateCcw, Route, ShieldCheck, Sun, Video, Wifi, X } from 'lucide-react'
import PresentationDashboard from './PresentationDashboard.jsx'

const API = 'http://127.0.0.1:8000/api'
const emptyStats = { total_unique_potholes: 0, current_lane_potholes: 0, adjacent_lane_potholes: 0, outside_road_potholes: 0, ignored_potholes: 0, lane_status: 'UNCERTAIN', lane_detection_confidence: 0, average_confidence: 0, lighting_condition: '—', speed_kmh: null, speed_label: 'N/A', driving_direction: 'Forward', high_risk: 0, medium_risk: 0, low_risk: 0, raw_yolo_detections: 0, detections: [], forward_alerts: [] }
const DEFAULT_DISTANCE_CALIBRATION = { cameraHeightM: 1.4, cameraPitchDeg: 8, verticalFovDeg: 50 }

function estimateAlertDistance(alert, timeSeconds, calibration) {
  const samples = alert?.distance_samples
  if (!samples?.length) return null
  const height = Number(calibration.cameraHeightM)
  const pitch = Number(calibration.cameraPitchDeg)
  const verticalFov = Number(calibration.verticalFovDeg)
  if (!(height >= 0.5 && height <= 4) || !(pitch >= 0 && pitch <= 30) || !(verticalFov >= 20 && verticalFov <= 100)) return null

  let contactY = samples[0].contact_y_fraction
  if (timeSeconds >= samples[samples.length - 1].time_seconds) contactY = samples[samples.length - 1].contact_y_fraction
  else if (timeSeconds > samples[0].time_seconds) {
    const rightIndex = samples.findIndex(sample => sample.time_seconds >= timeSeconds)
    const left = samples[Math.max(0, rightIndex - 1)]
    const right = samples[rightIndex]
    const span = right.time_seconds - left.time_seconds
    const portion = span > 0 ? (timeSeconds - left.time_seconds) / span : 0
    contactY = left.contact_y_fraction + (right.contact_y_fraction - left.contact_y_fraction) * portion
  }

  const fovRadians = verticalFov * Math.PI / 180
  const pitchRadians = pitch * Math.PI / 180
  const focalLengthInFrameHeights = 0.5 / Math.tan(fovRadians / 2)
  const rayAngle = pitchRadians + Math.atan((contactY - 0.5) / focalLengthInFrameHeights)
  if (!(rayAngle > 0 && rayAngle < Math.PI / 2)) return null
  const distance = height / Math.tan(rayAngle)
  return Number.isFinite(distance) && distance > 0 && distance <= 200 ? distance : null
}

function formatTime(value) {
  if (!Number.isFinite(value)) return '00:00'
  const seconds = Math.floor(value)
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

function App() {
  const [view, setView] = useState('presentation')
  const [job, setJob] = useState(null)
  const [stats, setStats] = useState(emptyStats)
  const [service, setService] = useState('checking')
  const [runtime, setRuntime] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [sourceName, setSourceName] = useState('')
  const inputRef = useRef(null)
  const videoRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [playbackTime, setPlaybackTime] = useState({ current: 0, duration: 0 })
  const [distanceCalibration, setDistanceCalibration] = useState(() => {
    try { return { ...DEFAULT_DISTANCE_CALIBRATION, ...JSON.parse(localStorage.getItem('roadwatch-distance-calibration') || '{}') } }
    catch { return DEFAULT_DISTANCE_CALIBRATION }
  })

  useEffect(() => { localStorage.setItem('roadwatch-distance-calibration', JSON.stringify(distanceCalibration)) }, [distanceCalibration])

  useEffect(() => { fetch(`${API}/health`).then(r => r.ok ? r.json() : Promise.reject()).then(x => { setService(x.model_available ? 'online' : 'model-missing'); setRuntime(x) }).catch(() => setService('offline')) }, [])
  useEffect(() => {
    if (!job?.job_id || !['queued', 'processing'].includes(job.status)) return
    let alive = true
    let timer
    const poll = async () => {
      if (!alive) return
      try {
        const response = await fetch(`${API}/status/${job.job_id}`)
        if (!response.ok) throw new Error('Could not read processing status.')
        const status = await response.json()
        if (!alive) return
        setJob(status)
        if (status.status === 'completed') {
          const result = await fetch(`${API}/results/${job.job_id}`).then(r => r.json())
          if (alive) { setStats(result); setBusy(false); setMessage('') }
        } else if (status.status === 'failed') { setBusy(false); setMessage(status.error || 'Video processing failed.') }
        else timer = setTimeout(poll, 1000)
      } catch (error) { if (alive) { setBusy(false); setMessage(`Backend unavailable: ${error.message}`) } }
    }
    timer = setTimeout(poll, 350)
    return () => { alive = false; clearTimeout(timer) }
  }, [job?.job_id])

  async function startDemo() {
    setMessage(''); setBusy(true); setJob(null); setStats(emptyStats); setPlaybackTime({ current: 0, duration: 0 }); setSourceName('assets/final.mp4')
    try { const response = await fetch(`${API}/process/demo`, { method: 'POST' }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Could not start demo processing.'); setJob({ ...data, progress: 0, message: 'Queued for processing' }) }
    catch (error) { setBusy(false); setMessage(`Could not start processing: ${error.message}`) }
  }
  async function upload(file) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.mp4')) { setMessage('Unsupported file. Choose an MP4 video.'); return }
    setMessage(''); setBusy(true); setJob(null); setStats(emptyStats); setPlaybackTime({ current: 0, duration: 0 }); setSourceName(file.name)
    try {
      const form = new FormData(); form.append('file', file)
      const response = await fetch(`${API}/process`, { method: 'POST', body: form }); const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Upload failed.')
      setJob({ ...data, progress: 0, message: 'Upload complete · queued for processing' })
    } catch (error) { setBusy(false); setMessage(error.message.includes('fetch') ? 'Backend unavailable. Start the FastAPI server and try again.' : error.message) }
  }
  const device = stats.inference_device ?? runtime?.inference_device
  const deviceLabel = device?.startsWith('cuda') ? 'GPU' : device === 'cpu' ? 'CPU' : 'CHECKING'
  const deviceName = stats.inference_device_name ?? runtime?.inference_device_name ?? 'Checking inference device'
  const complete = job?.status === 'completed'
  const alerts = complete ? stats.forward_alerts || [] : []
  const activeAlerts = alerts.filter(alert => playbackTime.current >= alert.start_seconds && playbackTime.current <= alert.end_seconds)
  const activeAlert = activeAlerts[0]
  const activeDistances = activeAlerts.map(alert => estimateAlertDistance(alert, playbackTime.current, distanceCalibration)).filter(Number.isFinite)
  const activeDistanceM = activeDistances.length ? Math.min(...activeDistances) : null
  const nextAlert = alerts.find(alert => alert.start_seconds > playbackTime.current)
  function jumpToAlert(alert) {
    const video = videoRef.current
    if (!video) return
    video.currentTime = alert.start_seconds
    setPlaybackTime(time => ({ ...time, current: alert.start_seconds }))
    video.play().catch(() => {})
  }
  const videoUrl = complete ? `${API}/video/${job.job_id}` : ''
  const statusLabel = service === 'online' ? 'SYSTEM ONLINE' : service === 'checking' ? 'CONNECTING' : service === 'model-missing' ? 'MODEL MISSING' : 'BACKEND OFFLINE'
  const laneConfidence = stats.lane_confidence_current ?? stats.lane_detection_confidence ?? 0
  const laneConfidenceClass = laneConfidence >= 0.7 ? 'risk-low' : laneConfidence >= 0.45 ? 'risk-medium' : 'risk-high'
  const laneStateClass = `lane-state-${String(stats.lane_status || 'UNCERTAIN').toLowerCase().replaceAll('_', '-')}`

  function handleVideoLoaded(event) {
    const duration = event.currentTarget.duration || 0
    const current = Math.min(playbackTime.current, duration)
    if (current > 0) event.currentTarget.currentTime = current
    setPlaybackTime({ current, duration })
  }

  if (view === 'presentation') return <PresentationDashboard
    stats={stats} job={job} busy={busy} complete={complete} sourceName={sourceName}
    activeAlerts={activeAlerts} playbackTime={playbackTime} activeDistanceM={activeDistanceM}
    distanceCalibration={distanceCalibration} onCalibrationChange={(key, value) => setDistanceCalibration(current => ({ ...current, [key]: Number(value) }))}
    videoUrl={videoUrl} videoRef={videoRef} onVideoLoaded={handleVideoLoaded}
    onTimeUpdate={event => setPlaybackTime({ current: event.currentTarget.currentTime, duration: event.currentTarget.duration || 0 })}
    onDemo={startDemo} onUpload={upload} onDetailedView={() => setView('detailed')}
    message={message} onDismissMessage={() => setMessage('')}
  />

  return <div className="app-shell" onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false) }} onDrop={e => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files?.[0]) }}>
    {dragging && <div className="drop-cover"><CloudUpload size={42}/><b>Drop MP4 to begin analysis</b><span>Maximum file size 500 MB</span></div>}
    <header className="topbar"><a className="brand" href="#"><span className="brand-icon"><Route size={20}/></span><span>ROADWATCH<small>VISION SYSTEMS</small></span></a><div className="topbar-center"><span className="live-dot"/>POTHOLE INTELLIGENCE <span className="separator">/</span> DASHBOARD</div><div className="topbar-actions"><button className="detailed-view-switch" onClick={() => setView('presentation')}>Presentation view <ArrowUpRight size={14}/></button><div className={`system-state ${service}`}><span className="state-light"/>{statusLabel}</div></div></header>
    <main className="content">
      <div className="page-heading"><div><div className="eyebrow"><span>ROAD SAFETY</span><ChevronRight size={13}/><span>COMPUTER VISION</span></div><h1>Pothole detection <span>&amp; lane intelligence</span></h1><p className="subtitle">Video analysis focused on hazards ahead in the vehicle’s current lane.</p></div><div className="session"><div className="session-label">ANALYSIS SESSION</div><div><span className="session-dot"/>{complete ? 'COMPLETE' : busy ? 'PROCESSING' : 'READY'} <span className="muted" title={deviceName}>&middot; LOCAL {deviceLabel}</span></div></div></div>

      <div className="workspace">
        <section className="main-column">
          <div className={`panel video-panel ${activeAlert ? 'has-forward-alert' : ''}`}>
            <div className="panel-top"><div><div className="section-kicker"><span className="kicker-bar"/>VIDEO ANALYSIS</div><h2>Road feed</h2></div><div className="feed-meta"><span className="rec-dot"/>{complete ? 'PROCESSED' : busy ? 'ANALYZING' : 'AWAITING INPUT'}<span className="meta-divider"/>{complete && stats.fps ? `${Math.round(stats.fps)} FPS` : 'MP4 INPUT'}</div></div>
            <div className={`forward-alert-banner ${activeAlert ? 'is-active' : ''}`} role="status" aria-live="polite">
              <div className="forward-alert-symbol">{activeAlert ? <AlertTriangle size={21}/> : <ShieldCheck size={20}/>}</div>
              <div className="forward-alert-copy"><span className="forward-alert-kicker">{activeAlert ? 'EARLY ROAD WARNING' : 'FORWARD HAZARD MONITOR'}</span><strong>{activeAlert ? activeAlerts.length > 1 ? `${activeAlerts.length} potholes ahead in your lane` : 'Pothole ahead in your lane' : busy ? 'Scanning the road ahead' : complete ? nextAlert ? 'Road clear at this moment' : 'No upcoming far-ahead alerts' : 'Ready to scan the road ahead'}</strong><small>{activeAlert ? `${activeAlerts.length > 1 ? `Tracked objects ${activeAlerts.map(alert => `#${alert.object_id}`).join(', ')}` : `Tracked object #${activeAlert.object_id}`} · ${activeDistanceM == null ? 'distance estimate unavailable' : `estimated distance ${activeDistanceM.toFixed(1)} m`} · approximate` : busy ? 'Alerts appear after the video finishes processing.' : complete && nextAlert ? `Next early alert at ${formatTime(nextAlert.start_seconds)}` : 'Distance is an adjustable camera-based estimate.'}</small></div>
              <span className={`forward-alert-status ${activeAlert ? 'warning' : ''}`}>{activeAlert ? 'POTHOLE AHEAD' : 'MONITORING'}</span>
              {!activeAlert && nextAlert && <button className="forward-alert-skip" onClick={() => jumpToAlert(nextAlert)} title="Jump to next early alert"><ChevronRight size={16}/></button>}
            </div>
            <div className={`video-stage ${!complete ? 'empty-stage' : ''}`}>
              {complete ? <video ref={videoRef} key={videoUrl} src={videoUrl} controls playsInline preload="metadata" onLoadedMetadata={handleVideoLoaded} onTimeUpdate={e => setPlaybackTime({ current: e.currentTarget.currentTime, duration: e.currentTarget.duration || 0 })} onRateChange={e => setPlaybackRate(e.currentTarget.playbackRate)} /> : <div className="stage-empty"><div className="road-glyph"><Route size={36}/></div><div className="stage-title">{busy ? 'Analyzing road feed' : 'No video loaded'}</div><div className="stage-copy">{busy ? (job?.message || 'Initializing the vision pipeline…') : 'Load a road video to begin lane-aware pothole analysis.'}</div>{busy && <div className="progress-track"><span style={{ width: `${job?.progress || 3}%` }}/></div>}<div className="stage-mark">{busy ? `${job?.progress || 0}%` : 'MP4 · UP TO 500 MB'}</div></div>}
              {complete && <div className="video-overlay-tag"><span className="live-dot"/>ANNOTATED FEED <span>·</span> CURRENT LANE</div>}
            </div>
            {complete && <div className="playback-extras"><button className="player-tool" onClick={() => { if (videoRef.current) { videoRef.current.pause(); videoRef.current.currentTime = 0; setPlaybackTime(t => ({ ...t, current: 0 })) } }} title="Restart from beginning"><RotateCcw size={14}/> Restart</button><span className="player-time">{formatTime(playbackTime.current)} / {formatTime(playbackTime.duration)}</span><label className="rate-tool">Speed<select aria-label="Playback speed" value={playbackRate} onChange={e => { const rate = Number(e.target.value); setPlaybackRate(rate); if (videoRef.current) videoRef.current.playbackRate = rate }}><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1">1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label></div>}
            <div className="video-footer"><div className="clip-name"><span className="file-icon"><Video size={14}/></span><div><b>{sourceName || 'Select a video source'}</b><small>{complete ? `${stats.frame_count || '—'} FRAMES PROCESSED` : 'LOCAL VIDEO INPUT'}</small></div></div><div className="source-actions"><button className="button secondary" onClick={() => inputRef.current?.click()} disabled={busy}><CloudUpload size={15}/> Upload video</button><button className="button primary" onClick={startDemo} disabled={busy}><Play size={14} fill="currentColor"/>{busy ? 'Processing…' : 'Run demo'}</button><input ref={inputRef} type="file" accept="video/mp4,.mp4" hidden onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; upload(file) }}/></div></div>
          </div>

          <div className="panel detections-panel"><div className="panel-top table-heading"><div><div className="section-kicker"><span className="kicker-bar"/>DETECTION LOG</div><h2>Current lane events <span className="count-chip">{stats.detections?.length || 0}</span></h2></div><div className="table-caption">UNIQUE TRACKED OBJECTS</div></div>
            <div className="table-wrap"><table><thead><tr><th>OBJECT</th><th>FRAME</th><th>CONFIDENCE</th><th>SEVERITY</th><th>LANE</th><th>POSITION</th></tr></thead><tbody>{stats.detections?.length ? stats.detections.map(item => <tr key={item.id}><td><span className="object-id">#{String(item.id).padStart(2, '0')}</span> Pothole</td><td className="mono">{item.frame}</td><td><div className="confidence-cell"><span>{Math.round(item.confidence * 100)}%</span><i><b style={{ width: `${item.confidence * 100}%` }}/></i></div></td><td><span className={`severity ${item.severity.toLowerCase()}`}><i/>{item.severity}</span></td><td><span className="lane-label"><Check size={12}/>Current lane</span></td><td className="muted">{item.position}</td></tr>) : <tr><td colSpan="6" className="empty-row">{busy ? 'Building the event log as detections are tracked…' : 'No events to display. Run the demo or upload an MP4.'}</td></tr>}</tbody></table></div>
            {complete && <div className="table-summary"><span><Activity size={13}/> {stats.raw_yolo_detections || 0} raw YOLO detections</span><span>{stats.current_lane_potholes || 0} unique ego-lane events</span></div>}
          </div>
        </section>

        <aside className="sidebar">
          <div className="sidebar-title"><div><div className="section-kicker"><span className="kicker-bar"/>OVERVIEW</div><h2>Detection stats</h2></div><button className="help-button" title="Severity is a visual estimate, not measured depth"><CircleHelp size={17}/></button></div>
          <div className={`forward-alert-card ${activeAlert ? 'is-active' : ''}`}><div className="forward-alert-card-head"><span><AlertTriangle size={15}/> EARLY ALERTS</span><b>{alerts.length}</b></div><p>{alerts.length ? 'Potholes spotted farther ahead in the current lane.' : complete ? 'No repeated far-ahead potholes were found.' : 'Alert moments will appear after analysis.'}</p>{alerts.length > 0 && <div className="forward-alert-list">{alerts.map((alert, index) => <button key={`${alert.object_id}-${alert.first_frame}-${index}`} className={activeAlert === alert ? 'selected' : ''} onClick={() => jumpToAlert(alert)}><span>#{alert.object_id} POTHOLE AHEAD</span><time>{formatTime(alert.start_seconds)}</time><ChevronRight size={13}/></button>)}</div>}</div>
          <div className="stat-card featured"><div className="stat-top"><span>TOTAL POTHOLES</span><span className="stat-icon"><AlertTriangle size={16}/></span></div><div className="stat-value">{stats.total_unique_potholes}<small> unique</small></div><div className="stat-foot"><span className="accent-text">{stats.raw_yolo_detections}</span> raw detections across video</div></div>
          <div className="lane-card"><div className="lane-card-head"><span className="lane-symbol"><Route size={17}/></span><span>EGO LANE STATUS</span><span className={`active-pill ${laneStateClass}`}><i/>{stats.lane_status}</span></div><div className="lane-card-body"><div className="lane-count">{stats.current_lane_potholes}<small>relevant events · lane {stats.ego_lane ?? '—'}</small></div><div className="lane-visual"><span className="lane-edge left"/><span className="lane-dash"/><span className="lane-edge right"/><span className="lane-car">⌃</span></div></div><div className="lane-risk"><span>LANE DETECTION CONFIDENCE</span><b className={laneConfidenceClass}><i/>{Math.round(laneConfidence * 100)}%</b></div></div>
          <div className="vehicle-status-card"><div className="vehicle-status-head"><span className="section-kicker"><span className="kicker-bar"/>VEHICLE STATUS</span><span className="status-note">VIDEO TELEMETRY</span></div><div className="vehicle-speed"><div><span>SPEED</span><b>{stats.speed_kmh == null ? (stats.speed_label || 'N/A') : `${stats.speed_kmh} km/h`}</b></div><small>{stats.speed_kmh == null ? 'No calibrated speed source' : 'Estimated speed'}</small></div><div className="vehicle-status-pair"><div><span>{stats.lighting_condition === 'NIGHT' ? <Moon size={13}/> : <Sun size={13}/>}LIGHTING</span><b className={stats.lighting_condition === 'NIGHT' ? 'night-state' : 'day-state'}>{stats.lighting_condition || '—'}</b></div><div><span><Compass size={13}/>DIRECTION</span><b>{stats.driving_direction || 'Forward'}</b></div></div></div>
          <div className="risk-grid"><div className="risk-card"><span className="risk-label"><i className="risk-dot high-dot"/>HIGH RISK</span><b>{stats.high_risk}</b><small>visual estimate</small></div><div className="risk-card"><span className="risk-label"><i className="risk-dot medium-dot"/>MEDIUM RISK</span><b>{stats.medium_risk}</b><small>visual estimate</small></div><div className="risk-card"><span className="risk-label"><i className="risk-dot low-dot"/>LOW RISK</span><b>{stats.low_risk}</b><small>visual estimate</small></div><div className="risk-card"><span className="risk-label"><Gauge size={13}/> YOLO CONFIDENCE</span><b>{Math.round(stats.average_confidence * 100)}<small>%</small></b><small>ego lane events</small></div></div>
          <div className="depth-note">Physical pothole depth <b>Not available</b></div>
          <div className="ignored-card"><div className="ignored-icon"><ShieldCheck size={18}/></div><div className="ignored-info"><span>ADJACENT LANE · IGNORED</span><b>{stats.adjacent_lane_potholes} <small>unique</small></b></div><span className="ignored-check"><Check size={15}/></span></div>
          <div className="ignored-card"><div className="ignored-icon"><AlertTriangle size={18}/></div><div className="ignored-info"><span>OUTSIDE ROAD · IGNORED</span><b>{stats.outside_road_potholes} <small>unique</small></b></div><span className="ignored-check"><Check size={15}/></span></div>
          <div className="pipeline-card"><div className="pipeline-head"><span className="section-kicker"><span className="kicker-bar"/>PIPELINE</span><span className="pipeline-ok"><Wifi size={12}/> READY</span></div><div className="pipeline-steps"><span><i className="step-done"><Check size={9}/></i>YOLOv8 <small>·</small> <em>best.pt</em></span><ArrowUpRight size={12}/><span><i className="step-done"><Check size={9}/></i>LANE CV</span><ArrowUpRight size={12}/><span><i className="step-done"><Check size={9}/></i>TRACKING</span></div><div className="pipeline-foot" title={deviceName}><HardDrive size={12}/> LOCAL INFERENCE <span>·</span> {deviceLabel}</div></div>
        </aside>
      </div>
      {message && <div className="toast"><AlertTriangle size={16}/><span>{message}</span><button onClick={() => setMessage('')}><X size={15}/></button></div>}
    </main>
    <footer className="footer"><span>ROADWATCH <i>VISION SYSTEMS</i></span><span>LOCAL PROCESSING <b>·</b> NO CLOUD UPLOAD</span><span>CV PROJECT <b>01</b></span></footer>
  </div>
}

export default App

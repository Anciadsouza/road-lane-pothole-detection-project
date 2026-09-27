import { useRef, useState } from 'react'
import { AlertTriangle, ArrowRight, Check, CloudUpload, Film, LayoutDashboard, Play, X } from 'lucide-react'
import './presentation.css'

function formatTime(value) {
  if (!Number.isFinite(value)) return '00:00'
  const seconds = Math.floor(value)
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

export default function PresentationDashboard({ stats, job, busy, complete, sourceName, activeAlerts, playbackTime, activeDistanceM, distanceCalibration, onCalibrationChange, videoUrl, videoRef, onTimeUpdate, onVideoLoaded, onDemo, onUpload, onDetailedView, message, onDismissMessage }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const duration = playbackTime.duration || (stats.fps ? stats.frame_count / stats.fps : 0)
  const currentLane = stats.current_lane_potholes || 0
  const outsideLane = (stats.adjacent_lane_potholes || 0) + (stats.outside_road_potholes || 0)
  const active = activeAlerts.length > 0

  return <div className="simple-dashboard" onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false) }} onDrop={event => { event.preventDefault(); setDragging(false); onUpload(event.dataTransfer.files?.[0]) }}>
    {dragging && <div className="simple-drop"><CloudUpload size={34}/><strong>Drop your video to analyze it</strong></div>}
    <main className={`simple-main ${complete ? 'has-result' : ''}`}>
      {!complete ? <section className="simple-start" aria-label="Upload road video">
        <span className="simple-eyebrow">ROAD VIDEO ANALYSIS</span>
        <h1>See what’s ahead.</h1>
        <p>Upload a road video to see potholes in your lane and the moments they appear.</p>
        <div className={`simple-upload-card ${busy ? 'is-busy' : ''}`}>
          <span className="simple-upload-icon">{busy ? <Film size={27}/> : <CloudUpload size={27}/>}</span>
          <strong>{busy ? 'Analyzing your video' : 'Upload an MP4 video'}</strong>
          <small>{busy ? job?.message || 'Finding the moments that matter…' : 'Choose a file or drag it anywhere on this page'}</small>
          {busy ? <><div className="simple-progress"><span style={{ width: `${job?.progress || 3}%` }}/></div><span className="simple-progress-label">{job?.progress || 0}% complete</span></> : <button className="simple-main-button" onClick={() => inputRef.current?.click()}><CloudUpload size={16}/> Choose video</button>}
        </div>
        {!busy && <button className="simple-demo-link" onClick={onDemo}><Play size={15} fill="currentColor"/> Or try the demo <ArrowRight size={15}/></button>}
        <span className="simple-local-note"><Check size={14}/> Processed locally on your device</span>
        <button className="simple-detail-link" onClick={onDetailedView}><LayoutDashboard size={15}/> Detailed dashboard</button>
      </section> : <section className="simple-result" aria-label="Video results">
        <div className="simple-result-heading"><div><span className="simple-eyebrow">ANALYSIS COMPLETE</span><h1>Your road, at a glance.</h1><p>{sourceName || 'Road video'} <span>·</span> {formatTime(duration)} video</p></div><button className="simple-reupload" onClick={() => inputRef.current?.click()}><CloudUpload size={16}/> New video</button></div>
        <div className="simple-result-grid">
          <div className="simple-video-card"><div className="simple-video-top"><span><Film size={16}/> Road footage</span><span>{formatTime(playbackTime.current)} / {formatTime(duration)}</span></div><div className="simple-video-wrap"><video ref={videoRef} key={videoUrl} src={videoUrl} controls playsInline preload="metadata" onLoadedMetadata={onVideoLoaded} onPlay={onVideoPlay} onTimeUpdate={onTimeUpdate}/>{active && <div className="simple-video-warning"><AlertTriangle size={17}/><span><b>{activeAlerts.length > 1 ? `${activeAlerts.length} potholes ahead in your lane` : 'POTHOLE AHEAD IN YOUR LANE'}</b><small>{activeDistanceM == null ? 'Distance estimate unavailable' : `Distance: ${activeDistanceM.toFixed(1)} m`} · approximate</small></span></div>}</div></div>
          <aside className="simple-summary"><div className="simple-summary-top"><span>RESULTS</span><span className="simple-complete-dot">COMPLETE</span></div><div className="simple-summary-main"><span>IN YOUR LANE</span><strong>{currentLane}</strong><p>{currentLane === 1 ? 'pothole found' : 'potholes found'}</p></div><div className="simple-summary-row"><span>Total detected</span><strong>{stats.total_unique_potholes || 0}</strong></div><div className="simple-summary-row"><span>Outside your lane</span><strong>{outsideLane}</strong></div></aside>
        </div>
        <div className="simple-result-footer"><p className="simple-footnote">Meter values are approximate camera-based estimates and assume a flat road.</p><button className="simple-detail-link" onClick={onDetailedView}><LayoutDashboard size={15}/> Detailed dashboard</button></div>
        <details className="distance-settings"><summary>Adjust distance estimate for this camera</summary><div className="distance-setting-fields"><label>Camera height <span><input type="number" min="0.5" max="4" step="0.1" value={distanceCalibration.cameraHeightM} onChange={event => onCalibrationChange('cameraHeightM', event.target.value)}/> m</span></label><label>Camera pitch down <span><input type="number" min="0" max="30" step="0.5" value={distanceCalibration.cameraPitchDeg} onChange={event => onCalibrationChange('cameraPitchDeg', event.target.value)}/> °</span></label><label>Vertical field of view <span><input type="number" min="20" max="100" step="1" value={distanceCalibration.verticalFovDeg} onChange={event => onCalibrationChange('verticalFovDeg', event.target.value)}/> °</span></label></div><p>Estimate uses pothole position in the frame and a flat-road camera model. Adjust these values for your camera; results are approximate, not measured.</p></details>
      </section>}
      <input ref={inputRef} type="file" accept="video/mp4,.mp4" hidden onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; onUpload(file) }}/>
    </main>
    {message && <div className="simple-toast" role="alert"><AlertTriangle size={16}/><span>{message}</span><button onClick={onDismissMessage} aria-label="Dismiss message"><X size={16}/></button></div>}
  </div>
}

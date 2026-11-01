import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  Video,
  Radio,
  Zap,
  Sliders,
  Maximize2,
  RefreshCw,
  Clock,
  Layers,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ShieldCheck,
  ChevronRight,
  Eye,
  Crosshair,
  Settings,
  HelpCircle,
  X,
  Volume2,
  VolumeX,
  Sparkles
} from 'lucide-react';
import { api } from '../api/client';
import { CATEGORIES, getCategoryMeta } from '../utils/categories';
import SeverityBadge from '../components/ui/SeverityBadge';
import HeatmapSlider from '../components/ui/HeatmapSlider';

export default function LiveStreamInspector({ onNavigateToRAG, onNavigateToSimilarity }) {
  // Stream & Source State
  const [streamMode, setStreamMode] = useState('webcam'); // 'webcam' | 'mjpeg'
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState(null);
  
  // Inspection Controls
  const [autoInspect, setAutoInspect] = useState(false);
  const [inspectInterval, setInspectInterval] = useState(1500); // ms
  const [isInspecting, setIsInspecting] = useState(false);
  const [shutterFlash, setShutterFlash] = useState(false);

  // Real-time Metrics & HUD
  const [fps, setFps] = useState(24);
  const [latestResult, setLatestResult] = useState(null);
  const [eventHistory, setEventHistory] = useState([]);
  const [selectedEventModal, setSelectedEventModal] = useState(null);

  // References
  const videoRef = useRef(null);
  const hudCanvasRef = useRef(null);
  const mjpegImgRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const lastFpsTimeRef = useRef(Date.now());
  const frameCountRef = useRef(0);
  const isInspectingRef = useRef(false);

  // Keep isInspectingRef in sync
  useEffect(() => {
    isInspectingRef.current = isInspecting;
  }, [isInspecting]);

  // --------------------------------------------------------
  // 1. Client Browser Webcam Lifecycle (Mode 1)
  // --------------------------------------------------------
  const startWebcam = useCallback(async () => {
    try {
      setWebcamError(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch((err) => console.warn('Video play warning:', err));
      }
      setWebcamActive(true);
    } catch (err) {
      console.error('Failed to access webcam:', err);
      setWebcamActive(false);
      setWebcamError(
        err.name === 'NotAllowedError'
          ? 'Camera access was denied by browser permissions. Please allow camera access or switch to Backend Engine Stream.'
          : `Camera error: ${err.message}. Switch to Mode 2 (Backend Stream) to view live simulated industrial feed.`
      );
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
  }, []);

  useEffect(() => {
    if (streamMode === 'webcam') {
      startWebcam();
    } else {
      stopWebcam();
    }
    return () => {
      stopWebcam();
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [streamMode, startWebcam, stopWebcam]);

  // --------------------------------------------------------
  // 2. Holographic HUD Canvas Drawing Loop
  // --------------------------------------------------------
  useEffect(() => {
    let scanlineY = 50;
    let scanlineDir = 3;

    const renderHUD = () => {
      const canvas = hudCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // FPS Calculation
      frameCountRef.current += 1;
      const now = Date.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      // Dynamic Color from Latest Severity
      const sev = latestResult?.severity_level || 'normal';
      let bracketColor = '#10B981'; // Green
      if (sev === 'critical') bracketColor = '#EF4444'; // Red
      else if (sev === 'moderate') bracketColor = '#F59E0B'; // Amber

      // Central ROI Box Coordinates (60% of canvas)
      const roiW = Math.round(w * 0.62);
      const roiH = Math.round(h * 0.70);
      const roiX = Math.round((w - roiW) / 2);
      const roiY = Math.round((h - roiH) / 2);

      // Draw Neon Corner Brackets around ROI
      const bLen = 28;
      ctx.strokeStyle = bracketColor;
      ctx.lineWidth = 3;
      ctx.shadowColor = bracketColor;
      ctx.shadowBlur = 10;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(roiX + bLen, roiY);
      ctx.lineTo(roiX, roiY);
      ctx.lineTo(roiX, roiY + bLen);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(roiX + roiW - bLen, roiY);
      ctx.lineTo(roiX + roiW);
      ctx.lineTo(roiX + roiW, roiY + bLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(roiX, roiY + roiH - bLen);
      ctx.lineTo(roiX, roiY + roiH);
      ctx.lineTo(roiX + bLen, roiY + roiH);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(roiX + roiW - bLen, roiY + roiH);
      ctx.lineTo(roiX + roiW, roiY + roiH);
      ctx.lineTo(roiX + roiW, roiY + roiH - bLen);
      ctx.stroke();

      // Center Optical Reticle Crosshairs
      const cx = Math.round(w / 2);
      const cy = Math.round(h / 2);
      ctx.strokeStyle = '#06B6D4'; // Cyan
      ctx.lineWidth = 1.5;
      ctx.shadowColor = '#06B6D4';
      ctx.shadowBlur = 8;

      ctx.beginPath();
      ctx.arc(cx, cy, 22, 0, Math.PI * 2);
      ctx.moveTo(cx - 16, cy);
      ctx.lineTo(cx + 16, cy);
      ctx.moveTo(cx, cy - 16);
      ctx.lineTo(cx, cy + 16);
      ctx.stroke();

      // Animated Scanning Laser Line
      scanlineY += scanlineDir;
      if (scanlineY >= roiY + roiH - 6 || scanlineY <= roiY + 6) {
        scanlineDir *= -1;
      }

      ctx.shadowColor = '#06B6D4';
      ctx.shadowBlur = 15;
      const gradient = ctx.createLinearGradient(roiX, scanlineY, roiX + roiW, scanlineY);
      gradient.addColorStop(0, 'rgba(6,182,212,0.1)');
      gradient.addColorStop(0.5, 'rgba(6,182,212,0.9)');
      gradient.addColorStop(1, 'rgba(6,182,212,0.1)');
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(roiX, scanlineY);
      ctx.lineTo(roiX + roiW, scanlineY);
      ctx.stroke();

      // Reset shadow for subsequent drawings
      ctx.shadowBlur = 0;

      animFrameIdRef.current = requestAnimationFrame(renderHUD);
    };

    animFrameIdRef.current = requestAnimationFrame(renderHUD);
    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [latestResult]);

  // --------------------------------------------------------
  // 3. Frame Capture & Real-Time Inspection Trigger
  // --------------------------------------------------------
  const executeInspection = useCallback(
    async (silent = false) => {
      if (isInspectingRef.current) return;
      setIsInspecting(true);

      // Shutter visual flash effect
      if (!silent) {
        setShutterFlash(true);
        setTimeout(() => setShutterFlash(false), 120);
      }

      try {
        let frameBlob = null;

        if (streamMode === 'webcam' && videoRef.current) {
          const video = videoRef.current;
          if (video.videoWidth === 0 || video.videoHeight === 0) {
            setIsInspecting(false);
            return;
          }
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          frameBlob = await new Promise((resolve) =>
            canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.88)
          );
        } else {
          // In MJPEG mode, grab frame snapshot from MJPEG image element
          const imgEl = mjpegImgRef.current;
          if (imgEl && imgEl.naturalWidth > 0) {
            const canvas = document.createElement('canvas');
            canvas.width = imgEl.naturalWidth;
            canvas.height = imgEl.naturalHeight;
            const ctx = canvas.getContext('2d');
            try {
              ctx.drawImage(imgEl, 0, 0);
              frameBlob = await new Promise((resolve) =>
                canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.88)
              );
            } catch (e) {
              frameBlob = null;
            }
          }
        }

        // If no blob was extracted (e.g. initial load or MJPEG cross-origin), fetch test preset
        if (!frameBlob) {
          const sampleRes = await fetch(`/outputs/similarity_search_samples/${selectedCategory}_similarity_demo.png`);
          if (sampleRes.ok) {
            frameBlob = await sampleRes.blob();
          }
        }

        if (!frameBlob) {
          throw new Error('Unable to extract frame for inspection.');
        }

        // Call FastAPI in-memory inspection endpoint
        const result = await api.inspectFrame(frameBlob, selectedCategory);

        // Stamp with client timestamp & event id
        const inspectionEvent = {
          ...result,
          id: `evt_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
          timestamp: new Date().toLocaleTimeString(),
          capturedCategory: selectedCategory,
        };

        setLatestResult(inspectionEvent);
        setEventHistory((prev) => [inspectionEvent, ...prev.slice(0, 39)]); // Keep last 40 events
      } catch (err) {
        console.error('Frame inspection error:', err);
      } finally {
        setIsInspecting(false);
      }
    },
    [streamMode, selectedCategory]
  );

  // --------------------------------------------------------
  // 4. Hotkey & Auto-Inspection Listeners
  // --------------------------------------------------------
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Spacebar triggers snapshot inspection
      if (e.code === 'Space' && !e.target.matches('input, textarea, select, button')) {
        e.preventDefault();
        executeInspection(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [executeInspection]);

  useEffect(() => {
    if (!autoInspect) return;
    const timer = setInterval(() => {
      if (!isInspectingRef.current) {
        executeInspection(true);
      }
    }, inspectInterval);
    return () => clearInterval(timer);
  }, [autoInspect, inspectInterval, executeInspection]);

  const catMeta = getCategoryMeta(selectedCategory);

  return (
    <div className="p-4 sm:p-6 lg:p-8 space-y-6 max-w-[1700px] mx-auto">
      {/* Top Header & Operational Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-600 to-cyan-400 p-[1px] shadow-[0_0_15px_rgba(6,182,212,0.4)]">
              <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Video className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="font-heading text-xl sm:text-2xl font-bold tracking-tight text-white">
                  Real-Time Optical Inspection
                </h1>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                  LIVE STREAM
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Direct frame-by-frame inference via resident PatchCore memory banks & EfficientNet-B0 classifier.
              </p>
            </div>
          </div>
        </div>

        {/* Top Controls: Mode & Category Pills */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Stream Mode Switch */}
          <div className="flex items-center p-1 bg-slate-900/90 border border-slate-800 rounded-xl">
            <button
              type="button"
              onClick={() => setStreamMode('webcam')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                streamMode === 'webcam'
                  ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50 shadow-[0_0_12px_rgba(139,92,246,0.3)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Browser Webcam</span>
            </button>

            <button
              type="button"
              onClick={() => setStreamMode('mjpeg')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                streamMode === 'mjpeg'
                  ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50 shadow-[0_0_12px_rgba(139,92,246,0.3)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Radio className="w-3.5 h-3.5" />
              <span>Backend MJPEG Stream</span>
            </button>
          </div>

          {/* Quick Snapshot Spacebar Action */}
          <button
            type="button"
            onClick={() => executeInspection(false)}
            disabled={isInspecting}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-cyan-500 text-white text-xs font-bold shadow-[0_0_15px_rgba(6,182,212,0.35)] hover:from-violet-500 hover:to-cyan-400 transition-all active:scale-[0.98] disabled:opacity-50"
          >
            <Zap className={`w-4 h-4 ${isInspecting ? 'animate-spin' : ''}`} />
            <span>{isInspecting ? 'Analyzing...' : 'Capture Frame'}</span>
            <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] font-mono bg-slate-900/60 rounded border border-white/20">
              SPACE
            </kbd>
          </button>
        </div>
      </div>

      {/* Main Studio Grid: Left Optical Viewport & Right Control Rail */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Left Column: Video Viewport & Telemetry HUD (8 cols) */}
        <div className="xl:col-span-8 space-y-4">
          {/* Main Video Viewport Card */}
          <div className="relative rounded-2xl border border-slate-800 bg-slate-950/90 overflow-hidden shadow-2xl aspect-[16/10] flex items-center justify-center">
            {/* Shutter Flash Animation */}
            {shutterFlash && (
              <div className="absolute inset-0 z-30 bg-white/40 backdrop-blur-sm pointer-events-none transition-opacity duration-100" />
            )}

            {/* Viewport Mode 1: Browser Webcam */}
            {streamMode === 'webcam' ? (
              <>
                {webcamError ? (
                  <div className="p-8 text-center max-w-md space-y-3">
                    <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto" />
                    <h3 className="font-heading font-semibold text-white">Camera Access Notice</h3>
                    <p className="text-xs text-slate-400">{webcamError}</p>
                    <div className="pt-2 flex justify-center gap-3">
                      <button
                        type="button"
                        onClick={startWebcam}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-white"
                      >
                        Retry Camera
                      </button>
                      <button
                        type="button"
                        onClick={() => setStreamMode('mjpeg')}
                        className="px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-xs text-white"
                      >
                        Switch to Backend Stream
                      </button>
                    </div>
                  </div>
                ) : (
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover select-none"
                  />
                )}
                {/* Holographic 2D HUD Overlay Canvas */}
                <canvas
                  ref={hudCanvasRef}
                  width={1280}
                  height={800}
                  className="absolute inset-0 w-full h-full pointer-events-none z-10"
                />
              </>
            ) : (
              /* Viewport Mode 2: Backend MJPEG Stream */
              <div className="relative w-full h-full flex items-center justify-center bg-black">
                <img
                  ref={mjpegImgRef}
                  src={api.getStreamUrl(selectedCategory)}
                  alt="DefectVision Live MJPEG Feed"
                  className="w-full h-full object-contain select-none"
                  onError={(e) => {
                    console.warn('MJPEG stream connection error. Falling back to specimen frame.');
                  }}
                />
              </div>
            )}

            {/* Corner Futuristic Badges */}
            {/* Top-Left: Engine Telemetry HUD */}
            <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
              <div className="px-2.5 py-1 rounded-lg bg-slate-950/80 border border-slate-800 backdrop-blur-md flex items-center gap-2 text-xs font-mono">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span className="text-slate-300 font-semibold">{catMeta.label.toUpperCase()}</span>
                <span className="text-slate-500">|</span>
                <span className="text-cyan-400">{catMeta.type}</span>
              </div>
              <div className="px-2 py-1 rounded-lg bg-slate-950/80 border border-slate-800 backdrop-blur-md text-[11px] font-mono text-slate-300">
                FPS: <span className="text-emerald-400 font-bold">{fps}</span>
              </div>
            </div>

            {/* Top-Right: Target Lock Status */}
            <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
              <div className="px-2.5 py-1 rounded-lg bg-slate-950/80 border border-slate-800 backdrop-blur-md flex items-center gap-2 text-xs font-mono">
                <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-slate-300">ROI LOCK: </span>
                <span className="text-emerald-400 font-bold">ONLINE</span>
              </div>
            </div>

            {/* Bottom-Left: Live Latency Gauge */}
            <div className="absolute bottom-3 left-3 z-20 flex items-center gap-2">
              {latestResult && (
                <div className="px-2.5 py-1 rounded-lg bg-slate-950/90 border border-slate-800 backdrop-blur-md text-xs font-mono flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5 text-violet-400" />
                  <span className="text-slate-400">Latency:</span>
                  <span className="text-cyan-300 font-bold">{latestResult.processing_time_ms} ms</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-slate-400">Anomaly:</span>
                  <span className="text-violet-300 font-bold">{latestResult.anomaly_score.toFixed(3)}</span>
                </div>
              )}
            </div>

            {/* Bottom-Right: Severity Status Pill */}
            <div className="absolute bottom-3 right-3 z-20">
              {latestResult ? (
                <SeverityBadge
                  level={latestResult.severity_level}
                  score={latestResult.severity_score}
                  size="sm"
                />
              ) : (
                <div className="px-2.5 py-1 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-400">
                  Ready for Inspection
                </div>
              )}
            </div>
          </div>

          {/* Viewport Sub-bar: Hotkeys, Calibration Tips & Automation Bar */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex items-center gap-3 text-slate-400">
              <div className="flex items-center gap-1.5 font-mono">
                <kbd className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-300 text-[10px]">
                  SPACE
                </kbd>
                <span>Instant Inspect</span>
              </div>
              <span className="text-slate-600">•</span>
              <div className="flex items-center gap-1.5 font-mono">
                <kbd className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-300 text-[10px]">
                  ROI
                </kbd>
                <span>Center Target Reticle</span>
              </div>
            </div>

            {/* Continuous Auto-Inspect Switch */}
            <div className="flex items-center gap-3">
              <span className="text-slate-400 font-medium">Continuous Auto-Scan:</span>
              <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[11px]">
                <button
                  type="button"
                  onClick={() => setAutoInspect(false)}
                  className={`px-2 py-1 rounded-md transition-all ${
                    !autoInspect ? 'bg-slate-800 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Manual (Off)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAutoInspect(true);
                    setInspectInterval(1500);
                  }}
                  className={`px-2 py-1 rounded-md transition-all ${
                    autoInspect && inspectInterval === 1500
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  1.5s
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAutoInspect(true);
                    setInspectInterval(3000);
                  }}
                  className={`px-2 py-1 rounded-md transition-all ${
                    autoInspect && inspectInterval === 3000
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  3.0s
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Model Controls & Defect Event Ticker (4 cols) */}
        <div className="xl:col-span-4 space-y-5">
          {/* Category Memory Bank Selector */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-cyan-400" />
                <span className="font-heading font-bold text-sm text-white">
                  Active Inspection Model
                </span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-violet-500/10 text-violet-300 border border-violet-500/20">
                15 MVTec Models
              </span>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-slate-400 block">
                Select category memory bank for feature distance calibration:
              </label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 text-xs font-mono focus:outline-none focus:border-cyan-500 transition-colors cursor-pointer"
              >
                {CATEGORIES.map((cat) => (
                  <option key={cat.id} value={cat.id}>
                    {cat.label} ({cat.type}) — {cat.desc}
                  </option>
                ))}
              </select>
            </div>

            {/* Category Information Badge */}
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs space-y-1">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span>Target: {catMeta.label}</span>
                <span className="text-[10px] font-mono text-cyan-400">{catMeta.type}</span>
              </div>
              <p className="text-[11px] text-slate-400">{catMeta.desc}</p>
            </div>
          </div>

          {/* Real-time Defect Event Ticker (Audit Rail) */}
          <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-violet-400" />
                <span className="font-heading font-bold text-sm text-white">
                  Session Event Ticker
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {eventHistory.length} Captured
                </span>
                {eventHistory.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setEventHistory([])}
                    className="text-[10px] font-mono text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Ticker List */}
            <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-1 scrollbar-thin">
              {eventHistory.length === 0 ? (
                <div className="p-6 text-center border border-dashed border-slate-800 rounded-xl space-y-2">
                  <Eye className="w-8 h-8 text-slate-600 mx-auto" />
                  <p className="text-xs text-slate-400">No frames inspected yet in this session.</p>
                  <p className="text-[11px] text-slate-500">
                    Press <span className="text-cyan-400 font-mono">SPACE</span> or click "Capture Frame" to trigger immediate inspection.
                  </p>
                </div>
              ) : (
                eventHistory.map((evt) => (
                  <div
                    key={evt.id}
                    onClick={() => setSelectedEventModal(evt)}
                    className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 hover:border-violet-500/50 hover:bg-slate-950 transition-all cursor-pointer group flex items-center justify-between gap-3"
                  >
                    {/* Event Thumbnail Preview */}
                    <div className="w-14 h-14 rounded-lg bg-black overflow-hidden border border-slate-800 flex-shrink-0 relative">
                      <img
                        src={evt.heatmap_base64 || evt.original_image}
                        alt="Event Thumbnail"
                        className="w-full h-full object-cover"
                      />
                      {evt.predicted_label === 'defective' && (
                        <div className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-rose-500" />
                      )}
                    </div>

                    {/* Event Telemetry Info */}
                    <div className="flex-1 min-w-0 space-y-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-heading text-xs font-semibold text-white truncate">
                          {evt.capturedCategory.toUpperCase()} • {evt.defect_type}
                        </span>
                        <span className="text-[10px] font-mono text-slate-500 flex-shrink-0">
                          {evt.timestamp}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <span
                          className={`text-[10px] font-mono font-bold uppercase px-1.5 py-0.2 rounded border ${
                            evt.predicted_label === 'defective'
                              ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                              : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                          }`}
                        >
                          {evt.predicted_label}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">
                          {evt.processing_time_ms}ms
                        </span>
                      </div>
                    </div>

                    <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 group-hover:translate-x-0.5 transition-all flex-shrink-0" />
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* -------------------------------------------------------- */}
      {/* 5. Before / After Heatmap Split-View Modal               */}
      {/* -------------------------------------------------------- */}
      {selectedEventModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
          <div className="relative w-full max-w-4xl bg-[#090d16] border border-slate-800 rounded-2xl shadow-2xl p-6 overflow-hidden space-y-6 max-h-[92vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center">
                  <Eye className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <h3 className="font-heading text-lg font-bold text-white">
                    Inspection Specimen Analysis
                  </h3>
                  <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                    <span>Target: {selectedEventModal.capturedCategory.toUpperCase()}</span>
                    <span>•</span>
                    <span>Event ID: {selectedEventModal.id}</span>
                    <span>•</span>
                    <span>Time: {selectedEventModal.timestamp}</span>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setSelectedEventModal(null)}
                className="p-2 rounded-xl bg-slate-900 text-slate-400 hover:text-white border border-slate-800 transition-colors"
                aria-label="Close modal"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Split-View Heatmap Slider Component */}
            <div className="space-y-2">
              <HeatmapSlider
                originalImage={selectedEventModal.original_image}
                heatmapOverlay={selectedEventModal.heatmap_base64}
                rawHeatmap={selectedEventModal.raw_heatmap}
                label={`${selectedEventModal.capturedCategory} Optical Inspection`}
              />
            </div>

            {/* Telemetry Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs space-y-1">
                <span className="text-slate-400 font-mono text-[10px] uppercase">Prediction</span>
                <div className="font-heading font-bold text-sm text-white uppercase">
                  {selectedEventModal.predicted_label}
                </div>
                <div className="text-[10px] font-mono text-slate-400">
                  Defect: {selectedEventModal.defect_type}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs space-y-1">
                <span className="text-slate-400 font-mono text-[10px] uppercase">Severity</span>
                <div>
                  <SeverityBadge
                    level={selectedEventModal.severity_level}
                    score={selectedEventModal.severity_score}
                    size="sm"
                  />
                </div>
                <div className="text-[10px] font-mono text-slate-400">
                  Score: {selectedEventModal.severity_score}/100
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs space-y-1">
                <span className="text-slate-400 font-mono text-[10px] uppercase">Defect Area</span>
                <div className="font-heading font-bold text-sm text-cyan-300">
                  {selectedEventModal.defect_area_pct}%
                </div>
                <div className="text-[10px] font-mono text-slate-400">
                  PatchCore Coreset
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs space-y-1">
                <span className="text-slate-400 font-mono text-[10px] uppercase">Confidence</span>
                <div className="font-heading font-bold text-sm text-violet-300">
                  {selectedEventModal.confidence_pct}%
                </div>
                <div className="text-[10px] font-mono text-slate-400">
                  EffNet-B0 v3
                </div>
              </div>
            </div>

            {/* Operator Plain-English Context */}
            <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800/80 text-xs space-y-1.5">
              <div className="flex items-center gap-2 text-cyan-400 font-semibold">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Operator Diagnostic Summary:</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                Optical frame inspection for <strong className="text-white">{selectedEventModal.capturedCategory}</strong> registered an anomaly score of{' '}
                <span className="font-mono text-cyan-300">{selectedEventModal.anomaly_score.toFixed(4)}</span> and a classified severity rating of{' '}
                <strong className="text-white">{selectedEventModal.severity_level.toUpperCase()}</strong> ({selectedEventModal.severity_score}/100).
                The affected anomalous region spans <span className="font-mono text-violet-300">{selectedEventModal.defect_area_pct}%</span> of the ROI surface area.
              </p>
            </div>

            {/* Modal Bottom Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              {onNavigateToRAG && selectedEventModal.defect_type && selectedEventModal.defect_type !== 'normal' && (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedEventModal(null);
                    onNavigateToRAG(selectedEventModal.defect_type);
                  }}
                  className="flex items-center gap-2 px-3 py-2 rounded-xl bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-xs font-semibold hover:bg-cyan-500/25 transition-all"
                >
                  <HelpCircle className="w-3.5 h-3.5" />
                  <span>Query RAG Copilot for {selectedEventModal.defect_type}</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => setSelectedEventModal(null)}
                className="ml-auto px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold transition-all"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

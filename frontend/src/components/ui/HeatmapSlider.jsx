import React, { useState } from 'react';
import { Eye, Layers, SplitSquareVertical, Sliders } from 'lucide-react';

export default function HeatmapSlider({ originalImage, heatmapOverlay, rawHeatmap, label = 'Defect Inspection' }) {
  const [blendAlpha, setBlendAlpha] = useState(50); // 0 to 100
  const [viewMode, setViewMode] = useState('blend'); // 'blend' | 'split' | 'raw'

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Mode Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2 bg-slate-900/80 border border-slate-800 rounded-lg">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setViewMode('blend')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              viewMode === 'blend'
                ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50 shadow-[0_0_10px_rgba(139,92,246,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Interactive Blend</span>
          </button>

          <button
            type="button"
            onClick={() => setViewMode('split')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              viewMode === 'split'
                ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50 shadow-[0_0_10px_rgba(139,92,246,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>

          {rawHeatmap && (
            <button
              type="button"
              onClick={() => setViewMode('raw')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                viewMode === 'raw'
                  ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50 shadow-[0_0_10px_rgba(139,92,246,0.3)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Anomaly Map Only</span>
            </button>
          )}
        </div>

        {viewMode === 'blend' && (
          <div className="flex items-center gap-3 px-2">
            <span className="text-xs text-slate-400 font-mono">Clean</span>
            <input
              type="range"
              min="0"
              max="100"
              value={blendAlpha}
              onChange={(e) => setBlendAlpha(Number(e.target.value))}
              aria-label="Heatmap opacity slider"
              className="w-32 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <span className="text-xs text-cyan-400 font-mono font-semibold">Heatmap ({blendAlpha}%)</span>
          </div>
        )}
      </div>

      {/* Visual Canvas Area */}
      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black/60 aspect-square flex items-center justify-center">
        {viewMode === 'split' ? (
          <div className="grid grid-cols-2 w-full h-full gap-1 p-1">
            <div className="relative w-full h-full rounded-lg overflow-hidden border border-slate-800">
              <img src={originalImage} alt="Clean Input" className="w-full h-full object-contain" />
              <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 text-[10px] font-mono text-slate-300 border border-slate-700">
                Original Input
              </div>
            </div>
            <div className="relative w-full h-full rounded-lg overflow-hidden border border-violet-500/40">
              <img src={heatmapOverlay} alt="PatchCore Heatmap" className="w-full h-full object-contain" />
              <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 text-[10px] font-mono text-cyan-300 border border-violet-700">
                PatchCore Anomaly Map
              </div>
            </div>
          </div>
        ) : viewMode === 'raw' ? (
          <div className="relative w-full h-full">
            <img src={rawHeatmap || heatmapOverlay} alt="Raw Heatmap" className="w-full h-full object-contain" />
            <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 text-[10px] font-mono text-cyan-300 border border-cyan-700">
              Normalized Patch Distance Map
            </div>
          </div>
        ) : (
          /* Interactive Blend Mode */
          <div className="relative w-full h-full flex items-center justify-center">
            {/* Base Image */}
            <img
              src={originalImage}
              alt="Base Specimen"
              className="w-full h-full object-contain select-none"
            />
            {/* Blended Heatmap Overlay */}
            <img
              src={heatmapOverlay}
              alt="Heatmap Overlay"
              className="absolute inset-0 w-full h-full object-contain select-none transition-opacity duration-75 pointer-events-none"
              style={{ opacity: blendAlpha / 100 }}
            />
            <div className="absolute bottom-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/90 text-xs font-mono text-slate-200 border border-slate-700/80 backdrop-blur-md">
              <Eye className="w-3.5 h-3.5 text-cyan-400" />
              <span>PatchCore Heatmap ({blendAlpha}%)</span>
            </div>
          </div>
        )}
      </div>

      {/* Heatmap Spectrum Legend */}
      <div className="flex items-center justify-between px-3 py-2 bg-slate-900/50 border border-slate-800/60 rounded-lg text-xs">
        <span className="text-slate-400 font-mono">Heatmap Spectrum:</span>
        <div className="flex items-center gap-2">
          <span className="text-blue-400 font-medium">Nominal (0.0)</span>
          <div className="w-28 h-2.5 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-400 to-red-600 shadow-inner" />
          <span className="text-red-400 font-medium">Anomaly Peak (1.0)</span>
        </div>
      </div>
    </div>
  );
}

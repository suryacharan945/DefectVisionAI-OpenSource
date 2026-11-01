import React from 'react';
import { ArrowRight, ShieldCheck, Cpu, Database, Award, Info } from 'lucide-react';
import IndustrialCore3D from '../3d/IndustrialCore3D';

export default function HeroSection({ onStartInspect, onViewModelComparison }) {
  return (
    <section className="relative min-h-[85vh] flex items-center justify-center overflow-hidden border-b border-slate-800/80">
      {/* Background Gradients & Ambient Effects */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.18),rgba(255,255,255,0))]" />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-tr from-violet-600/10 via-cyan-500/10 to-transparent rounded-full blur-3xl pointer-events-none" />

      {/* 3D Canvas Layer */}
      <div className="absolute inset-0 z-0 opacity-75 hover:opacity-100 transition-opacity duration-700 pointer-events-auto">
        <IndustrialCore3D />
      </div>

      {/* Glass UI Overlay Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-16 text-center flex flex-col items-center pointer-events-none">
        
        {/* System Tag */}
        <div className="pointer-events-auto inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/80 border border-violet-500/30 backdrop-blur-md mb-6 shadow-[0_0_20px_rgba(139,92,246,0.25)]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="text-xs font-mono font-medium text-slate-200">
            DefectVision AI • Production Inspection Engine v3.0
          </span>
        </div>

        {/* Cinematic Title */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-white max-w-4xl leading-[1.1] mb-6">
          Autonomous Defect Vision with{' '}
          <span className="bg-gradient-to-r from-violet-400 via-cyan-300 to-indigo-300 bg-clip-text text-transparent">
            Sub-Millimeter Heatmaps
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-base sm:text-lg text-slate-300 max-w-2xl font-normal leading-relaxed mb-8">
          A research-grade industrial anomaly inspection platform combining <strong className="text-cyan-300 font-medium">PatchCore</strong> memory banks,{' '}
          <strong className="text-violet-300 font-medium">EfficientNet-B0</strong> classification, and{' '}
          <strong className="text-slate-200 font-medium">LLM-grounded explanations</strong> over the 15-category MVTec AD benchmark.
        </p>

        {/* CTA Buttons */}
        <div className="pointer-events-auto flex flex-wrap items-center justify-center gap-4 mb-12">
          <button
            type="button"
            onClick={onStartInspect}
            className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-cyan-500 text-white font-semibold text-sm hover:from-violet-500 hover:to-cyan-400 shadow-[0_0_25px_rgba(139,92,246,0.45)] transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Launch Live Inspection</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={onViewModelComparison}
            className="flex items-center gap-2 px-5 py-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700/80 font-semibold text-sm transition-all hover:border-cyan-500/40 backdrop-blur-md"
          >
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>View 4-Model Leaderboard</span>
          </button>
        </div>

        {/* Key Metrics Banner */}
        <div className="pointer-events-auto w-full grid grid-cols-2 md:grid-cols-4 gap-3 max-w-4xl p-2 rounded-2xl bg-slate-950/60 border border-slate-800/80 backdrop-blur-xl shadow-2xl">
          <div className="p-3 text-center border-r border-slate-800/50">
            <div className="flex items-center justify-center gap-1 text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-1">
              <Award className="w-3.5 h-3.5 text-violet-400" />
              <span>Best AUROC</span>
            </div>
            <div className="text-2xl font-mono font-bold text-white">0.9732</div>
            <p className="text-[11px] text-slate-400">EfficientNet-B0 v3</p>
          </div>

          <div className="p-3 text-center border-r border-slate-800/50">
            <div className="flex items-center justify-center gap-1 text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-1">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>PatchCore Map</span>
            </div>
            <div className="text-2xl font-mono font-bold text-cyan-300">0.9651</div>
            <p className="text-[11px] text-slate-400">Localization AUROC</p>
          </div>

          <div className="p-3 text-center border-r border-slate-800/50">
            <div className="flex items-center justify-center gap-1 text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-1">
              <Database className="w-3.5 h-3.5 text-indigo-400" />
              <span>Categories</span>
            </div>
            <div className="text-2xl font-mono font-bold text-white">15</div>
            <p className="text-[11px] text-slate-400">Per-Category Banks</p>
          </div>

          <div className="p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Evaluated</span>
            </div>
            <div className="text-2xl font-mono font-bold text-white">1,725</div>
            <p className="text-[11px] text-slate-400">Leakage-Free Benchmark</p>
          </div>
        </div>

        {/* Honest Architecture Constraint Note */}
        <div className="pointer-events-auto mt-4 flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/40 border border-slate-800/60 text-[11px] text-slate-400">
          <Info className="w-3.5 h-3.5 text-cyan-400" />
          <span>Models are per-category. Select product type before upload so the matching PatchCore memory bank is routed.</span>
        </div>

      </div>
    </section>
  );
}

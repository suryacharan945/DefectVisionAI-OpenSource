import React, { useState } from 'react';
import {
  ShieldCheck,
  Video,
  Activity,
  Cpu,
  Layers,
  FileSearch,
  HelpCircle,
  BarChart3,
  Radio,
  Sparkles,
  Menu,
  X,
  Server,
  Zap,
  CheckCircle2
} from 'lucide-react';

export default function SidebarNav({ activeTab, onTabChange, backendHealth }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const navGroups = [
    {
      group: 'Core Inspection',
      items: [
        { id: 'inspect', label: 'Specimen Inspection', icon: ShieldCheck, tag: 'P0' },
        { id: 'live', label: 'Live Optical Stream', icon: Video, tag: 'LIVE', isLive: true },
        { id: 'models', label: 'Models Leaderboard', icon: Cpu, tag: 'P0' },
        { id: 'history', label: 'History Browser', icon: FileSearch, tag: 'P0' },
      ],
    },
    {
      group: 'High-Throughput & RAG',
      items: [
        { id: 'batch', label: 'Batch Processing', icon: Layers, tag: 'P1' },
        { id: 'rag', label: 'RAG Defect Copilot', icon: HelpCircle, tag: 'P1' },
        { id: 'dashboard', label: 'Ask the Data', icon: BarChart3, tag: 'P1' },
        { id: 'drift', label: 'Drift Monitor', icon: Radio, tag: 'P1' },
      ],
    },
    {
      group: 'Advanced Research',
      items: [
        { id: 'rca', label: 'RCA & Active Learning', icon: Sparkles, tag: 'P2' },
      ],
    },
  ];

  const isOnline = backendHealth && backendHealth.status === 'ready';

  const handleSelectTab = (id) => {
    onTabChange(id);
    setMobileOpen(false);
  };

  return (
    <>
      {/* Mobile Top Header */}
      <div className="md:hidden sticky top-0 z-40 w-full flex items-center justify-between px-4 py-3 bg-[#090d16]/95 border-b border-slate-800/80 backdrop-blur-xl">
        <div
          onClick={() => handleSelectTab('inspect')}
          className="flex items-center gap-2.5 cursor-pointer"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-cyan-400 p-[1px]">
            <div className="w-full h-full bg-slate-950 rounded-[7px] flex items-center justify-center">
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
          </div>
          <span className="font-heading text-base font-bold text-white">
            DefectVision<span className="text-violet-400">.AI</span>
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div
            className={`w-2.5 h-2.5 rounded-full ${
              isOnline ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'
            }`}
            title={isOnline ? 'Backend Online' : 'Backend Offline'}
          />
          <button
            type="button"
            onClick={() => setMobileOpen(!mobileOpen)}
            className="p-2 rounded-lg bg-slate-900 text-slate-300 border border-slate-800"
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5 text-cyan-400" />}
          </button>
        </div>
      </div>

      {/* Backdrop overlay on mobile */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar (Desktop Fixed Left & Mobile Drawer) */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 flex flex-col justify-between bg-[#090d16]/95 border-r border-slate-800/80 backdrop-blur-2xl p-4 transition-transform duration-300 ease-in-out md:translate-x-0 ${
          mobileOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Top: Brand Logo & Quick Action */}
        <div className="space-y-4">
          <div
            onClick={() => handleSelectTab('inspect')}
            className="flex items-center gap-3 cursor-pointer select-none group px-1 pt-1"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-600 via-indigo-600 to-cyan-400 p-[1px] shadow-[0_0_15px_rgba(139,92,246,0.35)] group-hover:shadow-[0_0_22px_rgba(6,182,212,0.5)] transition-all">
              <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-heading text-lg font-bold tracking-tight text-white group-hover:text-cyan-300 transition-colors">
                  DefectVision<span className="text-violet-400">.AI</span>
                </span>
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="px-1.5 py-0.2 text-[9px] font-mono uppercase tracking-wider rounded bg-violet-500/20 text-violet-300 border border-violet-500/30">
                  Production
                </span>
                <span className="text-[10px] text-slate-400 font-mono">v3.0</span>
              </div>
            </div>
          </div>

          {/* Quick Inspect CTA */}
          <button
            type="button"
            onClick={() => handleSelectTab('inspect')}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl bg-gradient-to-r from-violet-600 via-indigo-600 to-cyan-500 text-white text-xs font-semibold hover:from-violet-500 hover:to-cyan-400 transition-all shadow-[0_0_15px_rgba(139,92,246,0.3)] active:scale-[0.98]"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>New Live Inspection</span>
          </button>
        </div>

        {/* Middle: Grouped Navigation Links */}
        <div className="flex-1 overflow-y-auto py-5 space-y-6 scrollbar-thin pr-1">
          {navGroups.map((group, gIdx) => (
            <div key={gIdx} className="space-y-1.5">
              <div className="px-2 text-[10px] font-mono font-semibold uppercase tracking-wider text-slate-400">
                {group.group}
              </div>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => handleSelectTab(item.id)}
                      className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-gradient-to-r from-violet-600/30 to-cyan-600/10 text-cyan-300 border border-violet-500/50 shadow-[0_0_15px_rgba(139,92,246,0.2)]'
                          : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                        <span>{item.label}</span>
                      </div>
                      {item.isLive ? (
                        <span className="inline-flex items-center gap-1 text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse shadow-[0_0_8px_rgba(244,63,94,0.3)]">
                          <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping" />
                          LIVE
                        </span>
                      ) : (
                        <span
                          className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${
                            isActive
                              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                              : 'bg-slate-900 text-slate-400 border-slate-800'
                          }`}
                        >
                          {item.tag}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Bottom: Live System & Engine Telemetry */}
        <div className="pt-3 border-t border-slate-800/80 space-y-2">
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/90 text-xs font-mono space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-[10px] uppercase">Engine Status</span>
              <span
                className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                  isOnline
                    ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                    : backendHealth?.mode === 'demo'
                    ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                    : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isOnline
                      ? 'bg-emerald-400 animate-ping'
                      : backendHealth?.mode === 'demo'
                      ? 'bg-amber-400'
                      : 'bg-rose-400'
                  }`}
                />
                <span>
                  {isOnline
                    ? 'ONLINE (LIVE ML)'
                    : backendHealth?.mode === 'demo'
                    ? 'DEMO MODE'
                    : 'OFFLINE'}
                </span>
              </span>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-300">
              <span className="text-slate-400">Accelerator:</span>
              <span className="text-cyan-300 font-bold uppercase">{backendHealth?.device || 'CPU'}</span>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-300">
              <span className="text-slate-400">Memory Banks:</span>
              <span className="text-violet-300 font-bold">15 Categories</span>
            </div>
          </div>

          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 px-1 pt-0.5">
            <a
              href="https://defectvision-ai.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1 font-semibold"
            >
              <span>Vercel Live ↗</span>
            </a>
            <a
              href="https://github.com/suryacharan945/DefectVision-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-violet-400 transition-colors"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </aside>
    </>
  );
}

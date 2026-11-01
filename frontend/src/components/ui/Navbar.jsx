import React from 'react';
import { ShieldCheck, Activity, Cpu, Layers, FileSearch, HelpCircle, BarChart3, Radio, Sparkles } from 'lucide-react';

export default function Navbar({ activeTab, onTabChange, backendHealth }) {
  const navItems = [
    { id: 'inspect', label: 'Live Inspection', icon: ShieldCheck, tag: 'P0' },
    { id: 'models', label: 'Models Leaderboard', icon: Cpu, tag: 'P0' },
    { id: 'history', label: 'History Browser', icon: FileSearch, tag: 'P0' },
    { id: 'batch', label: 'Batch Processing', icon: Layers, tag: 'P1' },
    { id: 'rag', label: 'RAG Defect Copilot', icon: HelpCircle, tag: 'P1' },
    { id: 'dashboard', label: 'Ask the Data', icon: BarChart3, tag: 'P1' },
    { id: 'drift', label: 'Drift Monitor', icon: Radio, tag: 'P1' },
    { id: 'rca', label: 'RCA & Active Learning', icon: Sparkles, tag: 'P2' },
  ];

  const isOnline = backendHealth && backendHealth.status === 'ready';

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-[#090d16]/85 backdrop-blur-xl transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          
          {/* Logo Brand */}
          <div
            onClick={() => onTabChange('inspect')}
            className="flex items-center gap-3 cursor-pointer select-none group"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-600 via-indigo-600 to-cyan-400 p-[1px] shadow-[0_0_15px_rgba(139,92,246,0.35)] group-hover:shadow-[0_0_22px_rgba(6,182,212,0.5)] transition-all">
              <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Activity className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading text-lg font-bold tracking-tight text-white group-hover:text-cyan-300 transition-colors">
                  DefectVision<span className="text-violet-400">.AI</span>
                </span>
                <span className="px-1.5 py-0.2 text-[10px] font-mono uppercase tracking-wider rounded bg-violet-500/20 text-violet-300 border border-violet-500/30">
                  Production
                </span>
              </div>
              <p className="text-[10px] text-slate-400 hidden sm:block">Industrial Anomaly Vision System</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden xl:flex items-center gap-1 overflow-x-auto py-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onTabChange(item.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                    isActive
                      ? 'bg-violet-600/25 text-cyan-300 border border-violet-500/50 shadow-[0_0_12px_rgba(139,92,246,0.25)]'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Status Indicator & Action */}
          <div className="flex items-center gap-3">
            {/* Live Backend Heartbeat */}
            <div
              className={`flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-mono border backdrop-blur-md ${
                isOnline
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-400 animate-pulse'
              }`}
              title={
                isOnline
                  ? `FastAPI Server: Ready on ${backendHealth?.device || 'cpu'} | 15 PatchCore Models Loaded`
                  : 'Backend Disconnected: Ensure uvicorn is running on port 8000'
              }
            >
              <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'}`} />
              <span className="hidden sm:inline font-semibold">
                {isOnline ? `Online (${backendHealth?.device?.toUpperCase() || 'CPU'})` : 'Offline'}
              </span>
            </div>

            <button
              type="button"
              onClick={() => onTabChange('inspect')}
              className="hidden sm:flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-violet-600 to-cyan-500 text-white text-xs font-semibold hover:from-violet-500 hover:to-cyan-400 transition-all shadow-[0_0_15px_rgba(139,92,246,0.35)] active:scale-95"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Inspect Specimen</span>
            </button>
          </div>

        </div>

        {/* Mobile / Compact Tab Strip */}
        <div className="xl:hidden flex items-center gap-1 overflow-x-auto pb-2 pt-1 border-t border-slate-800/40">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-violet-600/30 text-cyan-300 border border-violet-500/50'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

      </div>
    </header>
  );
}

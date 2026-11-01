import React, { useState, useEffect } from 'react';
import {
  Radio,
  AlertTriangle,
  AlertOctagon,
  CheckCircle2,
  TrendingDown,
  Activity,
  Filter,
  RefreshCw
} from 'lucide-react';
import { api } from '../../api/client';
import { CATEGORIES } from '../../utils/categories';
import Tooltip from '../ui/Tooltip';

export default function DriftMonitorSection() {
  const [driftLogs, setDriftLogs] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDrift() {
      try {
        const data = await api.getDrift();
        setDriftLogs(data);
      } catch (err) {
        console.error('Failed to load drift data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadDrift();
  }, []);

  const streams = [
    {
      id: 'clean_normal_only',
      title: 'Clean Normal Stream',
      subtitle: 'Uncorrupted nominal baseline',
      expectedStatus: 'Minimal Drift (Healthy)',
      badgeClass: 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400',
      icon: CheckCircle2,
      desc: 'Samples match the reference distribution closely (embedding distance ~1.0, KL divergence near 0.0). No alerts triggered.',
    },
    {
      id: 'test_natural_mixed',
      title: 'Test Natural Mixed Stream',
      subtitle: 'Standard production stream with anomalies',
      expectedStatus: 'Expected Variance (Monitored)',
      badgeClass: 'bg-amber-500/15 border-amber-500/40 text-amber-400',
      icon: AlertTriangle,
      desc: 'Combines normal components with natural defects. Moderate distribution divergence due to defect patterns.',
    },
    {
      id: 'corrupted_production',
      title: 'Corrupted Production Stream',
      subtitle: 'Simulated lens blur, lighting shift & sensor noise',
      expectedStatus: 'Systematic Drift (Critical Alert)',
      badgeClass: 'bg-rose-500/15 border-rose-500/40 text-rose-400',
      icon: AlertOctagon,
      desc: 'Systematic sensor degradation causing severe statistical drift (KL divergence up to 63.8). Requires immediate recalibration.',
    },
  ];

  const filteredLogs = driftLogs.filter((row) =>
    selectedCategory === 'all' ? true : row.category === selectedCategory
  );

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20">
              Reliability Engine (P1)
            </span>
            <span className="text-xs font-mono text-slate-400">• Continuous Distribution Shift Tracking</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
            Production Drift & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Model Health Monitor</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Real benchmark drift logs tracking feature embedding distances and KL divergence to prevent silent model degradation on the factory floor.
          </p>
        </div>

        {/* Category filter */}
        <div className="flex items-center gap-2">
          <label htmlFor="drift-category-filter" className="text-xs font-semibold text-slate-300">Category</label>
          <select
            id="drift-category-filter"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            <option value="all">All 15 Categories</option>
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 3 Stream Diagnostic Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {streams.map((st) => {
          const Icon = st.icon;
          const streamRows = driftLogs.filter((r) => r.stream === st.id);
          const alertCount = streamRows.filter((r) => r.drift_alert === true || r.drift_alert === 'True').length;
          const totalCount = streamRows.length;
          const avgDist = streamRows.length
            ? (streamRows.reduce((acc, r) => acc + (parseFloat(r.embedding_distance_score) || 0), 0) / streamRows.length).toFixed(2)
            : '0.00';

          return (
            <div
              key={st.id}
              className="p-5 rounded-2xl border border-slate-800 bg-slate-950/70 backdrop-blur-xl flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className={`px-2.5 py-1 rounded-full text-xs font-semibold border flex items-center gap-1.5 ${st.badgeClass}`}>
                    <Icon className="w-3.5 h-3.5" />
                    <span>{st.expectedStatus}</span>
                  </div>
                  <span className="text-xs font-mono text-slate-400">
                    {alertCount} / {totalCount} Alerts
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mb-1">{st.title}</h3>
                <p className="text-xs font-mono text-slate-400 mb-3">{st.subtitle}</p>
                <p className="text-xs text-slate-300 leading-relaxed mb-4">{st.desc}</p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Mean Distance:</span>
                <span className="text-white font-bold text-sm">{avgDist}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Stream Comparison Data Table */}
      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/70 shadow-2xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-slate-100">Category Drift Metric Log</h3>
          </div>
          <span className="text-xs font-mono text-slate-400">Threshold: L2 &gt; 1.5 | KL &gt; 0.5</span>
        </div>

        {loading ? (
          <div className="py-16 text-center text-xs font-mono text-slate-400 flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
            <span>Loading stream drift logs...</span>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[500px]">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400 uppercase">
                <tr>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Stream Channel</th>
                  <th className="py-3 px-4">Images</th>
                  <th className="py-3 px-4">Embedding Dist (L2)</th>
                  <th className="py-3 px-4">KL Divergence</th>
                  <th className="py-3 px-4">Alert Status</th>
                  <th className="py-3 px-4">Alert Reasons</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {filteredLogs.map((row, idx) => {
                  const isAlert = row.drift_alert === true || row.drift_alert === 'True';
                  return (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-900/40 transition-colors ${
                        row.stream === 'corrupted_production' ? 'bg-rose-950/10' : ''
                      }`}
                    >
                      <td className="py-2.5 px-4 font-sans font-semibold text-slate-200 capitalize">
                        {row.category}
                      </td>
                      <td className="py-2.5 px-4 text-slate-300">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] ${
                            row.stream === 'clean_normal_only'
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : row.stream === 'test_natural_mixed'
                              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                              : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          }`}
                        >
                          {row.stream.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-slate-400">{row.n_images}</td>
                      <td className="py-2.5 px-4 font-bold text-white">
                        {typeof row.embedding_distance_score === 'number'
                          ? row.embedding_distance_score.toFixed(3)
                          : row.embedding_distance_score}
                      </td>
                      <td className="py-2.5 px-4 text-slate-300">
                        {typeof row.kl_divergence_score === 'number'
                          ? row.kl_divergence_score.toFixed(3)
                          : row.kl_divergence_score}
                      </td>
                      <td className="py-2.5 px-4">
                        {isAlert ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-rose-500/15 text-rose-400 border border-rose-500/40 font-semibold">
                            <AlertTriangle className="w-3 h-3" />
                            <span>DRIFT ALERT</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/15 text-emerald-400 border border-emerald-500/40">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>Normal</span>
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 text-[11px] text-slate-400 truncate max-w-[220px]">
                        {row.alert_reasons || '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

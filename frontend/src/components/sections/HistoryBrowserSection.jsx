import React, { useState, useEffect } from 'react';
import {
  FileSearch,
  Filter,
  Search,
  ChevronLeft,
  ChevronRight,
  Eye,
  X,
  Sparkles,
  Layers,
  RefreshCw,
  ExternalLink,
  Download
} from 'lucide-react';
import { api } from '../../api/client';
import { CATEGORIES } from '../../utils/categories';
import SeverityBadge from '../ui/SeverityBadge';
import MetricCard from '../ui/MetricCard';

export default function HistoryBrowserSection({ onInspectSpecimen }) {
  const [records, setRecords] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(15);
  const [category, setCategory] = useState('');
  const [severityLevel, setSeverityLevel] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  // Selected row for detail inspection modal
  const [selectedRecord, setSelectedRecord] = useState(null);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const data = await api.getSeverityLogs({
        category,
        severity_level: severityLevel,
        page,
        page_size: pageSize,
      });
      setRecords(data.records || []);
      setTotalRecords(data.total_records || 0);
      setTotalPages(data.total_pages || 1);
    } catch (err) {
      console.error('Failed to fetch severity logs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [category, severityLevel, page, pageSize]);

  // Client-side search filtering on defect_type / filename
  const filteredRecords = records.filter((r) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      r.id?.toLowerCase().includes(q) ||
      r.defect_type?.toLowerCase().includes(q) ||
      r.filename?.toLowerCase().includes(q) ||
      r.category?.toLowerCase().includes(q)
    );
  });

  const handleExportCSV = () => {
    if (!filteredRecords || filteredRecords.length === 0) return;
    const headers = ['id', 'category', 'defect_type', 'severity_score', 'severity_level', 'defect_area_pct', 'confidence_pct', 'anomaly_norm', 'is_defective'];
    const rows = filteredRecords.map((r) => [
      `"${(r.id || '').replace(/"/g, '""')}"`,
      `"${(r.category || '').replace(/"/g, '""')}"`,
      `"${(r.defect_type || '').replace(/"/g, '""')}"`,
      r.severity_score ?? '',
      `"${(r.severity_level || '').replace(/"/g, '""')}"`,
      r.defect_area_pct ?? '',
      r.confidence_pct ?? '',
      r.anomaly_norm ?? '',
      r.is_defective ? 'true' : 'false'
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `inspection_history_${category || 'all'}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Benchmark Dataset (P0)
            </span>
            <span className="text-xs font-mono text-slate-400">• 1,725 Scored Test Images</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
            Inspection History & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Benchmark Browser</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Query the complete 1,725-specimen MVTec AD test log. Click any record to inspect the image, localization heatmap, severity breakdown, and matching verified explanation.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-end sm:items-center gap-3">
          <button
            type="button"
            onClick={handleExportCSV}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-700 hover:border-cyan-500/40 transition-colors shadow-sm"
            title="Export filtered records to CSV"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export Filtered CSV</span>
          </button>
          <div className="text-right font-mono text-xs text-slate-400">
            Showing <span className="text-white font-bold">{records.length}</span> of{' '}
            <span className="text-cyan-300 font-bold">{totalRecords.toLocaleString()}</span> images
          </div>
        </div>
      </div>

      {/* Filter Controls Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6 p-4 rounded-xl bg-slate-950/70 border border-slate-800 backdrop-blur-md">
        
        {/* Search Input */}
        <div className="relative">
          <label htmlFor="history-search-input" className="sr-only">Search specimen</label>
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            id="history-search-input"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by ID, defect type..."
            className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Category Filter */}
        <div className="relative">
          <label htmlFor="history-category-filter" className="sr-only">Filter by category</label>
          <select
            id="history-category-filter"
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
              setPage(1);
            }}
            className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            <option value="">All Categories (15)</option>
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label} ({c.type})
              </option>
            ))}
          </select>
        </div>

        {/* Severity Level Filter */}
        <div className="relative">
          <label htmlFor="history-severity-filter" className="sr-only">Filter by severity</label>
          <select
            id="history-severity-filter"
            value={severityLevel}
            onChange={(e) => {
              setSeverityLevel(e.target.value);
              setPage(1);
            }}
            className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            <option value="">All Severity Levels</option>
            <option value="Critical">Critical (Score ≥ 70)</option>
            <option value="Moderate">Moderate (Score 40–69)</option>
            <option value="Minor">Minor / Normal (Score &lt; 40)</option>
          </select>
        </div>

        {/* Page Size & Reset */}
        <div className="flex items-center gap-2">
          <label htmlFor="history-pagesize-select" className="sr-only">Items per page</label>
          <select
            id="history-pagesize-select"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="w-24 px-2 py-2 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-300 focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            <option value={15}>15 / page</option>
            <option value={25}>25 / page</option>
            <option value={50}>50 / page</option>
          </select>

          {(category || severityLevel || searchQuery) && (
            <button
              type="button"
              onClick={() => {
                setCategory('');
                setSeverityLevel('');
                setSearchQuery('');
                setPage(1);
              }}
              className="flex-1 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 text-xs font-medium border border-slate-700 transition-colors text-center"
            >
              Reset Filters
            </button>
          )}
        </div>

      </div>

      {/* Table Content */}
      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/70 backdrop-blur-xl shadow-2xl mb-6">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800/90 bg-slate-900/60 font-mono text-slate-400 uppercase">
                <th className="py-3 px-3">Specimen ID</th>
                <th className="py-3 px-3">Category</th>
                <th className="py-3 px-3">Defect Type</th>
                <th className="py-3 px-3">Severity Score</th>
                <th className="py-3 px-3">Severity Level</th>
                <th className="py-3 px-3">Defect Area %</th>
                <th className="py-3 px-3">Confidence %</th>
                <th className="py-3 px-3">Anomaly Norm</th>
                <th className="py-3 px-3 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 font-mono">
              {loading ? (
                <tr>
                  <td colSpan={9} className="py-16 text-center text-slate-400">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
                      <span>Querying benchmark database...</span>
                    </div>
                  </td>
                </tr>
              ) : filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-16 text-center text-slate-400">
                    No inspection logs matched the selected filter criteria.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((rec) => (
                  <tr
                    key={rec.id}
                    onClick={() => setSelectedRecord(rec)}
                    className="hover:bg-slate-900/50 cursor-pointer transition-colors group"
                  >
                    <td className="py-3 px-3 font-semibold text-cyan-400">{rec.id}</td>
                    <td className="py-3 px-3 font-sans capitalize text-slate-200">{rec.category}</td>
                    <td className="py-3 px-3 text-slate-300 capitalize">{rec.defect_type}</td>
                    <td className="py-3 px-3 font-bold text-white">
                      {typeof rec.severity_score === 'number' ? rec.severity_score.toFixed(1) : rec.severity_score}
                    </td>
                    <td className="py-3 px-3">
                      <SeverityBadge level={rec.severity_level} showScore={false} size="sm" />
                    </td>
                    <td className="py-3 px-3 text-slate-300">
                      {typeof rec.defect_area_pct === 'number' ? `${rec.defect_area_pct.toFixed(1)}%` : `${rec.defect_area_pct}%`}
                    </td>
                    <td className="py-3 px-3 text-slate-300">
                      {typeof rec.confidence_pct === 'number' ? `${rec.confidence_pct.toFixed(1)}%` : `${rec.confidence_pct}%`}
                    </td>
                    <td className="py-3 px-3 text-slate-400">
                      {typeof rec.anomaly_norm_0_100 === 'number' ? rec.anomaly_norm_0_100.toFixed(1) : rec.anomaly_norm_0_100}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        type="button"
                        aria-label={`Inspect ${rec.id}`}
                        className="p-1.5 rounded-lg bg-slate-900 group-hover:bg-cyan-500/20 text-slate-400 group-hover:text-cyan-300 border border-slate-800 transition-all"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-slate-400">
          <div>
            Page <span className="text-white font-bold">{page}</span> of{' '}
            <span className="text-white font-bold">{totalPages}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={page <= 1 || loading}
              onClick={() => setPage(page - 1)}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>

            <button
              type="button"
              disabled={page >= totalPages || loading}
              onClick={() => setPage(page + 1)}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <span>Next</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Row Click: Detailed Inspection Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md animate-in fade-in duration-200">
          <div className="relative w-full max-w-2xl rounded-2xl border border-slate-700 bg-[#0d111b] p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
            
            {/* Modal Header */}
            <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-4 mb-5">
              <div className="flex items-center gap-3">
                <span className="font-mono text-base font-bold text-cyan-400">{selectedRecord.id}</span>
                <span className="font-sans font-semibold text-white capitalize text-lg">
                  {selectedRecord.category} • {selectedRecord.defect_type}
                </span>
                <SeverityBadge
                  level={selectedRecord.severity_level}
                  score={selectedRecord.severity_score}
                  size="sm"
                />
              </div>

              <button
                type="button"
                onClick={() => setSelectedRecord(null)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Specimen Visual */}
            {selectedRecord.display_image_url && (
              <div className="mb-5 rounded-xl overflow-hidden border border-slate-800 bg-black/60 aspect-video flex items-center justify-center relative">
                <img
                  src={selectedRecord.display_image_url}
                  alt={selectedRecord.filename}
                  className="max-h-full object-contain"
                />
                <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded bg-slate-950/80 text-xs font-mono text-cyan-300 border border-slate-700">
                  Defect Specimen & PatchCore Heatmap
                </div>
              </div>
            )}

            {/* Metrics Breakdown */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mb-5">
              <MetricCard
                title="Severity"
                value={selectedRecord.severity_score}
                unit="/100"
                accentColor="#ef4444"
              />
              <MetricCard
                title="Confidence"
                value={selectedRecord.confidence_pct}
                unit="%"
                isPercent={true}
                accentColor="#8b5cf6"
              />
              <MetricCard
                title="Defect Area"
                value={selectedRecord.defect_area_pct}
                unit="%"
                isPercent={true}
                accentColor="#06b6d4"
              />
              <MetricCard
                title="Anomaly Score"
                value={selectedRecord.raw_patchcore_score}
                accentColor="#38bdf8"
              />
            </div>

            {/* Verified Benchmark Explanation */}
            <div className="p-4 rounded-xl bg-slate-900/90 border border-violet-500/30 text-xs leading-relaxed space-y-2 mb-5">
              <div className="flex items-center gap-1.5 text-violet-300 font-semibold uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                <span>Benchmark Operator Explanation</span>
              </div>
              <p className="text-slate-200">
                {selectedRecord.explanation ||
                  `The automated vision pipeline verified a ${selectedRecord.severity_level} ${selectedRecord.defect_type} anomaly covering ${selectedRecord.defect_area_pct}% of the ${selectedRecord.category} component surface.`}
              </p>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
              <button
                type="button"
                onClick={() => setSelectedRecord(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white text-xs font-semibold"
              >
                Close
              </button>
            </div>

          </div>
        </div>
      )}
    </section>
  );
}

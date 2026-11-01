import React, { useState, useEffect, useRef } from 'react';
import {
  Layers,
  UploadCloud,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  FileCheck,
  BarChart3,
  ListOrdered,
  Clock,
  Download
} from 'lucide-react';
import { api } from '../../api/client';
import { CATEGORIES } from '../../utils/categories';
import SeverityBadge from '../ui/SeverityBadge';
import MetricCard from '../ui/MetricCard';

export default function BatchInspectionSection() {
  const [category, setCategory] = useState('bottle');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [batchStatus, setBatchStatus] = useState(null);
  const [activeReport, setActiveReport] = useState(null);
  const [pastBatches, setPastBatches] = useState([]);
  const [polling, setPolling] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const fileInputRef = useRef(null);

  // Load past batches
  useEffect(() => {
    async function loadPastBatches() {
      try {
        const list = await api.getBatchList();
        setPastBatches(list);

        // If demo batch exists and no active report, load it by default for instant gratification!
        if (list.length > 0 && !activeReport) {
          loadBatchReport(list[0].batch_id);
        }
      } catch (err) {
        console.warn('Could not load past batches:', err);
      }
    }
    loadPastBatches();
  }, []);

  const loadBatchReport = async (id) => {
    try {
      const res = await api.getBatchStatus(id);
      if (res && res.report) {
        setActiveReport(res.report);
        setBatchId(id);
      }
    } catch (err) {
      console.error('Failed to load batch report:', err);
    }
  };

  const handleFilesSelect = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setSelectedFiles(files);
    setErrorMessage(null);
  };

  const handleStartBatch = async () => {
    if (selectedFiles.length === 0) {
      setErrorMessage('Please select one or more images to run batch inspection.');
      return;
    }

    setErrorMessage(null);
    setBatchStatus({ status: 'queued', progress: 0, total: selectedFiles.length });
    setActiveReport(null);
    setPolling(true);

    try {
      const res = await api.submitBatch(selectedFiles, category);
      setBatchId(res.batch_id);

      // Start polling
      pollBatch(res.batch_id);
    } catch (err) {
      console.error('Batch submission failed:', err);
      setErrorMessage(err.detail || err.message || 'Failed to submit batch job.');
      setPolling(false);
    }
  };

  const pollBatch = (id) => {
    const interval = setInterval(async () => {
      try {
        const statusData = await api.getBatchStatus(id);
        setBatchStatus(statusData);

        if (statusData.status === 'completed') {
          clearInterval(interval);
          setPolling(false);
          setActiveReport(statusData.report);
          // Refresh past batches list
          api.getBatchList().then(setPastBatches).catch(() => {});
        } else if (statusData.status.startsWith('failed')) {
          clearInterval(interval);
          setPolling(false);
          setErrorMessage(statusData.status);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1500);
  };

  const handleDownloadBatchCSV = () => {
    if (!activeReport?.results || activeReport.results.length === 0) return;
    const headers = ['filename', 'category', 'predicted_label', 'severity_level', 'severity_score', 'confidence_pct'];
    const rows = activeReport.results.map((r, i) => [
      `"${(r.filename || `Image_${i + 1}`).replace(/"/g, '""')}"`,
      r.category || '',
      r.predicted_label || '',
      r.severity_level || '',
      r.severity_score ?? '',
      r.confidence_pct ?? ''
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `batch_${activeReport.batch_id || 'report'}_results.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadBatchJSON = () => {
    if (!activeReport) return;
    const jsonStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(activeReport, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', jsonStr);
    link.setAttribute('download', `batch_${activeReport.batch_id || 'report'}.json`);
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
            <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-violet-500/10 text-violet-400 border border-violet-500/20">
              High-Throughput (P1)
            </span>
            <span className="text-xs font-mono text-slate-400">• Asynchronous Background Queue</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
            Batch Upload & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Multi-Specimen Inspection</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Process high-volume industrial inspection batches asynchronously. Live polling reports progress, defect rates, and severity distribution.
          </p>
        </div>

        {/* Category selector */}
        <div className="flex flex-col gap-1 min-w-[240px]">
          <label htmlFor="batch-category-select" className="text-xs font-semibold text-slate-300">Target Category</label>
          <select
            id="batch-category-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            disabled={polling}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label} ({c.type})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Upload Dropzone & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-12">
        
        {/* Upload Zone (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700/80 hover:border-cyan-400/80 rounded-2xl p-6 bg-slate-950/60 hover:bg-slate-900/40 text-center cursor-pointer transition-all"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFilesSelect}
              className="hidden"
            />
            <div className="w-12 h-12 rounded-xl bg-violet-500/10 border border-violet-500/30 flex items-center justify-center text-cyan-400 mx-auto mb-3">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div className="text-sm font-semibold text-slate-200">
              Select multiple specimen images
            </div>
            <p className="text-xs text-slate-400 mt-1">Batch queue supports up to 200 files simultaneously</p>
            {selectedFiles.length > 0 && (
              <div className="mt-3 px-3 py-1 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-xs font-mono inline-block">
                {selectedFiles.length} files selected
              </div>
            )}
          </div>

          {errorMessage && (
            <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
              <AlertOctagon className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          <button
            type="button"
            disabled={selectedFiles.length === 0 || polling}
            onClick={handleStartBatch}
            className={`flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-semibold text-sm transition-all shadow-lg ${
              polling
                ? 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white shadow-[0_0_20px_rgba(139,92,246,0.35)]'
            }`}
          >
            {polling ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                <span>Processing Batch Queue ({batchStatus?.progress || 0} / {batchStatus?.total || selectedFiles.length})...</span>
              </>
            ) : (
              <>
                <Layers className="w-4 h-4" />
                <span>Submit Batch ({selectedFiles.length} Images)</span>
              </>
            )}
          </button>

          {/* Past Batches Selector */}
          {pastBatches.length > 0 && (
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                <span>Inspect Past Completed Batches</span>
              </div>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {pastBatches.map((b) => (
                  <button
                    key={b.batch_id}
                    type="button"
                    onClick={() => loadBatchReport(b.batch_id)}
                    className={`w-full flex items-center justify-between p-2 rounded-lg text-xs font-mono transition-colors text-left ${
                      batchId === b.batch_id
                        ? 'bg-violet-600/25 border border-cyan-500/40 text-cyan-300'
                        : 'bg-slate-950 hover:bg-slate-850 text-slate-300 border border-slate-850'
                    }`}
                  >
                    <span className="truncate max-w-[140px] font-semibold">{b.batch_id}</span>
                    <span className="text-slate-400">
                      {b.total_inspected} items ({b.defect_rate_pct}% def)
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Batch Report View (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {activeReport ? (
            <div className="space-y-6 animate-in fade-in duration-300">
              
              {/* Batch Summary Metrics */}
              <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 backdrop-blur-xl">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 border-b border-slate-800 pb-3">
                  <div>
                    <div className="text-[10px] font-mono uppercase text-slate-400">Batch Inspection Summary</div>
                    <h3 className="text-lg font-bold text-white font-mono">{activeReport.batch_id}</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-cyan-400 mr-2">{activeReport.timestamp}</span>
                    <button
                      type="button"
                      onClick={handleDownloadBatchCSV}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                      title="Download Batch Results as CSV"
                    >
                      <Download className="w-3.5 h-3.5 text-cyan-400" />
                      <span>CSV</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleDownloadBatchJSON}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                      title="Download Batch Results as JSON"
                    >
                      <Download className="w-3.5 h-3.5 text-violet-400" />
                      <span>JSON</span>
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  <MetricCard
                    title="Inspected"
                    value={activeReport.total_inspected}
                    unit="files"
                    accentColor="#8b5cf6"
                  />
                  <MetricCard
                    title="Defect Rate"
                    value={activeReport.defect_rate_pct}
                    unit="%"
                    isPercent={true}
                    accentColor="#ef4444"
                  />
                  <MetricCard
                    title="Defective"
                    value={activeReport.defective_count}
                    unit="rejects"
                    accentColor="#f59e0b"
                  />
                  <MetricCard
                    title="Nominal"
                    value={activeReport.normal_count}
                    unit="passed"
                    accentColor="#10b981"
                  />
                </div>

                {/* Severity Breakdown Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                    <span>Severity Distribution Breakdown</span>
                    <span className="text-[11px] font-mono text-slate-400">
                      Critical: {activeReport.severity_distribution?.Critical || 0} | Moderate:{' '}
                      {activeReport.severity_distribution?.Moderate || 0} | Minor:{' '}
                      {activeReport.severity_distribution?.Minor || 0}
                    </span>
                  </div>

                  <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden flex">
                    <div
                      className="bg-rose-500 transition-all"
                      style={{
                        width: `${
                          ((activeReport.severity_distribution?.Critical || 0) /
                            (activeReport.total_inspected || 1)) *
                          100
                        }%`,
                      }}
                      title="Critical Severity"
                    />
                    <div
                      className="bg-amber-500 transition-all"
                      style={{
                        width: `${
                          ((activeReport.severity_distribution?.Moderate || 0) /
                            (activeReport.total_inspected || 1)) *
                          100
                        }%`,
                      }}
                      title="Moderate Severity"
                    />
                    <div
                      className="bg-emerald-500 transition-all"
                      style={{
                        width: `${
                          ((activeReport.severity_distribution?.Minor || 0) /
                            (activeReport.total_inspected || 1)) *
                          100
                        }%`,
                      }}
                      title="Minor / Normal"
                    />
                  </div>
                </div>

              </div>

              {/* Per-Item Results Table */}
              {activeReport.results && activeReport.results.length > 0 && (
                <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/70">
                  <div className="p-3 border-b border-slate-800 text-xs font-semibold uppercase text-slate-300">
                    Inspected Items Details ({activeReport.results.length} specimens)
                  </div>
                  <div className="overflow-x-auto max-h-72">
                    <table className="w-full text-left border-collapse text-xs font-mono">
                      <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400">
                        <tr>
                          <th className="py-2 px-3">Filename</th>
                          <th className="py-2 px-3">Category</th>
                          <th className="py-2 px-3">Verdict</th>
                          <th className="py-2 px-3">Severity</th>
                          <th className="py-2 px-3">Confidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {activeReport.results.map((item, idx) => (
                          <tr key={idx} className="hover:bg-slate-900/40">
                            <td className="py-2 px-3 text-slate-300 truncate max-w-[140px]">
                              {item.filename || `Image_${idx + 1}`}
                            </td>
                            <td className="py-2 px-3 capitalize text-slate-400">{item.category}</td>
                            <td className="py-2 px-3">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                                  item.predicted_label === 'defective'
                                    ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
                                    : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                                }`}
                              >
                                {item.predicted_label}
                              </span>
                            </td>
                            <td className="py-2 px-3">
                              <SeverityBadge level={item.severity_level} score={item.severity_score} size="sm" />
                            </td>
                            <td className="py-2 px-3 text-slate-300">
                              {item.confidence_pct}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

            </div>
          ) : (
            <div className="h-full min-h-[380px] rounded-2xl border border-slate-800 bg-slate-950/40 p-8 flex flex-col items-center justify-center text-center">
              <FileCheck className="w-12 h-12 text-slate-600 mb-3" />
              <h4 className="text-base font-bold text-slate-300 mb-1">No Batch Report Loaded</h4>
              <p className="text-xs text-slate-400 max-w-sm">
                Upload files on the left to start a batch run, or select a previously processed batch to examine the aggregate results.
              </p>
            </div>
          )}
        </div>

      </div>
    </section>
  );
}

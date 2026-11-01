import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  TrendingUp,
  BrainCircuit,
  Send,
  RefreshCw,
  Sliders,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { api } from '../../api/client';
import MetricCard from '../ui/MetricCard';

export default function ActiveLearningRCASection() {
  const [sensitivityData, setSensitivityData] = useState([]);
  const [loadingSens, setLoadingSens] = useState(true);

  // RCA chat
  const [rcaQuestion, setRcaQuestion] = useState('');
  const [rcaMessages, setRcaMessages] = useState([
    {
      role: 'assistant',
      content:
        'I am the Root Cause Pattern Analyzer. Ask me about confidence-severity correlation, high-severity low-confidence outlier cases, or the weakest detection categories across architectures.',
    },
  ]);
  const [askingRca, setAskingRca] = useState(false);

  useEffect(() => {
    async function loadSensitivity() {
      try {
        const data = await api.getActiveLearningSensitivity();
        setSensitivityData(data);
      } catch (err) {
        console.error('Failed to load active learning sensitivity:', err);
      } finally {
        setLoadingSens(false);
      }
    }
    loadSensitivity();
  }, []);

  const handleAskRCA = async (qText) => {
    const q = (qText || rcaQuestion).trim();
    if (!q || askingRca) return;

    const newMsgs = [...rcaMessages, { role: 'user', content: q }];
    setRcaMessages(newMsgs);
    setRcaQuestion('');
    setAskingRca(true);

    try {
      const res = await api.askRCA(q);
      setRcaMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: res.answer,
        },
      ]);
    } catch (err) {
      console.error('RCA query failed:', err);
      setRcaMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: `Error performing RCA: ${err.message || 'Unknown error'}`,
        },
      ]);
    } finally {
      setAskingRca(false);
    }
  };

  const sampleRcaQueries = [
    'What is the correlation between confidence and severity?',
    'Show me high severity cases with low model confidence',
    'How does physical defect area relate to overall severity?',
    'What are the weakest categories for detection?',
  ];

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 border-b border-slate-800/80 pb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            Advanced Intelligence (P2)
          </span>
          <span className="text-xs font-mono text-slate-400">• Active Learning Budgeting & Root Cause Patterns</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
          Active Learning & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Root Cause Insights</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Quantify human-in-the-loop review efficiency curves and interrogate statistical correlations across confidence, area, and defect severity.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-12">
        
        {/* Left Column: Active Learning Efficiency Curve (6 cols) */}
        <div className="lg:col-span-6 flex flex-col gap-5">
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-950/70 backdrop-blur-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-white">Active Learning Efficiency Curve</h3>
              </div>
              <span className="text-xs font-mono text-slate-400">Error Capture vs Review Budget</span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed mb-5">
              By ranking images by model prediction uncertainty, human operators review only a fraction of the total dataset while capturing the vast majority of classification errors.
            </p>

            {loadingSens ? (
              <div className="py-12 text-center text-xs font-mono text-slate-400">
                Loading sensitivity matrix...
              </div>
            ) : (
              <div className="space-y-4">
                {sensitivityData.map((row, idx) => {
                  const budgetPct = parseFloat(row.pct_of_dataset) || 0;
                  const capturePct = parseFloat(row.pct_of_total_errors_captured) || 0;
                  return (
                    <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
                      <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                        <span className="text-slate-300">
                          Review Budget: <strong className="text-cyan-300">{budgetPct.toFixed(1)}%</strong> ({row.n_flagged} images)
                        </span>
                        <span className="text-emerald-400 font-bold">
                          {capturePct.toFixed(1)}% Errors Captured
                        </span>
                      </div>

                      {/* Bar Visualization */}
                      <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden flex gap-0.5">
                        <div
                          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 rounded-full transition-all"
                          style={{ width: `${capturePct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="mt-5 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                <strong>Key Takeaway:</strong> Reviewing just <strong>10.4%</strong> of uncertain images captures over <strong>82.5%</strong> of all false-positives and false-negatives, saving 89% in human review costs.
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: RCA Root Cause Pattern Assistant (6 cols) */}
        <div className="lg:col-span-6 flex flex-col h-[600px] rounded-2xl border border-slate-800 bg-slate-950/80 backdrop-blur-xl shadow-2xl overflow-hidden">
          
          <div className="p-4 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-violet-400" />
              <div>
                <div className="text-xs font-bold text-white">Root Cause Pattern Analyzer</div>
                <div className="text-[10px] font-mono text-slate-400">Statistical Outliers & Architecture Correlations</div>
              </div>
            </div>
          </div>

          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {rcaMessages.map((m, idx) => (
              <div
                key={idx}
                className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-br-xs'
                      : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-xs'
                  }`}
                >
                  <p>{m.content}</p>
                </div>
              </div>
            ))}

            {askingRca && (
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 py-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Interrogating correlation matrices...</span>
              </div>
            )}
          </div>

          {/* Quick RCA questions */}
          <div className="p-3 border-t border-slate-800 bg-slate-900/40 overflow-x-auto">
            <div className="text-[10px] font-mono uppercase text-slate-400 mb-1.5">Pattern Inquiries</div>
            <div className="flex items-center gap-1.5 whitespace-nowrap">
              {sampleRcaQueries.map((q, qIdx) => (
                <button
                  key={qIdx}
                  type="button"
                  onClick={() => handleAskRCA(q)}
                  className="px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-750 text-slate-300 text-[11px] border border-slate-700/60 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleAskRCA();
            }}
            className="p-3 border-t border-slate-800 bg-slate-900/80 flex items-center gap-2"
          >
            <input
              type="text"
              value={rcaQuestion}
              onChange={(e) => setRcaQuestion(e.target.value)}
              placeholder="Ask about correlation, high severity low confidence, outliers..."
              className="flex-1 px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
            <button
              type="submit"
              disabled={!rcaQuestion.trim() || askingRca}
              className="p-2 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 text-white disabled:opacity-40 hover:from-violet-500 hover:to-cyan-400 transition-all shadow-md"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>

        </div>

      </div>
    </section>
  );
}

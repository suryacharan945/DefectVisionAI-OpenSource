import React, { useState } from 'react';
import {
  BarChart3,
  Send,
  Sparkles,
  CheckCircle2,
  HelpCircle,
  TrendingUp,
  PieChart,
  ShieldCheck,
  RefreshCw,
  Cpu
} from 'lucide-react';
import { api } from '../../api/client';

export default function ConversationalDashboardSection() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Welcome to Ask the Data! I compute real-time statistical analytics over the 1,725-benchmark dataset. Ask me about defect rates, model rankings, category weaknesses, severity distributions, or production drift.',
      intent: 'welcome',
      source: 'system',
    },
  ]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);

  const sampleQueries = [
    { label: 'Most common defect in metal_nut?', query: 'What is the most common defect in metal_nut?' },
    { label: 'Defect rates by category', query: 'Show me the defect rates across all product categories' },
    { label: 'Overall severity breakdown', query: 'What is the severity breakdown and average score?' },
    { label: 'Rank models by performance', query: 'Rank all models by mean category AUROC' },
    { label: 'Worst category for PatchCore', query: 'Which category is the weakest for PatchCore?' },
    { label: 'Production drift summary', query: 'Summarize production stream drift alerts' },
    { label: 'Active learning with 10% budget', query: 'What is active learning efficiency with a 10% review budget?' },
  ];

  const handleSend = async (queryText) => {
    const q = (queryText || question).trim();
    if (!q || loading) return;

    const newMsgs = [...messages, { role: 'user', content: q }];
    setMessages(newMsgs);
    setQuestion('');
    setLoading(true);

    try {
      const res = await api.askDashboard(q);
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: res.answer,
          intent: res.intent,
          source: res.source,
        },
      ]);
    } catch (err) {
      console.error('Dashboard ask failed:', err);
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: `Analytics query error: ${err.message || 'Unable to compute analytic.'}`,
          intent: 'error',
          source: 'error',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 border-b border-slate-800/80 pb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-violet-500/10 text-violet-400 border border-violet-500/20">
            Analytics Engine (P1)
          </span>
          <span className="text-xs font-mono text-slate-400">• Grounded in Benchmark CSV Records</span>
        </div>
        <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
          Ask the Data & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Conversational Analytics</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Ask questions in natural English. Intent routing executes exact dataset analytics backed by <code className="text-cyan-400">numeric_validator</code> to eliminate hallucination.
        </p>
      </div>

      <div className="max-w-4xl mx-auto flex flex-col h-[650px] rounded-2xl border border-slate-800 bg-slate-950/80 backdrop-blur-xl shadow-2xl overflow-hidden">
        
        {/* Chat Bar Header */}
        <div className="p-4 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <div className="text-sm font-bold text-white">Dataset Analytics Copilot</div>
              <div className="text-[10px] font-mono text-slate-400">Grounded in 1,725 Images • Intent-Routed Computations</div>
            </div>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-700/80 text-[10px] font-mono text-slate-300">
            <CheckCircle2 className="w-3 h-3 text-cyan-400" />
            <span>Exact Numeric Grounding</span>
          </div>
        </div>

        {/* Message Thread */}
        <div className="flex-1 p-5 overflow-y-auto space-y-4">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl p-4 text-xs leading-relaxed ${
                  m.role === 'user'
                    ? 'bg-gradient-to-r from-violet-600 to-cyan-600 text-white rounded-br-xs shadow-md'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-bl-xs'
                }`}
              >
                <p className="text-sm">{m.content}</p>

                {m.intent && m.intent !== 'welcome' && (
                  <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span className="capitalize">Intent: {m.intent.replace(/_/g, ' ')}</span>
                    <span className="text-cyan-400 font-semibold">Source: {m.source}</span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 py-3">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Routing intent & computing exact metrics from benchmark logs...</span>
            </div>
          )}
        </div>

        {/* Quick Analytical Query Chips */}
        <div className="p-3 border-t border-slate-800 bg-slate-900/40 overflow-x-auto">
          <div className="text-[10px] font-mono uppercase text-slate-400 mb-1.5">Common Questions</div>
          <div className="flex items-center gap-2 whitespace-nowrap">
            {sampleQueries.map((sq, sIdx) => (
              <button
                key={sIdx}
                type="button"
                onClick={() => handleSend(sq.query)}
                className="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-750 text-slate-300 hover:text-white text-xs border border-slate-700/70 hover:border-cyan-500/40 transition-colors"
              >
                {sq.label}
              </button>
            ))}
          </div>
        </div>

        {/* Chat Input */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="p-3 border-t border-slate-800 bg-slate-900/80 flex items-center gap-2"
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything about defect rates, model rankings, drift..."
            className="flex-1 px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
          <button
            type="submit"
            disabled={!question.trim() || loading}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 text-white font-medium text-xs disabled:opacity-40 hover:from-violet-500 hover:to-cyan-400 transition-all shadow-md flex items-center gap-1.5"
          >
            <Send className="w-3.5 h-3.5" />
            <span>Ask</span>
          </button>
        </form>

      </div>
    </section>
  );
}

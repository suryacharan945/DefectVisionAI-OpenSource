import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  MessageSquare,
  Send,
  Sparkles,
  HelpCircle,
  Search,
  ShieldCheck,
  AlertTriangle,
  Lightbulb,
  CheckCircle2,
  RefreshCw
} from 'lucide-react';
import { api } from '../../api/client';

export default function KnowledgeBaseRAGSection({ initialQuery = '' }) {
  const [entries, setEntries] = useState([]);
  const [filteredEntries, setFilteredEntries] = useState([]);
  const [searchKb, setSearchKb] = useState('');
  const [loadingKb, setLoadingKb] = useState(true);

  // Chat states
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I am your Industrial Defect Copilot, grounded in the 16 MVTec defect taxonomies. Ask me anything about defect root causes, process impact, repair solutions, or prevention protocols.',
      retrieved_defect_types: [],
    },
  ]);
  const [question, setQuestion] = useState(initialQuery);
  const [asking, setAsking] = useState(false);

  // Load KB entries from /rag/entries
  useEffect(() => {
    async function loadKb() {
      try {
        const data = await api.getRagEntries();
        setEntries(data);
        setFilteredEntries(data);
      } catch (err) {
        console.error('Failed to load RAG knowledge base:', err);
      } finally {
        setLoadingKb(false);
      }
    }
    loadKb();
  }, []);

  // Filter KB cards
  useEffect(() => {
    if (!searchKb) {
      setFilteredEntries(entries);
      return;
    }
    const q = String(searchKb || '').toLowerCase();
    setFilteredEntries(
      entries.filter(
        (e) =>
          e.defect_type?.toLowerCase().includes(q) ||
          e.cause?.toLowerCase().includes(q) ||
          e.solution?.toLowerCase().includes(q)
      )
    );
  }, [searchKb, entries]);

  const handleAsk = async (queryToAsk) => {
    const q = (queryToAsk || question).trim();
    if (!q || asking) return;

    // Append user message
    const newMsgs = [...messages, { role: 'user', content: q }];
    setMessages(newMsgs);
    setQuestion('');
    setAsking(true);

    try {
      const res = await api.askRAG(q);
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: res.answer,
          retrieved_defect_types: res.retrieved_defect_types || [],
        },
      ]);
    } catch (err) {
      console.error('RAG query failed:', err);
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: `Error retrieving answer: ${err.message || 'Unable to complete RAG search.'}`,
          retrieved_defect_types: [],
        },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const samplePrompts = [
    'What causes contamination in pharmaceutical pills?',
    'How do we prevent scratches on metal nuts?',
    'What is the operational impact of broken bottle sidewalls?',
    'How to resolve thread defects in metal grids?',
  ];

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Grounded AI (P1)
            </span>
            <span className="text-xs font-mono text-slate-400">• 16 Defect Taxonomies • SentenceTransformer Index</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
            Knowledge Base & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">RAG Defect Copilot</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Semantic retrieval across industrial causes, operational impacts, corrective solutions, and prevention protocols.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-12">
        
        {/* Left Column: Interactive Chat Interface (5 cols) */}
        <div className="lg:col-span-5 flex flex-col h-[650px] rounded-2xl border border-slate-800 bg-slate-950/80 backdrop-blur-xl shadow-2xl overflow-hidden">
          
          {/* Chat Header */}
          <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center text-cyan-400">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold text-slate-100">Defect Assistant Copilot</div>
                <div className="text-[10px] text-slate-400 font-mono">Semantic RAG + Grounded Verification</div>
              </div>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Ready
            </span>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-br-xs shadow-md'
                      : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-xs'
                  }`}
                >
                  <p>{m.content}</p>

                  {/* Retrieved defect sources */}
                  {m.retrieved_defect_types && m.retrieved_defect_types.length > 0 && (
                    <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex items-center gap-1.5 flex-wrap">
                      <span className="text-[10px] font-mono text-slate-400">Sources:</span>
                      {m.retrieved_defect_types.map((src, sIdx) => (
                        <span
                          key={sIdx}
                          className="px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 text-[10px] font-mono capitalize"
                        >
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {asking && (
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 py-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Searching knowledge base vectors...</span>
              </div>
            )}
          </div>

          {/* Quick Suggestions */}
          <div className="p-3 border-t border-slate-800/60 bg-slate-900/30 overflow-x-auto">
            <div className="text-[10px] font-mono uppercase text-slate-400 mb-1.5">Suggested Prompts</div>
            <div className="flex items-center gap-1.5 whitespace-nowrap">
              {samplePrompts.map((p, pIdx) => (
                <button
                  key={pIdx}
                  type="button"
                  onClick={() => handleAsk(p)}
                  className="px-2.5 py-1 rounded-md bg-slate-800/70 hover:bg-slate-750 text-slate-300 text-[11px] border border-slate-700/60 transition-colors"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          {/* Chat Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleAsk();
            }}
            className="p-3 border-t border-slate-800 bg-slate-900/70 flex items-center gap-2"
          >
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about defect causes, solutions, prevention..."
              className="flex-1 px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
            <button
              type="submit"
              disabled={!question.trim() || asking}
              className="p-2 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 text-white disabled:opacity-40 hover:from-violet-500 hover:to-cyan-400 transition-all shadow-md"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>

        </div>

        {/* Right Column: 16 Defect Knowledge Cards Grid (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          
          {/* Card Search & Stats */}
          <div className="flex items-center justify-between gap-3 p-3 rounded-xl bg-slate-950/70 border border-slate-800">
            <div className="relative flex-1">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchKb}
                onChange={(e) => setSearchKb(e.target.value)}
                placeholder="Filter 16 defect cards by name, cause..."
                className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>
            <span className="text-xs font-mono text-slate-400">
              {filteredEntries.length} of {entries.length} Taxonomies
            </span>
          </div>

          {/* Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[580px] overflow-y-auto pr-1">
            {loadingKb ? (
              <div className="col-span-2 py-16 text-center text-slate-400 font-mono text-xs">
                Loading defect taxonomy cards...
              </div>
            ) : filteredEntries.length === 0 ? (
              <div className="col-span-2 py-16 text-center text-slate-400 text-xs">
                No taxonomy entries match "{searchKb}".
              </div>
            ) : (
              filteredEntries.map((entry) => (
                <div
                  key={entry.defect_type}
                  onClick={() =>
                    handleAsk(`Explain the root cause and prevention of ${entry.defect_type} defect`)
                  }
                  className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:bg-slate-850 hover:border-violet-500/40 transition-all cursor-pointer group flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-heading font-bold text-sm text-cyan-300 capitalize group-hover:text-white transition-colors">
                        {String(entry.defect_type || '').replace(/_/g, ' ')}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 group-hover:text-cyan-400">
                        Ask Copilot →
                      </span>
                    </div>

                    <div className="space-y-2 text-xs text-slate-300">
                      <div>
                        <span className="text-slate-400 font-semibold block text-[11px]">Probable Cause:</span>
                        <p className="line-clamp-2 text-slate-300">{entry.cause}</p>
                      </div>

                      <div>
                        <span className="text-slate-400 font-semibold block text-[11px]">Corrective Action:</span>
                        <p className="line-clamp-2 text-slate-300">{entry.solution}</p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>Impact: {entry.impact ? 'Documented' : 'N/A'}</span>
                    <span className="text-violet-400 font-semibold">1-Click Query</span>
                  </div>
                </div>
              ))
            )}
          </div>

        </div>

      </div>
    </section>
  );
}

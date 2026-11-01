import React, { useState, useEffect } from 'react';
import {
  Search,
  Layers,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sparkles,
  ExternalLink
} from 'lucide-react';
import { api } from '../../api/client';
import { CATEGORIES } from '../../utils/categories';

export default function SimilarityExplorerSection({ initialCategory = 'bottle' }) {
  const [category, setCategory] = useState(initialCategory);
  const [gallery, setGallery] = useState([]);
  const [matches, setMatches] = useState([]);
  const [queryFile, setQueryFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingGallery, setLoadingGallery] = useState(true);

  // Load gallery for selected category
  useEffect(() => {
    async function loadGallery() {
      setLoadingGallery(true);
      try {
        const res = await api.getCategoryGallery(category);
        setGallery(res.gallery || []);
      } catch (err) {
        console.warn('Could not load gallery:', err);
      } finally {
        setLoadingGallery(false);
      }
    }
    loadGallery();
  }, [category]);

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setQueryFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);

    try {
      const res = await api.searchSimilarity(file, category);
      setMatches(res.matches || []);
    } catch (err) {
      console.error('Similarity search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2.5 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Vector Space (P1)
            </span>
            <span className="text-xs font-mono text-slate-400">• FAISS L2 Embedding Search (1280-dim)</span>
          </div>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
            Similarity Explorer & <span className="bg-gradient-to-r from-violet-400 to-cyan-300 bg-clip-text text-transparent">Vector Retrieval</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Query 1280-dimensional penultimate representations from EfficientNet-B0 to retrieve nearest neighbor historical defects in milliseconds.
          </p>
        </div>

        {/* Category selector */}
        <div className="flex flex-col gap-1 min-w-[220px]">
          <label htmlFor="similarity-category-select" className="text-xs font-semibold text-slate-300">Category Index</label>
          <select
            id="similarity-category-select"
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
              setMatches([]);
            }}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Query Image Upload (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-950/70 backdrop-blur-xl">
            <h3 className="text-sm font-bold text-white mb-1">Query Specimen</h3>
            <p className="text-xs text-slate-400 mb-4">
              Upload an image to compute its feature vector and retrieve the 5 most similar historical specimens.
            </p>

            <label htmlFor="similarity-upload-input" className="block border-2 border-dashed border-slate-700 hover:border-cyan-400 rounded-xl p-4 text-center cursor-pointer transition-all bg-slate-900/50">
              <input
                id="similarity-upload-input"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              {previewUrl ? (
                <div className="flex flex-col items-center gap-2">
                  <img src={previewUrl} alt="Query Specimen" className="max-h-48 object-contain rounded-lg" />
                  <span className="text-[11px] font-mono text-cyan-300 underline">Change Image</span>
                </div>
              ) : (
                <div className="py-6 flex flex-col items-center">
                  <UploadCloud className="w-8 h-8 text-cyan-400 mb-2" />
                  <span className="text-xs font-semibold text-slate-200">Upload to Search Vector Gallery</span>
                  <span className="text-[10px] text-slate-400 mt-1">PNG, JPG, TIFF</span>
                </div>
              )}
            </label>

            {loading && (
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 mt-4 py-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Extracting 1280-dim embedding & querying FAISS...</span>
              </div>
            )}
          </div>

          {/* Quick info note */}
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 text-xs text-slate-400 leading-relaxed">
            <span className="font-semibold text-cyan-300 block mb-1">How Similarity Search Works:</span>
            Global average pooling of EfficientNet-B0 penultimate activations captures fine spatial textures. FAISS executes hardware-accelerated L2 Euclidean distance searches against the indexed gallery.
          </div>
        </div>

        {/* Right Column: Nearest Matches or Category Gallery (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          
          {matches.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Top 5 Nearest Historical Matches
                  </h3>
                </div>
                <span className="text-xs font-mono text-slate-400">Ranked by L2 Distance</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {matches.map((m, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl border border-slate-800 bg-slate-950/70 hover:border-cyan-500/40 transition-all flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                          #{idx + 1} Match
                        </span>
                        <span className="text-[11px] font-mono text-slate-400">
                          dist: {m.l2_distance ? m.l2_distance.toFixed(3) : '0.00'}
                        </span>
                      </div>

                      <div className="w-full aspect-square rounded-lg bg-black/50 border border-slate-800 flex items-center justify-center overflow-hidden mb-3">
                        <img
                          src={`/outputs/similarity_search_samples/${m.category}_similarity_demo.png`}
                          alt={m.defect_type}
                          className="w-full h-full object-contain"
                        />
                      </div>

                      <div className="text-xs font-sans">
                        <div className="text-slate-400 text-[10px] uppercase font-mono">Defect Type</div>
                        <div className="font-semibold text-slate-100 capitalize">
                          {m.defect_type || 'Unknown'}
                        </div>
                      </div>
                    </div>

                    <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                      <span>Status: {m.is_normal ? 'Nominal' : 'Defective'}</span>
                      <span className="text-cyan-400">Historical Specimen</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Category Representative Gallery */
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-violet-400" />
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    {category.replace('_', ' ')} FAISS Index Gallery Preview
                  </h3>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  {gallery.length} Sample Vectors
                </span>
              </div>

              {loadingGallery ? (
                <div className="py-20 text-center text-xs font-mono text-slate-400">
                  Loading gallery entries...
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {gallery.map((item, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl border border-slate-800/80 bg-slate-950/60 hover:bg-slate-900/50 transition-colors"
                    >
                      <div className="w-full aspect-square rounded-lg bg-black/40 border border-slate-850 flex items-center justify-center overflow-hidden mb-2">
                        <img
                          src={item.sample_url}
                          alt={item.defect_type}
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <div className="text-xs font-sans">
                        <span className="text-[10px] font-mono text-slate-400 block truncate">
                          {item.filename || `Sample_${idx + 1}`}
                        </span>
                        <span className="font-semibold text-slate-200 capitalize block truncate">
                          {item.defect_type === 'none' ? 'Nominal specimen' : item.defect_type}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>

      </div>
    </section>
  );
}

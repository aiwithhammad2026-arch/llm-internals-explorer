import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { EmbeddingResponse, TokenEmbeddingItem } from '../types/api';
import { Badge } from './Badge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { Play, Binary, AlertCircle, Info } from 'lucide-react';

interface EmbeddingViewProps {
  initialPrompt?: string;
}

export const EmbeddingView: React.FC<EmbeddingViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [data, setData] = useState<EmbeddingResponse | null>(null);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFetchEmbeddings = async (inputText: string) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getEmbeddings(inputText, 16);
      setData(res);
      setSelectedTokenIdx(0);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to extract embeddings');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleFetchEmbeddings(initialPrompt);
  }, []);

  const currentToken: TokenEmbeddingItem | undefined = data?.tokens[selectedTokenIdx];

  // Prepare chart data for first 16 dimensions
  const chartData = currentToken
    ? Array.from({ length: data?.slice_size || 16 }).map((_, dim) => ({
        dimension: `d${dim}`,
        Token: currentToken.token_embedding_slice[dim],
        Positional: currentToken.positional_embedding_slice[dim],
        Combined: currentToken.combined_embedding_slice[dim],
      }))
    : [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Embedding Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Learned Token Embeddings (<code className="text-emerald-400">wte</code>) + Positional Embeddings (
            <code className="text-purple-400">wpe</code>) entering Transformer Block 0
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleFetchEmbeddings(text)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleFetchEmbeddings(text)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Extracting...' : 'Extract Embeddings'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <strong className="font-semibold">Backend Error:</strong>
            <p className="mt-0.5 text-xs text-rose-200">{error}</p>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!data && !loading && !error && (
        <div className="p-10 text-center rounded-xl bg-slate-900/50 border border-dashed border-slate-800 space-y-3">
          <Binary className="w-8 h-8 mx-auto text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-300">No Embeddings Extracted Yet</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Enter prompt text above and click <strong className="text-slate-400">Extract Embeddings</strong> to inspect token vectors (wte), position vectors (wpe), and their combined sum.
          </p>
        </div>
      )}

      {/* Data display */}
      {data && currentToken && (
        <div className="space-y-6">
          {/* Shape Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Token Embedding Shape (wte)</span>
              <div className="text-lg font-mono font-bold text-emerald-400 mt-1">
                [{data.token_embedding_shape.join(', ')}]
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Batch × SeqLen × HiddenDim (768)</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Positional Embedding Shape (wpe)</span>
              <div className="text-lg font-mono font-bold text-purple-400 mt-1">
                [{data.positional_embedding_shape.join(', ')}]
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Positions 0 to {data.token_count - 1}</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Combined Sum (wte + wpe)</span>
              <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                [{data.combined_embedding_shape.join(', ')}]
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Equals hidden_states[0] in eval mode</p>
            </div>
          </div>

          {/* Token Selector */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Select Token in Sequence to Inspect Vectors:
            </label>
            <div className="flex flex-wrap gap-2">
              {data.tokens.map((tok) => {
                const isSelected = selectedTokenIdx === tok.index;
                return (
                  <button
                    key={tok.index}
                    onClick={() => setSelectedTokenIdx(tok.index)}
                    className={`px-3 py-1.5 rounded-lg border text-sm font-mono transition-all flex items-center gap-2 ${
                      isSelected
                        ? 'bg-indigo-600/30 border-indigo-400 text-indigo-200 shadow-md shadow-indigo-500/10'
                        : 'bg-slate-950 hover:bg-slate-800 border-slate-700 text-slate-300'
                    }`}
                  >
                    <span className="text-slate-400 text-[10px]">Pos {tok.index}</span>
                    <strong className="text-amber-300">{JSON.stringify(tok.token_str)}</strong>
                    <span className="text-[10px] text-slate-400">ID: {tok.token_id}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Bar Chart Visualization (First 16 Dimensions) */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Binary className="w-4 h-4 text-cyan-400" />
                  Vector Dimension Values (First 16 of 768 dimensions for Token {selectedTokenIdx}: {JSON.stringify(currentToken.token_str)})
                </h3>
                <p className="text-xs text-slate-400">
                  Notice: <strong className="text-cyan-400">Combined</strong> vector equals <strong className="text-emerald-400">Token (wte)</strong> + <strong className="text-purple-400">Positional (wpe)</strong> at every single dimension.
                </p>
              </div>
            </div>

            <div className="h-72 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="dimension" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(val: any) => typeof val === 'number' ? val.toFixed(6) : val}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Bar dataKey="Token" fill="#10b981" name="Token Embedding (wte)" />
                  <Bar dataKey="Positional" fill="#a855f7" name="Positional Embedding (wpe)" />
                  <Bar dataKey="Combined" fill="#06b6d4" name="Combined Sum (wte + wpe)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Statistical Moments Table across all 768 Dimensions */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Full 768-Dimensional Summary Statistics (Token {selectedTokenIdx})
              </span>
              <span className="text-xs text-slate-400">
                Calculated across all 768 real float values
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Embedding Part</th>
                    <th className="px-4 py-3">Vector Role</th>
                    <th className="px-4 py-3">Min</th>
                    <th className="px-4 py-3">Max</th>
                    <th className="px-4 py-3">Mean</th>
                    <th className="px-4 py-3">Std Dev</th>
                    <th className="px-4 py-3">L2 Norm (||v||)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  <tr className="hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-semibold text-emerald-400">Word Token (wte)</td>
                    <td className="px-4 py-3 text-slate-400 font-sans">Learned semantic representation</td>
                    <td className="px-4 py-3">{currentToken.token_stats.min.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.token_stats.max.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.token_stats.mean.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.token_stats.std.toFixed(6)}</td>
                    <td className="px-4 py-3 text-emerald-300 font-bold">{currentToken.token_stats.norm.toFixed(6)}</td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-semibold text-purple-400">Word Position (wpe)</td>
                    <td className="px-4 py-3 text-slate-400 font-sans">Learned sequence location (pos {selectedTokenIdx})</td>
                    <td className="px-4 py-3">{currentToken.positional_stats.min.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.positional_stats.max.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.positional_stats.mean.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.positional_stats.std.toFixed(6)}</td>
                    <td className="px-4 py-3 text-purple-300 font-bold">{currentToken.positional_stats.norm.toFixed(6)}</td>
                  </tr>
                  <tr className="hover:bg-slate-800/40 bg-slate-950/40">
                    <td className="px-4 py-3 font-semibold text-cyan-400">Combined Sum</td>
                    <td className="px-4 py-3 text-slate-300 font-sans font-medium">Input to Transformer Block 0</td>
                    <td className="px-4 py-3">{currentToken.combined_stats.min.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.combined_stats.max.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.combined_stats.mean.toFixed(6)}</td>
                    <td className="px-4 py-3">{currentToken.combined_stats.std.toFixed(6)}</td>
                    <td className="px-4 py-3 text-cyan-300 font-bold">{currentToken.combined_stats.norm.toFixed(6)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Educational Callout */}
          <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-900/60 flex items-start gap-3 text-blue-200 text-xs">
            <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <strong className="font-semibold text-blue-300">Architectural Note:</strong>
                <Badge origin={data.explanation_origin || "Educational explanation"} size="sm" />
              </div>
              <p className="mt-1 leading-relaxed text-blue-200/90">{data.explanation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

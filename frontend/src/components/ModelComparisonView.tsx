import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { CompareModelsResponse } from '../types/api';
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
import {
  GitCompare,
  Play,
  Layers,
  AlertCircle,
  Loader2,
  Info,
  Zap,
} from 'lucide-react';

interface ModelComparisonViewProps {
  initialPrompt?: string;
}

export const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [data, setData] = useState<CompareModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async (inputText: string) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.compareModels(inputText, 'distilgpt2', 'gpt2');
      setData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to evaluate models for comparison. Check model loading or memory.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleCompare(initialPrompt);
  }, []);

  // Format dual-bar chart data: top 10 candidates side-by-side
  const chartData = data
    ? Array.from({ length: 10 }).map((_, idx) => {
        const candA = data.model_a.top_candidates[idx];
        const candB = data.model_b.top_candidates[idx];
        return {
          rank: `#${idx + 1}`,
          tokenA: candA ? JSON.stringify(candA.token_str) : '',
          probA: candA ? Math.round(candA.probability * 10000) / 100 : 0,
          tokenB: candB ? JSON.stringify(candB.token_str) : '',
          probB: candB ? Math.round(candB.probability * 10000) / 100 : 0,
        };
      })
    : [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-indigo-400" />
            Model Comparison: distilgpt2 vs gpt2
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Side-by-side architectural and next-token probability benchmarking within the trusted GPT-2 family
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input Box */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Comparison Prompt Context
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCompare(text)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleCompare(text)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Evaluating Both Models...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Compare Models
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <strong className="font-semibold">Comparison Error:</strong>
            <p className="mt-0.5 text-xs text-rose-200">{error}</p>
          </div>
        </div>
      )}

      {/* Loading state note if downloading second model */}
      {loading && !data && (
        <div className="p-8 rounded-xl bg-slate-900 border border-slate-800 text-center space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-400 mx-auto" />
          <h3 className="text-sm font-bold text-slate-200">Evaluating Dual Models...</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Loading weights for <code className="text-cyan-400">distilgpt2</code> and{' '}
            <code className="text-purple-400">gpt2</code>. This takes a brief moment on initial load.
          </p>
        </div>
      )}

      {/* Empty State */}
      {!data && !loading && !error && (
        <div className="p-10 text-center rounded-xl bg-slate-900/50 border border-dashed border-slate-800 space-y-3">
          <GitCompare className="w-8 h-8 mx-auto text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-300">No Model Comparison Run Yet</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Click <strong className="text-slate-400">Compare Models</strong> to evaluate distilgpt2 (~82M) against gpt2 (~124M) side-by-side on architecture, parameter counts, and output distributions.
          </p>
        </div>
      )}

      {/* Comparison Workspace */}
      {data && (
        <div className="space-y-6">
          {/* Side-by-Side Model Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model A: distilgpt2 */}
            <div className="p-5 rounded-xl bg-gradient-to-br from-cyan-950/30 via-slate-900 to-slate-900 border border-cyan-800/60 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div>
                  <h3 className="text-base font-bold text-cyan-300 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-cyan-400" />
                    {data.model_a.model_name}
                  </h3>
                  <span className="text-[11px] text-slate-400">Distilled Knowledge Variant</span>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-cyan-950 border border-cyan-800 text-xs font-mono font-bold text-cyan-300">
                  {data.model_a.param_count_display}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Layers</span>
                  <span className="text-slate-100 font-bold">{data.model_a.num_layers} blocks</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Heads / Dim</span>
                  <span className="text-slate-100">{data.model_a.num_heads} / 64</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Hidden Size</span>
                  <span className="text-slate-100">{data.model_a.hidden_size}</span>
                </div>
              </div>

              {/* Top candidate */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">
                  Top 1 Prediction
                </span>
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-cyan-300 font-bold text-sm">
                    {JSON.stringify(data.model_a.top_1_token_str)}
                  </span>
                  <span className="text-emerald-400 font-bold">
                    {(data.model_a.top_1_probability * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Model B: gpt2 */}
            <div className="p-5 rounded-xl bg-gradient-to-br from-purple-950/30 via-slate-900 to-slate-900 border border-purple-800/60 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div>
                  <h3 className="text-base font-bold text-purple-300 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-purple-400" />
                    {data.model_b.model_name}
                  </h3>
                  <span className="text-[11px] text-slate-400">Full Standard GPT-2 Base</span>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-purple-950 border border-purple-800 text-xs font-mono font-bold text-purple-300">
                  {data.model_b.param_count_display}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Layers</span>
                  <span className="text-slate-100 font-bold">{data.model_b.num_layers} blocks</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Heads / Dim</span>
                  <span className="text-slate-100">{data.model_b.num_heads} / 64</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 font-sans block">Hidden Size</span>
                  <span className="text-slate-100">{data.model_b.hidden_size}</span>
                </div>
              </div>

              {/* Top candidate */}
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">
                  Top 1 Prediction
                </span>
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-purple-300 font-bold text-sm">
                    {JSON.stringify(data.model_b.top_1_token_str)}
                  </span>
                  <span className="text-emerald-400 font-bold">
                    {(data.model_b.top_1_probability * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Configuration Differences Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Architectural Configuration Differences
              </span>
              <span className="text-xs text-slate-400">
                distilgpt2 halves depth from 12 to 6 blocks
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3 font-sans">Architectural Dimension</th>
                    <th className="px-4 py-3 text-cyan-300">distilgpt2</th>
                    <th className="px-4 py-3 text-purple-300">gpt2</th>
                    <th className="px-4 py-3 font-sans">Delta / Impact</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  <tr>
                    <td className="px-4 py-2.5 font-sans font-medium text-slate-200">Total Parameters</td>
                    <td className="px-4 py-2.5 text-cyan-300 font-bold">81,912,576 (~82M)</td>
                    <td className="px-4 py-2.5 text-purple-300 font-bold">124,439,808 (~124M)</td>
                    <td className="px-4 py-2.5 text-amber-300 font-sans">+42.5M params (+51.9% in gpt2)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-sans font-medium text-slate-200">Transformer Layers (Depth)</td>
                    <td className="px-4 py-2.5 text-cyan-300 font-bold">6 blocks</td>
                    <td className="px-4 py-2.5 text-purple-300 font-bold">12 blocks</td>
                    <td className="px-4 py-2.5 text-indigo-300 font-sans">Exact 2x depth reduction in distilgpt2</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-sans font-medium text-slate-200">Attention Heads</td>
                    <td className="px-4 py-2.5">12 heads</td>
                    <td className="px-4 py-2.5">12 heads</td>
                    <td className="px-4 py-2.5 text-slate-400 font-sans">Identical (12 heads)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-sans font-medium text-slate-200">Hidden Dimension</td>
                    <td className="px-4 py-2.5">768 dims</td>
                    <td className="px-4 py-2.5">768 dims</td>
                    <td className="px-4 py-2.5 text-slate-400 font-sans">Identical (768 dims)</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-sans font-medium text-slate-200">Vocabulary Size</td>
                    <td className="px-4 py-2.5">50,257 tokens</td>
                    <td className="px-4 py-2.5">50,257 tokens</td>
                    <td className="px-4 py-2.5 text-slate-400 font-sans">Shared BPE Vocabulary</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Dual Next-Token Candidates Probability Chart */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-bold text-slate-200">
                Top 10 Next-Token Predictions Comparison
              </h3>
              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-1.5 text-cyan-300 font-mono">
                  <span className="w-3 h-3 rounded bg-cyan-500"></span> distilgpt2
                </span>
                <span className="flex items-center gap-1.5 text-purple-300 font-mono">
                  <span className="w-3 h-3 rounded bg-purple-500"></span> gpt2
                </span>
              </div>
            </div>

            <div className="h-72 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="rank" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(val: any, name: any, item: any) => [
                      `${val}% (${name === 'distilgpt2' ? item.payload.tokenA : item.payload.tokenB})`,
                      name,
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="probA" fill="#06b6d4" name="distilgpt2" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="probB" fill="#a855f7" name="gpt2" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Educational Explanation Box */}
          <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-900/60 flex items-start gap-3 text-blue-200 text-xs">
            <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <strong className="font-semibold text-blue-300">
                  Knowledge Distillation & Model Scaling:
                </strong>
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

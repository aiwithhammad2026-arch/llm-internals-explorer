import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { LogitLensResponse, LogitLensLayerStep } from '../types/api';
import { Badge } from './Badge';
import {
  LineChart,
  Line,
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
  Play,
  Eye,
  TrendingUp,
  Activity,
  Layers,
  Sparkles,
  Info,
  AlertCircle,
  CheckCircle2,
  Target,
} from 'lucide-react';

interface LogitLensViewProps {
  initialPrompt?: string;
}

export const LogitLensView: React.FC<LogitLensViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [topK, setTopK] = useState<number>(5);
  const [data, setData] = useState<LogitLensResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const textRef = React.useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  const handleRunLogitLens = async (
    inputText: string,
    targetTokenIdx?: number,
    kCount: number = 5
  ) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const idx = targetTokenIdx === -1 ? undefined : targetTokenIdx;
      const res = await api.getLogitLens(inputText, idx, kCount);
      setData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to compute Logit Lens residual stream evolution.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleRunLogitLens(initialPrompt, -1, 5);
  }, [initialPrompt]);

  // Find emergence layers
  const firstTop5Layer = data?.layers.find((l) => l.target_rank <= 5);
  const firstTop1Layer = data?.layers.find((l) => l.target_rank === 1);

  // Line chart data for Target Probability and Entropy across layers
  const trajectoryChartData = data?.layers.map((l: LogitLensLayerStep) => ({
    layerName: l.layer === 0 ? 'L0 (Embed)' : `L${l.layer} (Block ${l.layer - 1})`,
    targetProb: Math.round(l.target_probability * 10000) / 100, // percentage
    topCandidateProb: Math.round((l.top_candidates[0]?.probability || 0) * 10000) / 100,
    entropy: l.entropy,
    cosineSim: l.cosine_sim_to_previous !== null ? Math.round(l.cosine_sim_to_previous * 100) / 100 : null,
  })) || [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Eye className="w-5 h-5 text-indigo-400" />
            Logit Lens & Residual Stream Evolution
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Apply the unembedding head (<code className="text-cyan-400">ln_f + lm_head</code>) to intermediate hidden states to observe token emergence and entropy collapse
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input and Controls */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleRunLogitLens(text, -1, topK)}
            placeholder="Type a sentence to unembed through the layers..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="text-xs text-slate-400 font-mono hidden sm:inline">Top-K:</span>
            <select
              value={topK}
              onChange={(e) => {
                const k = parseInt(e.target.value, 10);
                setTopK(k);
                handleRunLogitLens(textRef.current, -1, k);
              }}
              className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-2.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
            </select>
          </div>
          <button
            onClick={() => handleRunLogitLens(text, -1, topK)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20 shrink-0"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Unembedding...' : 'Run Logit Lens'}
          </button>
        </div>

        {/* Token Position Ribbon */}
        {data && (
          <div className="pt-2 border-t border-slate-800/80">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
              <span className="text-xs text-slate-400 font-medium">
                Inspecting token position (default is last token for next-token prediction):
              </span>
              <span className="text-xs font-mono text-cyan-400">
                Selected: position {data.selected_token_index} ({JSON.stringify(data.selected_token_str)})
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Array.from({ length: data.token_count }).map((_, idx) => {
                const isSelected = idx === data.selected_token_index;
                const isLast = idx === data.token_count - 1;
                return (
                  <button
                    key={idx}
                    onClick={() => {
                      handleRunLogitLens(textRef.current, idx, topK);
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                      isSelected
                        ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400'
                        : 'bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-800'
                    }`}
                  >
                    <span>Pos {idx}</span>
                    {isLast && (
                      <span className="text-[10px] px-1 py-0.2 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/50">
                        Next-Token
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-900/60 flex items-start gap-3 text-red-200 text-sm">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Logit Lens Error</p>
            <p className="text-xs text-red-300/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !data && !error && (
        <div className="p-12 text-center rounded-xl bg-slate-900/40 border border-slate-800/80 text-slate-400">
          <Eye className="w-10 h-10 mx-auto mb-3 text-slate-600 animate-pulse" />
          <p className="text-sm font-medium text-slate-300">Ready to Unembed Residual States</p>
          <p className="text-xs text-slate-500 mt-1">
            Click &quot;Run Logit Lens&quot; above to project intermediate layer hidden states onto the vocabulary.
          </p>
        </div>
      )}

      {/* Live Data Display */}
      {data && (
        <>
          {/* Key Metrics Overview Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-indigo-950/80 border border-indigo-700/40 text-indigo-400">
                <Target className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Final Output Token</p>
                <p className="text-base font-bold font-mono text-slate-100 flex items-center gap-1.5 mt-0.5">
                  <span className="text-emerald-400">{JSON.stringify(data.final_top_token_str)}</span>
                  <span className="text-xs text-slate-400 font-normal">#{data.final_top_token_id}</span>
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-cyan-950/80 border border-cyan-700/40 text-cyan-400">
                <TrendingUp className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">First in Top-5</p>
                <p className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
                  {firstTop5Layer ? firstTop5Layer.layer_name : 'Late blocks'}
                </p>
                <p className="text-[10px] text-slate-400">
                  {firstTop5Layer ? `Rank #${firstTop5Layer.target_rank} (p=${(firstTop5Layer.target_probability * 100).toFixed(2)}%)` : 'Not in top 5'}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-emerald-950/80 border border-emerald-700/40 text-emerald-400">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Rank #1 Crystallization</p>
                <p className="text-sm font-bold font-mono text-emerald-300 mt-0.5">
                  {firstTop1Layer ? firstTop1Layer.layer_name : 'None'}
                </p>
                <p className="text-[10px] text-slate-400">
                  {firstTop1Layer ? `p=${(firstTop1Layer.target_probability * 100).toFixed(2)}% at crystallization` : 'Did not reach #1'}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-amber-950/80 border border-amber-700/40 text-amber-400">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Final Entropy</p>
                <p className="text-base font-bold font-mono text-amber-300 mt-0.5">
                  {data.layers[data.layers.length - 1]?.entropy.toFixed(2)} nats
                </p>
                <p className="text-[10px] text-slate-400">
                  Embedding: {data.layers[0]?.entropy.toFixed(2)} nats
                </p>
              </div>
            </div>
          </div>

          {/* Layer-by-Layer Progression Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Layer-by-Layer Unembedding Trajectory
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Tracking the interim Top-1 candidate versus the eventual target token{' '}
                  <span className="font-mono text-emerald-400">{JSON.stringify(data.final_top_token_str)}</span>
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
                {data.layers.length} Layers Evaluated
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/70 text-slate-400 uppercase font-mono text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Layer</th>
                    <th className="px-4 py-3">Interim Top-1 Candidate</th>
                    <th className="px-4 py-3">Candidate Probability</th>
                    <th className="px-4 py-3">Target Rank Progression</th>
                    <th className="px-4 py-3">Target Probability</th>
                    <th className="px-4 py-3">Shannon Entropy</th>
                    <th className="px-4 py-3">Cosine Sim (Δ)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {data.layers.map((step) => {
                    const topCand = step.top_candidates[0];
                    const isTargetTop = step.target_rank === 1;
                    const isTargetTop5 = step.target_rank <= 5;

                    return (
                      <tr
                        key={step.layer}
                        className={`hover:bg-slate-800/40 transition-colors ${
                          isTargetTop ? 'bg-emerald-950/10' : ''
                        }`}
                      >
                        <td className="px-4 py-3.5 font-bold text-slate-200 whitespace-nowrap">
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] border border-slate-700/60">
                            {step.layer_name}
                          </span>
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap">
                          {topCand ? (
                            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-slate-950 border border-slate-700/80 text-cyan-300 font-bold">
                              {JSON.stringify(topCand.token_str)}
                              <span className="text-[10px] text-slate-400 font-normal">#{topCand.token_id}</span>
                            </span>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap font-semibold text-slate-100">
                          {topCand ? `${(topCand.probability * 100).toFixed(2)}%` : '—'}
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold ${
                              isTargetTop
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : isTargetTop5
                                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                                : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }`}
                          >
                            Rank #{step.target_rank.toLocaleString()}
                          </span>
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap font-semibold">
                          <div className="flex items-center gap-2">
                            <span className={isTargetTop ? 'text-emerald-400 font-bold' : 'text-slate-300'}>
                              {(step.target_probability * 100).toFixed(2)}%
                            </span>
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                              <div
                                className={`h-full ${isTargetTop ? 'bg-emerald-400' : 'bg-cyan-500'}`}
                                style={{ width: `${Math.min(step.target_probability * 100, 100)}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap text-amber-300">
                          {step.entropy.toFixed(2)} nats
                        </td>
                        <td className="px-4 py-3.5 whitespace-nowrap text-slate-400">
                          {step.cosine_sim_to_previous !== null ? (
                            <span className="text-indigo-300">
                              {step.cosine_sim_to_previous.toFixed(4)}
                            </span>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Interactive Convergence Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Probability Trajectory */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-1 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                Target Token Probability Evolution (%)
              </h4>
              <p className="text-[11px] text-slate-400 mb-4">
                Watch the probability of final target token {JSON.stringify(data.final_top_token_str)} surge through the layers
              </p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trajectoryChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="layerName"
                      stroke="#64748b"
                      tick={{ fontSize: 10 }}
                      angle={-20}
                      textAnchor="end"
                    />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} unit="%" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '0.5rem',
                        fontSize: '11px',
                        color: '#f8fafc',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Line
                      type="monotone"
                      dataKey="targetProb"
                      name={`Target ${JSON.stringify(data.final_top_token_str)} (%)`}
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#10b981' }}
                      activeDot={{ r: 6 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="topCandidateProb"
                      name="Interim Top-1 Probability (%)"
                      stroke="#06b6d4"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={{ r: 3, fill: '#06b6d4' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Shannon Entropy Collapse */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider mb-1 flex items-center gap-2">
                <Activity className="w-4 h-4 text-amber-400" />
                Shannon Entropy & Cosine Similarity Trajectory
              </h4>
              <p className="text-[11px] text-slate-400 mb-4">
                Quantifies vocabulary distribution uncertainty and representation alignment across consecutive layers
              </p>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={trajectoryChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="layerName"
                      stroke="#64748b"
                      tick={{ fontSize: 10 }}
                      angle={-20}
                      textAnchor="end"
                    />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '0.5rem',
                        fontSize: '11px',
                        color: '#f8fafc',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="entropy" name="Shannon Entropy (nats)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Detailed Candidate Breakdown per Layer */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
            <h4 className="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Layer-by-Layer Top-{topK} Candidate Ranking Cards
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {data.layers.map((l) => (
                <div
                  key={l.layer}
                  className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2">
                      <span className="text-xs font-bold text-slate-200">{l.layer_name}</span>
                      <span className="text-[10px] font-mono text-slate-400">
                        Target: #{l.target_rank}
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {l.top_candidates.map((cand) => (
                        <div
                          key={cand.rank}
                          className="flex items-center justify-between text-xs font-mono"
                        >
                          <span className="text-slate-400 flex items-center gap-1.5 truncate">
                            <span className="text-[10px] text-slate-600">#{cand.rank}</span>
                            <span className="text-slate-200 font-semibold truncate">
                              {JSON.stringify(cand.token_str)}
                            </span>
                          </span>
                          <span className="text-cyan-400 font-medium shrink-0">
                            {(cand.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="pt-2 mt-2 border-t border-slate-900 text-[10px] text-slate-500 flex justify-between">
                    <span>Norm: {l.residual_norm.toFixed(1)}</span>
                    <span>Entropy: {l.entropy.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Educational Explanation Box */}
          <div className="p-5 rounded-xl bg-indigo-950/20 border border-indigo-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                <Info className="w-4 h-4" />
                Educational Explanation: How the Logit Lens Works
              </span>
              <Badge origin="Educational explanation" />
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {data.explanation}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-xs text-slate-400 border-t border-indigo-900/30">
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">1. Residual Stream Highway</p>
                <p className="text-[11px]">
                  Each block adds attention and MLP outputs into the residual stream (<code className="text-cyan-400">x = x + f(x)</code>), preserving semantic coordinates across depth.
                </p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">2. Early Vocabulary Projection</p>
                <p className="text-[11px]">
                  Unembedding with the shared language head allows us to query what the model would predict if computation were terminated prematurely at layer <code className="text-cyan-400">l</code>.
                </p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">3. Rank Convergence</p>
                <p className="text-[11px]">
                  Facts and concepts do not appear instantaneously at the final layer; they gradually rise from tens of thousands of ranks down to #1 through successive transformer blocks.
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { AttentionResponse, ModelInfoResponse } from '../types/api';
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
import { Play, Eye, Binary, Info, AlertCircle, CheckCircle2 } from 'lucide-react';

interface AttentionViewProps {
  initialPrompt?: string;
}

export const AttentionView: React.FC<AttentionViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);

  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [data, setData] = useState<AttentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [hoveredCell, setHoveredCell] = useState<{
    sourceIdx: number;
    sourceToken: string;
    targetIdx: number;
    targetToken: string;
    weight: number;
    isMasked: boolean;
  } | null>(null);

  // 1. Fetch model info dynamically on mount
  useEffect(() => {
    api.getModelInfo()
      .then((info) => setModelInfo(info))
      .catch((err) => console.warn('Could not load dynamic model info:', err));
  }, []);

  const numLayers = modelInfo?.num_layers ?? 6;
  const numHeads = modelInfo?.num_heads ?? 12;

  const handleFetchAttention = async (
    inputText: string,
    layerIdx: number,
    headIdx: number
  ) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getAttention(inputText, layerIdx, headIdx);
      setData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to extract attention weights');
      }
    } finally {
      setLoading(false);
    }
  };

  const textRef = React.useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  useEffect(() => {
    handleFetchAttention(textRef.current, layer, head);
  }, [layer, head]);

  // Prepare Q/K/V chart data for first 16 dimensions
  const qkvChartData = data?.qkv
    ? Array.from({ length: 16 }).map((_, dim) => ({
        dimension: `d${dim}`,
        Query: data.qkv.q_slice[dim],
        Key: data.qkv.k_slice[dim],
        Value: data.qkv.v_slice[dim],
      }))
    : [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Attention Heatmap & Q/K/V Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real self-attention weights extracted via eager attention and fused{' '}
            <code className="text-cyan-400">c_attn</code> projections
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input and Dynamic Selectors */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleFetchAttention(text, layer, head)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleFetchAttention(text, layer, head)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Analyzing...' : 'Analyze Attention'}
          </button>
        </div>

        {/* Dynamic Layer and Head Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-800/80">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Select Transformer Layer:</span>
              <span className="font-mono text-cyan-400 font-bold">
                Layer {layer} / {numLayers - 1}
              </span>
            </label>
            <select
              value={layer}
              onChange={(e) => setLayer(parseInt(e.target.value, 10))}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {Array.from({ length: numLayers }).map((_, idx) => (
                <option key={idx} value={idx}>
                  Layer {idx} {idx === 0 ? '(First Layer)' : idx === numLayers - 1 ? '(Final Layer)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Select Attention Head:</span>
              <span className="font-mono text-indigo-400 font-bold">
                Head {head} / {numHeads - 1}
              </span>
            </label>
            <select
              value={head}
              onChange={(e) => setHead(parseInt(e.target.value, 10))}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {Array.from({ length: numHeads }).map((_, idx) => (
                <option key={idx} value={idx}>
                  Head {idx} (Dimension: 64)
                </option>
              ))}
            </select>
          </div>
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
          <Eye className="w-8 h-8 mx-auto text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-300">No Attention Matrix Extracted</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Select a layer and head above and click <strong className="text-slate-400">Analyze Attention</strong> to view the real attention matrix and Q/K/V tensors.
          </p>
        </div>
      )}

      {/* Heatmap Section */}
      {data && (
        <div className="space-y-6">
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-cyan-400" />
                  Self-Attention Heatmap (Layer {data.layer}, Head {data.head})
                </h3>
                <p className="text-xs text-slate-400">
                  Rows = <span className="text-amber-300 font-medium">Source token</span> (query, attending).
                  Columns = <span className="text-cyan-300 font-medium">Target token</span> (key, attended-to).
                </p>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-3.5 h-3.5 rounded bg-cyan-500 inline-block"></span> High weight
                </span>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-3.5 h-3.5 rounded bg-slate-950 border border-slate-700 inline-block"></span> Low weight
                </span>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-3.5 h-3.5 rounded bg-slate-800 border border-dashed border-slate-600 inline-block"></span> Masked (causal)
                </span>
              </div>
            </div>

            {/* Plain CSS Grid Heatmap */}
            <div className="overflow-x-auto py-2">
              <div
                className="inline-grid gap-1.5 p-4 rounded-xl bg-slate-950 border border-slate-800"
                style={{
                  gridTemplateColumns: `auto repeat(${data.token_count}, minmax(75px, 1fr))`,
                }}
              >
                {/* Header Row (Target Tokens) */}
                <div className="text-center font-bold text-xs text-slate-400 p-2 border-b border-r border-slate-800">
                  Target →<br />Source ↓
                </div>
                {data.tokens.map((tok, cIdx) => (
                  <div
                    key={`col-${cIdx}`}
                    className="text-center text-xs font-mono font-semibold text-cyan-300 p-2 truncate"
                    title={`Key position ${cIdx}: ${tok}`}
                  >
                    <span className="block text-[10px] text-slate-500 font-sans">k_{cIdx}</span>
                    {JSON.stringify(tok)}
                  </div>
                ))}

                {/* Grid Rows */}
                {data.matrix.map((row, rIdx) => (
                  <React.Fragment key={`row-${rIdx}`}>
                    {/* Source Token Label */}
                    <div className="flex items-center gap-1 text-xs font-mono font-semibold text-amber-300 pr-3 border-r border-slate-800">
                      <span className="text-[10px] text-slate-500 font-sans">q_{rIdx}</span>
                      <span className="truncate">{JSON.stringify(data.tokens[rIdx])}</span>
                    </div>

                    {/* Cells in Row */}
                    {row.map((weight, cIdx) => {
                      const isMasked = cIdx > rIdx;
                      const isSelf = cIdx === rIdx;

                      if (isMasked) {
                        return (
                          <div
                            key={`cell-${rIdx}-${cIdx}`}
                            onMouseEnter={() =>
                              setHoveredCell({
                                sourceIdx: rIdx,
                                sourceToken: data.tokens[rIdx],
                                targetIdx: cIdx,
                                targetToken: data.tokens[cIdx],
                                weight: 0.0,
                                isMasked: true,
                              })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                            className="h-14 rounded-lg bg-slate-900/60 border border-dashed border-slate-800 flex flex-col items-center justify-center text-[10px] text-slate-600 select-none cursor-not-allowed transition-colors"
                          >
                            <span className="text-[9px] uppercase tracking-wider text-slate-600 font-sans font-semibold">
                              masked
                            </span>
                            <span className="text-[8px] text-slate-700 font-sans">(causal)</span>
                          </div>
                        );
                      }

                      // Active attention weight cell
                      const intensity = Math.min(1, Math.max(0.12, weight));
                      const isHovered =
                        hoveredCell?.sourceIdx === rIdx && hoveredCell?.targetIdx === cIdx;

                      return (
                        <div
                          key={`cell-${rIdx}-${cIdx}`}
                          onMouseEnter={() =>
                            setHoveredCell({
                              sourceIdx: rIdx,
                              sourceToken: data.tokens[rIdx],
                              targetIdx: cIdx,
                              targetToken: data.tokens[cIdx],
                              weight: weight,
                              isMasked: false,
                            })
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{
                            backgroundColor: `rgba(6, 182, 212, ${intensity * 0.85})`,
                          }}
                          className={`h-14 rounded-lg border flex flex-col items-center justify-center font-mono cursor-pointer transition-all shadow-sm ${
                            isHovered
                              ? 'border-white scale-105 z-10 shadow-lg shadow-cyan-500/20'
                              : isSelf
                              ? 'border-cyan-400/50'
                              : 'border-slate-800 hover:border-cyan-300'
                          }`}
                        >
                          <span
                            className={`text-xs font-bold ${
                              intensity > 0.45 ? 'text-slate-950 font-black' : 'text-slate-100'
                            }`}
                          >
                            {(weight * 100).toFixed(1)}%
                          </span>
                          <span
                            className={`text-[9px] ${
                              intensity > 0.45 ? 'text-slate-800' : 'text-cyan-200'
                            }`}
                          >
                            {weight.toFixed(4)}
                          </span>
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* Hover Tooltip Box */}
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs flex items-center justify-between min-h-[46px]">
              {hoveredCell ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-400">Selected Attention Connection:</span>
                  <span className="font-mono text-amber-300">
                    Pos {hoveredCell.sourceIdx}: {JSON.stringify(hoveredCell.sourceToken)}
                  </span>
                  <span className="text-slate-500">→ attends to →</span>
                  <span className="font-mono text-cyan-300">
                    Pos {hoveredCell.targetIdx}: {JSON.stringify(hoveredCell.targetToken)}
                  </span>
                  {hoveredCell.isMasked ? (
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-sans font-medium text-[11px]">
                      Masked by Causal Decoder (Upper Triangle = 0.0)
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-mono font-bold text-[11px] border border-cyan-800">
                      Weight: {(hoveredCell.weight * 100).toFixed(2)}% ({hoveredCell.weight.toFixed(6)})
                    </span>
                  )}
                </div>
              ) : (
                <span className="text-slate-500 italic">
                  Hover over any heatmap cell to inspect the source-target token routing and exact probability.
                </span>
              )}
            </div>
          </div>

          {/* Q/K/V Extraction & Mathematical Recomputation Panel */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Binary className="w-4 h-4 text-emerald-400" />
                  Q/K/V Extraction & Mathematical Identity (Layer {data.layer}, Head {data.head})
                </h3>
                <p className="text-xs text-slate-400">
                  Extracted from the fused <code className="text-cyan-400">c_attn</code> projection (width{' '}
                  <strong className="text-slate-200">{data.qkv.fused_c_attn_width} = 3 × 768</strong>)
                </p>
              </div>

              {/* Verified Badge */}
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 text-xs">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>
                  Recomputed Diff: <strong>{data.qkv.max_recomputed_difference.toFixed(6)}</strong>
                </span>
              </div>
            </div>

            {/* Shape Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs">
                <span className="text-slate-400 block font-sans text-[11px]">Query Shape [seq, 64]</span>
                <span className="text-cyan-400 font-bold text-sm">[{data.qkv.q_shape.join(', ')}]</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs">
                <span className="text-slate-400 block font-sans text-[11px]">Key Shape [seq, 64]</span>
                <span className="text-purple-400 font-bold text-sm">[{data.qkv.k_shape.join(', ')}]</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs">
                <span className="text-slate-400 block font-sans text-[11px]">Value Shape [seq, 64]</span>
                <span className="text-emerald-400 font-bold text-sm">[{data.qkv.v_shape.join(', ')}]</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs">
                <span className="text-slate-400 block font-sans text-[11px]">Fused c_attn Width</span>
                <span className="text-amber-400 font-bold text-sm">2,304 dims</span>
              </div>
            </div>

            {/* Q/K/V Vector Slices Chart */}
            <div className="h-64 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qkvChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="Query" fill="#06b6d4" name="Query Vector (Q)" />
                  <Bar dataKey="Key" fill="#a855f7" name="Key Vector (K)" />
                  <Bar dataKey="Value" fill="#10b981" name="Value Vector (V)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Mathematical Identity Proof Note */}
            <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-900/60 text-xs text-emerald-200 leading-relaxed flex items-start gap-2.5">
              <Info className="w-4 h-4 shrink-0 mt-0.5 text-emerald-400" />
              <div>
                <strong className="text-emerald-300 font-semibold">Mathematical Identity Verification:</strong>
                <p className="mt-0.5">
                  The max difference of{' '}
                  <code className="text-white bg-emerald-950 px-1 rounded font-bold">
                    {data.qkv.max_recomputed_difference.toFixed(6)}
                  </code>{' '}
                  proves that computing <code className="text-cyan-300">softmax((Q @ K^T) / sqrt(64) + mask)</code> directly
                  from the intercepted <code className="text-purple-300">c_attn</code> projection reproduces the model’s true
                  forward-pass attention matrix to machine precision.
                </p>
              </div>
            </div>
          </div>

          {/* Top 15 Weights Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Top 15 Attention Weights (Layer {data.layer}, Head {data.head})
              </span>
              <span className="text-xs text-slate-400">Sorted by attention probability</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Source Token (Query)</th>
                    <th className="px-4 py-3">Target Token (Key)</th>
                    <th className="px-4 py-3">Weight (%)</th>
                    <th className="px-4 py-3">Visual Proportion</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {data.top_weights.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-2.5 font-semibold text-slate-400">#{idx + 1}</td>
                      <td className="px-4 py-2.5 text-amber-300 font-semibold">
                        Pos {item.source_index}: {JSON.stringify(item.source_token)}
                      </td>
                      <td className="px-4 py-2.5 text-cyan-300 font-semibold">
                        Pos {item.target_index}: {JSON.stringify(item.target_token)}
                      </td>
                      <td className="px-4 py-2.5 font-bold text-slate-100">
                        {(item.weight * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-2.5 w-48">
                        <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full"
                            style={{ width: `${Math.min(100, item.weight * 100)}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Educational Explanation Banner */}
          <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-900/60 flex items-start gap-3 text-blue-200 text-xs">
            <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <strong className="font-semibold text-blue-300">Causal Attention & First-Token Routing:</strong>
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

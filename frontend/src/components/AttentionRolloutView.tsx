import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { AttentionRolloutResponse, LayerRolloutStep } from '../types/api';
import { Badge } from './Badge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import {
  Play,
  GitBranch,
  Layers,
  Sliders,
  Info,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  Share2,
  Anchor,
} from 'lucide-react';

interface AttentionRolloutViewProps {
  initialPrompt?: string;
}

export const AttentionRolloutView: React.FC<AttentionRolloutViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [residualWeight, setResidualWeight] = useState<number>(0.5);
  const [headReduction, setHeadReduction] = useState<'mean' | 'max' | 'min'>('mean');
  const [selectedLayer, setSelectedLayer] = useState<number>(5);
  const [data, setData] = useState<AttentionRolloutResponse | null>(null);
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

  const textRef = React.useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  const handleComputeRollout = async (
    inputText: string,
    alpha: number = 0.5,
    reduction: 'mean' | 'max' | 'min' = 'mean'
  ) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getAttentionRollout(inputText, alpha, reduction);
      setData(res);
      // Clamp selected layer within bounds if model layer count differs
      setSelectedLayer((prev) => (prev >= res.layer_steps.length ? res.layer_steps.length - 1 : prev));
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to compute Attention Rollout cumulative flow.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleComputeRollout(initialPrompt, 0.5, 'mean');
  }, [initialPrompt]);

  const currentLayerStep: LayerRolloutStep | undefined = data?.layer_steps[selectedLayer];

  // Helper to color cells based on cumulative flow intensity
  const getCellColor = (val: number, isMasked: boolean) => {
    if (isMasked) return 'bg-slate-900/60 pattern-hatched border-slate-900';
    if (val === 0) return 'bg-slate-950 text-slate-600';
    if (val < 0.05) return 'bg-indigo-950/40 text-indigo-300';
    if (val < 0.15) return 'bg-indigo-900/60 text-indigo-200';
    if (val < 0.3) return 'bg-indigo-800/80 text-white font-medium';
    if (val < 0.6) return 'bg-cyan-600 text-white font-bold';
    return 'bg-emerald-500 text-slate-950 font-extrabold shadow-sm shadow-emerald-500/20';
  };

  // Bar chart data for the end-to-end attribution vector
  const attributionChartData = data?.final_attribution.map((item) => ({
    name: JSON.stringify(item.token_str),
    percentage: item.percentage,
    isSink: item.token_index === 0,
    rawScore: item.attribution_score,
  })) || [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-indigo-400" />
            Attention Rollout & Token Attribution Flow
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Quantify cumulative information flow through multi-layer residual connections (Abnar &amp; Zuidema, ACL 2020)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input and Configuration Controls */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleComputeRollout(text, residualWeight, headReduction)}
            placeholder="Type a sentence to trace cumulative attention flow..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleComputeRollout(text, residualWeight, headReduction)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20 shrink-0"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Propagating...' : 'Compute Rollout'}
          </button>
        </div>

        {/* Hyperparameter Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-slate-800/80">
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" /> Residual Weight (&alpha;):
              </span>
              <span className="font-mono font-bold text-cyan-400">{residualWeight.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={residualWeight}
              onChange={(e) => {
                const w = parseFloat(e.target.value);
                setResidualWeight(w);
                handleComputeRollout(textRef.current, w, headReduction);
              }}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>0.1 (High Residual Highway)</span>
              <span>0.5 (Default Balanced)</span>
              <span>0.9 (High Attention Focus)</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Head Reduction Strategy:</span>
              <span className="font-mono text-indigo-400 font-bold uppercase">{headReduction}</span>
            </label>
            <select
              value={headReduction}
              onChange={(e) => {
                const r = e.target.value as 'mean' | 'max' | 'min';
                setHeadReduction(r);
                handleComputeRollout(textRef.current, residualWeight, r);
              }}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="mean">Mean (Head-Averaged Attention Flow - Standard)</option>
              <option value="max">Max (Dominant Routing Head)</option>
              <option value="min">Min (Conservative Core Attention)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-900/60 flex items-start gap-3 text-red-200 text-sm">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Attention Rollout Error</p>
            <p className="text-xs text-red-300/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !data && !error && (
        <div className="p-12 text-center rounded-xl bg-slate-900/40 border border-slate-800/80 text-slate-400">
          <GitBranch className="w-10 h-10 mx-auto mb-3 text-slate-600 animate-pulse" />
          <p className="text-sm font-medium text-slate-300">Ready to Trace Attention Rollout</p>
          <p className="text-xs text-slate-500 mt-1">
            Click &quot;Compute Rollout&quot; to recursively propagate attention matrices across all layers.
          </p>
        </div>
      )}

      {/* Live Data Display */}
      {data && currentLayerStep && (
        <>
          {/* Summary KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-emerald-950/80 border border-emerald-700/40 text-emerald-400">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Top Attributed Token</p>
                <p className="text-base font-bold font-mono text-slate-100 flex items-center gap-1.5 mt-0.5">
                  <span className="text-emerald-400">{JSON.stringify(data.top_attributed_token)}</span>
                </p>
                <p className="text-[10px] text-slate-400">
                  Credit: {(data.top_attributed_score * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-amber-950/80 border border-amber-700/40 text-amber-400">
                <Anchor className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Attention Sink Share</p>
                <p className="text-base font-bold font-mono text-amber-300 mt-0.5">
                  {(data.final_attribution[0]?.percentage ?? 0).toFixed(1)}%
                </p>
                <p className="text-[10px] text-slate-400">
                  Position 0 ({JSON.stringify(data.tokens[0])})
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-cyan-950/80 border border-cyan-700/40 text-cyan-400">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Inspecting Depth</p>
                <p className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
                  {currentLayerStep.layer_name}
                </p>
                <p className="text-[10px] text-slate-400">
                  {selectedLayer === data.layer_steps.length - 1 ? 'Final Cumulative Output' : 'Intermediate Cumulative Flow'}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-indigo-950/80 border border-indigo-700/40 text-indigo-400">
                <Share2 className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 uppercase font-semibold">Residual Highway (&alpha;)</p>
                <p className="text-base font-bold font-mono text-indigo-300 mt-0.5">
                  {data.residual_weight.toFixed(2)}
                </p>
                <p className="text-[10px] text-slate-400">
                  Identity: {(1.0 - data.residual_weight).toFixed(2)}
                </p>
              </div>
            </div>
          </div>

          {/* Depth Evolution Tabs */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 overflow-x-auto">
            <span className="text-xs text-slate-400 font-semibold px-3 py-1 shrink-0 uppercase tracking-wider">
              Rollout Depth:
            </span>
            {data.layer_steps.map((step) => {
              const isSelected = step.layer === selectedLayer;
              const isFinal = step.layer === data.layer_steps.length - 1;
              return (
                <button
                  key={step.layer}
                  onClick={() => setSelectedLayer(step.layer)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 shrink-0 ${
                    isSelected
                      ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <span>Layer {step.layer}</span>
                  {isFinal && (
                    <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-700/50">
                      Final
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Matrix and Attribution Chart Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Cumulative Rollout Heatmap Matrix */}
            <div className="lg:col-span-7 p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                      <GitBranch className="w-4 h-4 text-cyan-400" />
                      Cumulative Rollout Matrix: {currentLayerStep.layer_name}
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Rows = Source tokens. Columns = Information origin tokens.
                    </p>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-300 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                    Row Sum = 1.000
                  </span>
                </div>

                {/* Grid Container */}
                <div className="overflow-x-auto pb-2">
                  <div
                    className="grid gap-1 min-w-[320px]"
                    style={{
                      gridTemplateColumns: `auto repeat(${data.token_count}, minmax(40px, 1fr))`,
                    }}
                  >
                    {/* Header Row: Target Tokens */}
                    <div className="text-[10px] text-slate-500 font-mono p-1"></div>
                    {data.tokens.map((tok, cIdx) => (
                      <div
                        key={cIdx}
                        className="text-[10px] font-mono text-slate-400 p-1 text-center truncate font-bold"
                        title={tok}
                      >
                        {JSON.stringify(tok)}
                      </div>
                    ))}

                    {/* Matrix Rows */}
                    {currentLayerStep.rollout_matrix.map((row, rIdx) => (
                      <React.Fragment key={rIdx}>
                        {/* Row Header: Source Token */}
                        <div
                          className="text-[10px] font-mono text-slate-300 pr-2 flex items-center justify-end font-semibold truncate"
                          title={data.tokens[rIdx]}
                        >
                          {JSON.stringify(data.tokens[rIdx])}
                        </div>

                        {/* Row Cells */}
                        {row.map((val, cIdx) => {
                          const isMasked = cIdx > rIdx;
                          return (
                            <div
                              key={cIdx}
                              onMouseEnter={() =>
                                setHoveredCell({
                                  sourceIdx: rIdx,
                                  sourceToken: data.tokens[rIdx],
                                  targetIdx: cIdx,
                                  targetToken: data.tokens[cIdx],
                                  weight: val,
                                  isMasked,
                                })
                              }
                              onMouseLeave={() => setHoveredCell(null)}
                              className={`h-9 flex items-center justify-center rounded border text-[10px] font-mono transition-all cursor-pointer ${getCellColor(
                                val,
                                isMasked
                              )}`}
                            >
                              {isMasked ? (
                                <span className="text-[9px] text-slate-700 uppercase font-sans">mask</span>
                              ) : (
                                (val * 100).toFixed(0) + '%'
                              )}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>

              {/* Hover Tooltip / Detail Bar */}
              <div className="mt-4 p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs flex items-center justify-between min-h-[44px]">
                {hoveredCell ? (
                  hoveredCell.isMasked ? (
                    <span className="text-slate-500 flex items-center gap-1.5 font-mono">
                      <AlertCircle className="w-3.5 h-3.5 text-slate-500" />
                      Causal Mask: {JSON.stringify(hoveredCell.sourceToken)} cannot attend to future token {JSON.stringify(hoveredCell.targetToken)}.
                    </span>
                  ) : (
                    <span className="text-slate-200 flex items-center gap-2 font-mono">
                      <span className="text-cyan-400 font-bold">{JSON.stringify(hoveredCell.sourceToken)}</span>
                      <span className="text-slate-500">&rarr;</span>
                      <span className="text-emerald-400 font-bold">{JSON.stringify(hoveredCell.targetToken)}</span>:
                      <span className="text-white font-bold">{(hoveredCell.weight * 100).toFixed(2)}%</span> of cumulative information
                    </span>
                  )
                ) : (
                  <span className="text-slate-500 text-[11px] flex items-center gap-1.5">
                    <HelpCircle className="w-3.5 h-3.5" /> Hover over any matrix cell to inspect specific token routing.
                  </span>
                )}
                <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-indigo-950 border border-indigo-700"></span> 0%
                  <span className="w-2.5 h-2.5 rounded-sm bg-cyan-600 ml-1"></span> 50%
                  <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 ml-1"></span> 100%
                </div>
              </div>
            </div>

            {/* Right: End-to-End Attribution Bar Chart */}
            <div className="lg:col-span-5 p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2 mb-1">
                  <Anchor className="w-4 h-4 text-emerald-400" />
                  Next-Token Attribution Credit (%)
                </h3>
                <p className="text-[11px] text-slate-400 mb-4">
                  End-to-end information credit assigned to each input token for generating the next token
                </p>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={attributionChartData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 10 }} unit="%" />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 11, fontFamily: 'monospace' }} width={80} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0f172a',
                          borderColor: '#334155',
                          borderRadius: '0.5rem',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                        formatter={(val: unknown) => {
                          const num = typeof val === 'number' ? val : Number(val);
                          return [`${num.toFixed(2)}%`, 'Attribution Credit'];
                        }}
                      />
                      <Bar dataKey="percentage" radius={[0, 4, 4, 0]}>
                        {attributionChartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.isSink ? '#f59e0b' : '#10b981'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Legend and Analysis */}
              <div className="pt-3 border-t border-slate-800/80 space-y-2 text-xs">
                <div className="flex items-center gap-4 text-[11px]">
                  <span className="flex items-center gap-1.5 text-amber-300 font-mono">
                    <span className="w-2.5 h-2.5 rounded-sm bg-amber-500"></span> Position 0 (Attention Sink)
                  </span>
                  <span className="flex items-center gap-1.5 text-emerald-300 font-mono">
                    <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span> Contextual Tokens
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Notice that the first token <code className="text-amber-400">{JSON.stringify(data.tokens[0])}</code> absorbs a massive share of credit. In autoregressive models, position 0 acts as a universal information delimiter, while remaining weight distributes across key semantic entities.
                </p>
              </div>
            </div>
          </div>

          {/* Educational Explanation Box */}
          <div className="p-5 rounded-xl bg-indigo-950/20 border border-indigo-900/40 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                <Info className="w-4 h-4" />
                Educational Explanation: Attention Rollout Mechanics
              </span>
              <Badge origin="Educational explanation" />
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {data.explanation}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-xs text-slate-400 border-t border-indigo-900/30">
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">1. The Residual Problem</p>
                <p className="text-[11px]">
                  Looking only at Layer 5 attention is misleading because 50% of the representation flows directly through the identity residual stream without being re-routed by attention.
                </p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">2. Recursive Accumulation</p>
                <p className="text-[11px]">
                  Multiplying the adjusted matrices <code className="text-cyan-400">R_l = A_hat_l &times; R_&#123;l-1&#125;</code> propagates weights through all intermediate paths, creating a true end-to-end flow.
                </p>
              </div>
              <div>
                <p className="font-semibold text-slate-200 mb-0.5">3. Attention Sinks</p>
                <p className="text-[11px]">
                  Tokens with high rollout scores are not necessarily the most &quot;important&quot; in a human sense; position 0 absorbs excess probability mass to maintain stable softmax scaling.
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

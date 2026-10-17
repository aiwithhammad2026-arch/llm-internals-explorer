import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { LayerInspectionResponse, ModelInfoResponse, LayerActivationComponent } from '../types/api';
import { Badge } from './Badge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import {
  Play,
  Box,
  ArrowDown,
  Layers,
  Sparkles,
  Info,
  AlertCircle,
  Zap,
  Activity,
} from 'lucide-react';

interface BlockInspectionViewProps {
  initialPrompt?: string;
}

export const BlockInspectionView: React.FC<BlockInspectionViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [layer, setLayer] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(0);

  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [data, setData] = useState<LayerInspectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 1. Fetch model info dynamically
  useEffect(() => {
    api.getModelInfo()
      .then((info) => setModelInfo(info))
      .catch((err) => console.warn('Could not load dynamic model info:', err));
  }, []);

  const numLayers = modelInfo?.num_layers ?? 6;

  const handleInspect = async (inputText: string, layerIdx: number, tokIdx: number) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.inspectLayer(inputText, layerIdx, 16, tokIdx);
      setData(res);
      // Ensure token index is clamped within bounds if text shortened
      if (tokIdx >= res.token_count) {
        setTokenIndex(0);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to inspect transformer block activations');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleInspect(text, layer, tokenIndex);
  }, [layer, tokenIndex]);

  // Visual styling dictionary for flow components
  const getComponentMeta = (name: string) => {
    switch (name) {
      case 'input_hidden_state':
        return {
          title: '1. Input Hidden State',
          badgeColor: 'border-indigo-500/40 bg-indigo-950/40 text-indigo-300',
          barColor: '#6366f1',
          icon: Layers,
        };
      case 'ln_1_output':
        return {
          title: '2. LayerNorm 1 (Pre-LN)',
          badgeColor: 'border-amber-500/40 bg-amber-950/40 text-amber-300',
          barColor: '#f59e0b',
          icon: Activity,
        };
      case 'attn_output':
        return {
          title: '3. Multi-Head Self-Attention Output',
          badgeColor: 'border-cyan-500/40 bg-cyan-950/40 text-cyan-300',
          barColor: '#06b6d4',
          icon: Zap,
        };
      case 'residual_after_attn':
        return {
          title: '4. First Residual Addition (x + Attn)',
          badgeColor: 'border-purple-500/40 bg-purple-950/40 text-purple-300',
          barColor: '#a855f7',
          icon: Sparkles,
        };
      case 'ln_2_output':
        return {
          title: '5. LayerNorm 2 (Pre-LN)',
          badgeColor: 'border-amber-500/40 bg-amber-950/40 text-amber-300',
          barColor: '#f59e0b',
          icon: Activity,
        };
      case 'mlp_output':
        return {
          title: '6. Feed-Forward MLP Output',
          badgeColor: 'border-emerald-500/40 bg-emerald-950/40 text-emerald-300',
          barColor: '#10b981',
          icon: Box,
        };
      case 'residual_after_mlp':
        return {
          title: '7. Second Residual Addition (Exit Stream)',
          badgeColor: 'border-rose-500/40 bg-rose-950/40 text-rose-300',
          barColor: '#f43f5e',
          icon: Sparkles,
        };
      default:
        return {
          title: name,
          badgeColor: 'border-slate-700 bg-slate-900 text-slate-300',
          barColor: '#94a3b8',
          icon: Box,
        };
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Transformer Block Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Internal forward-pass dataflow through a chosen layer captured via PyTorch forward hooks
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Illustrative diagram" />
          <Badge origin="Real model output" />
        </div>
      </div>

      {/* Control Panel: Text, Layer Selector & Token Position Selector */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleInspect(text, layer, tokenIndex)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleInspect(text, layer, tokenIndex)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Inspecting...' : 'Inspect Block'}
          </button>
        </div>

        {/* Dynamic Selectors */}
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
                  Layer {idx} {idx === 0 ? '(First Block)' : idx === numLayers - 1 ? '(Final Block)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
              <span>Inspect Token Vector at Position:</span>
              <span className="font-mono text-indigo-400 font-bold">
                Index {tokenIndex} {data ? `(${JSON.stringify(data.tokens[tokenIndex] || '')})` : ''}
              </span>
            </label>
            <select
              value={tokenIndex}
              onChange={(e) => setTokenIndex(parseInt(e.target.value, 10))}
              disabled={!data || data.tokens.length === 0}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
            >
              {data?.tokens.map((tok, idx) => (
                <option key={idx} value={idx}>
                  Token {idx}: {JSON.stringify(tok)}
                </option>
              )) || <option value={0}>Token 0</option>}
            </select>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <strong className="font-semibold">Inspection Error:</strong>
            <p className="mt-0.5 text-xs text-rose-200">{error}</p>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!data && !loading && !error && (
        <div className="p-10 text-center rounded-xl bg-slate-900/50 border border-dashed border-slate-800 space-y-3">
          <Box className="w-8 h-8 mx-auto text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-300">No Block Activations Inspected</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Choose a layer and click <strong className="text-slate-400">Inspect Block</strong> to examine real intermediate activations across Pre-LN, self-attention, and MLP sub-layers.
          </p>
        </div>
      )}

      {/* Vertical Dataflow Diagram */}
      {data && (
        <div className="space-y-4">
          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Box className="w-4 h-4 text-indigo-400" />
              Layer {data.layer} Internal Execution Sequence (Token {data.token_index}: {JSON.stringify(data.token_str)})
            </span>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">
                Hidden Size: <strong className="text-slate-200 font-mono">{data.hidden_size}</strong>
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-[11px] text-slate-400">
                Sequence Length: <strong className="text-slate-200 font-mono">{data.token_count}</strong>
              </span>
            </div>
          </div>

          <div className="relative pl-6 space-y-6 before:absolute before:left-3 before:top-4 before:bottom-4 before:w-0.5 before:bg-slate-800">
            {data.components
              .filter((comp) => comp.is_captured)
              .map((comp: LayerActivationComponent, idx: number) => {
                const meta = getComponentMeta(comp.name);
                const Icon = meta.icon;

                // Format chart data for first 16 values
                const chartData = comp.slice_values.map((val, dIdx) => ({
                  dim: `d${dIdx}`,
                  value: val,
                }));

                return (
                  <div key={comp.name} className="relative">
                    {/* Flow Connector Dot */}
                    <div className="absolute -left-6 top-5 w-6 h-6 rounded-full bg-slate-900 border-2 border-indigo-500/60 flex items-center justify-center -translate-x-1/2 shadow-md">
                      <span className="text-[10px] font-bold text-indigo-300 font-mono">{idx + 1}</span>
                    </div>

                    {/* Component Card */}
                    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3 hover:border-slate-700 transition-all">
                      {/* Component Title & Metadata */}
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
                        <div className="flex items-center gap-2.5">
                          <div className={`p-1.5 rounded-lg border ${meta.badgeColor}`}>
                            <Icon className="w-4 h-4" />
                          </div>
                          <div>
                            <h4 className="text-sm font-bold text-slate-200">{meta.title}</h4>
                            <p className="text-xs text-slate-400">{comp.description}</p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-300">
                            Shape: [{comp.shape.join(', ')}]
                          </span>
                          <span className="px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/40 text-[10px] text-emerald-300 font-semibold">
                            Captured Hook
                          </span>
                        </div>
                      </div>

                      {/* Summary Statistics Pill */}
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/60">
                        <div>
                          <span className="text-slate-500 text-[10px] block font-sans">Min</span>
                          <span className="text-slate-200">{comp.stats.min.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block font-sans">Max</span>
                          <span className="text-slate-200">{comp.stats.max.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block font-sans">Mean</span>
                          <span className="text-slate-200">{comp.stats.mean.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block font-sans">Std Dev</span>
                          <span className="text-slate-200">{comp.stats.std.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 text-[10px] block font-sans">L2 Norm</span>
                          <span className="text-indigo-400 font-bold">{comp.stats.norm.toFixed(4)}</span>
                        </div>
                      </div>

                      {/* 16-Value Slice Bar Chart */}
                      <div className="space-y-1.5 pt-1">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-slate-400">
                            First 16 Activation Dimensions (Token {data.token_index}: {JSON.stringify(data.token_str)})
                          </span>
                          <span className="text-slate-500 font-mono text-[10px]">
                            dims 0..15 of 768
                          </span>
                        </div>
                        <div className="h-28 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="dim" stroke="#64748b" tick={{ fontSize: 10 }} />
                              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{
                                  backgroundColor: '#0f172a',
                                  borderColor: '#334155',
                                  borderRadius: '6px',
                                  fontSize: '11px',
                                }}
                                formatter={(val: any) => typeof val === 'number' ? val.toFixed(6) : val}
                              />
                              <Bar dataKey="value" fill={meta.barColor} radius={[2, 2, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    {/* Arrow down to next component */}
                    {idx < data.components.length - 1 && (
                      <div className="flex justify-center -my-3 py-1">
                        <ArrowDown className="w-4 h-4 text-slate-600" />
                      </div>
                    )}
                  </div>
                );
              })}
          </div>

          {/* Educational Explanation Box */}
          <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-900/60 flex items-start gap-3 text-blue-200 text-xs">
            <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <strong className="font-semibold text-blue-300">
                  Pre-LN Transformer Architecture & Residual Streams:
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

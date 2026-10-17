import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { PredictNextTokenResponse, GenerateResponse, TutorCompareGenerationResponse } from '../types/api';
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
import { Play, AlertCircle, Info, Zap, Flame, Sliders, Bot, Sparkles, Loader2 } from 'lucide-react';

interface NextTokenViewProps {
  initialPrompt?: string;
}

export const NextTokenView: React.FC<NextTokenViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(10);
  const [maxNewTokens, setMaxNewTokens] = useState(15);

  const [predictData, setPredictData] = useState<PredictNextTokenResponse | null>(null);
  const [generateData, setGenerateData] = useState<GenerateResponse | null>(null);
  const [compareData, setCompareData] = useState<TutorCompareGenerationResponse | null>(null);

  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingGenerate, setLoadingGenerate] = useState(false);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  const handlePredict = async (currentPrompt: string, temp: number, k: number) => {
    if (!currentPrompt.trim()) return;
    setLoadingPredict(true);
    setError(null);
    try {
      const res = await api.predictNextToken(currentPrompt, temp, k);
      setPredictData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to compute next-token probabilities');
      }
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoadingGenerate(true);
    setError(null);
    try {
      const res = await api.generate(prompt, maxNewTokens, temperature, 50);
      setGenerateData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to generate text');
      }
    } finally {
      setLoadingGenerate(false);
    }
  };

  const handleCompareGemini = async () => {
    if (!prompt.trim()) return;
    setLoadingCompare(true);
    setCompareError(null);
    try {
      const res = await api.compareGeneration(prompt, generateData?.continuation);
      setCompareData(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setCompareError(err.message);
      } else {
        setCompareError('Failed to generate comparison with Gemini. Check GEMINI_API_KEY in backend/.env.');
      }
    } finally {
      setLoadingCompare(false);
    }
  };

  useEffect(() => {
    handlePredict(initialPrompt, temperature, topK);
  }, []);

  // Format chart data
  const chartData = predictData?.top_candidates.map((c) => ({
    name: JSON.stringify(c.token_str),
    probability: Math.round(c.probability * 10000) / 100, // percentage
    probRaw: c.probability,
    logit: c.logit,
    rank: c.rank,
  })) || [];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Next-Token Prediction & Generation
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Raw final-position logits transformed into normalized probabilities via temperature-scaled Softmax
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Controls & Prompt Box */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Prompt Context
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePredict(prompt, temperature, topK)}
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handlePredict(prompt, temperature, topK)}
            disabled={loadingPredict || !prompt.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            <Play className="w-4 h-4 fill-current" />
            {loadingPredict ? 'Predicting...' : 'Predict Next Token'}
          </button>
        </div>

        {/* Sliders: Temperature & Top-K */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 pt-3 border-t border-slate-800">
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-amber-400" /> Temperature:
              </span>
              <span className="font-mono font-bold text-amber-400">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.5"
              step="0.05"
              value={temperature}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setTemperature(val);
                handlePredict(prompt, val, topK);
              }}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <p className="text-[10px] text-slate-400">
              Low: sharpens peaks. High: flattens probabilities.
            </p>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" /> Top-K Candidates:
              </span>
              <span className="font-mono font-bold text-cyan-400">{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="30"
              step="1"
              value={topK}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                setTopK(val);
                handlePredict(prompt, temperature, val);
              }}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <p className="text-[10px] text-slate-400">Number of ranked candidates to inspect.</p>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-indigo-400" /> Max New Tokens:
              </span>
              <span className="font-mono font-bold text-indigo-400">{maxNewTokens}</span>
            </div>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-1.5">
              <button
                onClick={handleGenerate}
                disabled={loadingGenerate || !prompt.trim()}
                className="py-1.5 px-3 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium text-xs transition-colors flex items-center justify-center gap-1.5 shadow"
              >
                <Zap className="w-3.5 h-3.5 fill-current" />
                {loadingGenerate ? 'Generating...' : 'Generate Local'}
              </button>

              <button
                onClick={handleCompareGemini}
                disabled={loadingCompare || !prompt.trim()}
                className="py-1.5 px-3 rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-xs transition-colors flex items-center justify-center gap-1.5 shadow"
              >
                {loadingCompare ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Querying Gemini...
                  </>
                ) : (
                  <>
                    <Bot className="w-3.5 h-3.5" />
                    Compare with Gemini
                  </>
                )}
              </button>
            </div>
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

      {/* Compare Error */}
      {compareError && (
        <div className="p-4 rounded-lg bg-amber-950/60 border border-amber-800 text-amber-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
          <div>
            <strong className="font-semibold">Gemini Tutor Notice:</strong>
            <p className="mt-0.5 text-xs text-amber-200">{compareError}</p>
          </div>
        </div>
      )}

      {/* Compare with Gemini Side-by-Side Panel */}
      {compareData && (
        <div className="p-5 rounded-xl bg-gradient-to-br from-indigo-950/40 via-slate-900 to-slate-950 border border-indigo-500/40 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-slate-100">
                Foundational Model Comparison ({compareData.model_name})
              </h3>
            </div>
            <Badge origin="Gemini-written explanation" />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Local distilgpt2 output */}
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5" /> Local distilgpt2 (82M params)
              </span>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                <span className="text-slate-500">{prompt}</span>
                <span className="text-emerald-300 font-bold ml-1">
                  {generateData?.continuation || ' [Click "Generate Local" to run distilgpt2]'}
                </span>
              </p>
            </div>

            {/* Google Gemini output */}
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-cyan-400 flex items-center gap-1.5">
                <Bot className="w-3.5 h-3.5" /> Google {compareData.model_name}
              </span>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                <span className="text-slate-500">{prompt}</span>
                <span className="text-cyan-300 font-bold ml-1">
                  {compareData.gemini_continuation}
                </span>
              </p>
            </div>
          </div>

          {/* Model Disclaimer */}
          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/60 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
            <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-cyan-400" />
            <span>{compareData.disclaimer}</span>
          </div>
        </div>
      )}

      {/* Autoregressive Generation Result (if generated) */}
      {generateData && (
        <div className="p-5 rounded-xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-500/40 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-emerald-400" /> Autoregressive Continuation Output
            </span>
            <span className="text-xs text-slate-400">
              Generated {generateData.new_token_count} new tokens ({generateData.finish_reason})
            </span>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 font-mono text-sm leading-relaxed">
            <span className="text-slate-400">{generateData.prompt}</span>
            <span className="text-emerald-300 font-bold bg-emerald-950/40 px-1 rounded ml-1">
              {generateData.continuation}
            </span>
          </div>

          {/* Tokens list */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {generateData.new_tokens.map((t) => (
              <span
                key={t.step}
                className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
              >
                #{t.step}: <strong className="text-emerald-400">{JSON.stringify(t.token_str)}</strong> (ID {t.token_id})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!predictData && !loadingPredict && !error && (
        <div className="p-10 text-center rounded-xl bg-slate-900/50 border border-dashed border-slate-800 space-y-3">
          <Zap className="w-8 h-8 mx-auto text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-300">No Predictions Computed Yet</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Enter a prompt above and click <strong className="text-slate-400">Compute Probabilities</strong> to evaluate model logits and next-token distributions.
          </p>
        </div>
      )}

      {/* Results Display */}
      {predictData && (
        <div className="space-y-6">
          {/* Top Token Highlight Card */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-slate-400 font-medium">Rank #1 Next-Token Candidate</span>
              <div className="text-2xl font-mono font-bold text-amber-300 mt-1 flex items-center gap-3">
                <span>{JSON.stringify(predictData.top_1_token_str)}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-normal">
                  ID: {predictData.top_1_token_id}
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-normal">
                  {(predictData.top_candidates[0].probability * 100).toFixed(2)}% probability
                </span>
              </div>
            </div>
            <div className="text-right sm:text-right">
              <span className="text-xs text-slate-400">Total Top-{topK} Probability Mass</span>
              <div className="text-xl font-mono font-bold text-cyan-400 mt-1">
                {(predictData.total_probability_top_k * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Probability Bar Chart */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-200">
                Top Candidate Probability Distribution (Softmax over 50,257 tokens)
              </h3>
              <span className="text-xs text-slate-400">
                Temperature = <strong>{predictData.temperature_used}</strong>
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="name"
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    angle={-20}
                    textAnchor="end"
                  />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(val: any, _name: any, item: any) => [
                      `${val}% (Logit: ${item.payload.logit})`,
                      'Probability',
                    ]}
                  />
                  <Bar dataKey="probability" name="Probability (%)">
                    {chartData.map((_entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index === 0 ? '#10b981' : index < 3 ? '#38bdf8' : '#6366f1'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Candidates Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Ranked Candidates Table
              </span>
              <span className="text-xs text-slate-400">
                Vocabulary: <strong>50,257</strong> possible tokens
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Token ID</th>
                    <th className="px-4 py-3">Subword Byte Symbol</th>
                    <th className="px-4 py-3">Decoded String</th>
                    <th className="px-4 py-3">Raw Logit</th>
                    <th className="px-4 py-3">Softmax Probability</th>
                    <th className="px-4 py-3">Cumulative Mass</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {predictData.top_candidates.map((c) => (
                    <tr
                      key={c.rank}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        c.rank === 1 ? 'bg-emerald-950/20' : ''
                      }`}
                    >
                      <td className="px-4 py-3 font-semibold text-slate-400">#{c.rank}</td>
                      <td className="px-4 py-3 text-cyan-400 font-bold">{c.token_id}</td>
                      <td className="px-4 py-3 text-amber-300 font-semibold">{c.token_bytes_display}</td>
                      <td className="px-4 py-3 text-slate-200">{JSON.stringify(c.token_str)}</td>
                      <td className="px-4 py-3 text-slate-400">{c.logit.toFixed(4)}</td>
                      <td className="px-4 py-3 text-emerald-400 font-bold">
                        {(c.probability * 100).toFixed(4)}%
                      </td>
                      <td className="px-4 py-3 text-slate-400">
                        {(c.cumulative_probability * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Temperature Explanation Banner */}
          <div className="p-4 rounded-xl bg-blue-950/40 border border-blue-900/60 flex items-start gap-3 text-blue-200 text-xs">
            <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <strong className="font-semibold text-blue-300">Temperature & Logits Principle:</strong>
                <Badge origin={predictData.explanation_origin || "Educational explanation"} size="sm" />
              </div>
              <p className="mt-1 leading-relaxed text-blue-200/90">{predictData.temperature_explanation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

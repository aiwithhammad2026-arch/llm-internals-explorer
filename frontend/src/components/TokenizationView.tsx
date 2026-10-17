import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { TokenizeResponse } from '../types/api';
import { Badge } from './Badge';
import { Play, Sparkles, AlertCircle, Hash, Layers, Binary } from 'lucide-react';

interface TokenizationViewProps {
  initialPrompt?: string;
}

export const TokenizationView: React.FC<TokenizationViewProps> = ({
  initialPrompt = 'The capital of France is',
}) => {
  const [text, setText] = useState(initialPrompt);
  const [data, setData] = useState<TokenizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number | null>(null);

  const samplePrompts = [
    'The capital of France is',
    'Transformers are attention based models',
    'Hello world',
    'Deep learning models learn internal representations',
  ];

  const handleTokenize = async (inputText: string) => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.tokenize(inputText);
      setData(res);
      setSelectedTokenIdx(null);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to tokenize text');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleTokenize(initialPrompt);
  }, [initialPrompt]);

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* View Header with Origin Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Tokenization Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real Byte-Pair Encoding (BPE) from the active <code className="text-cyan-300">distilgpt2</code> tokenizer
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <Badge origin="Educational explanation" />
        </div>
      </div>

      {/* Input Section */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Input Prompt Text
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleTokenize(text)}
            placeholder="Type any text to tokenize..."
            className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <button
            onClick={() => handleTokenize(text)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
          >
            <Play className="w-4 h-4 fill-current" />
            {loading ? 'Tokenizing...' : 'Tokenize'}
          </button>
        </div>

        {/* Sample Prompt Chips */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-cyan-400" /> Samples:
          </span>
          {samplePrompts.map((p) => (
            <button
              key={p}
              onClick={() => {
                setText(p);
                handleTokenize(p);
              }}
              className="text-xs px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Error Banner */}
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
          <h3 className="text-sm font-semibold text-slate-300">No Tokens Extracted Yet</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Enter prompt text above and click <strong className="text-slate-400">Tokenize</strong> to inspect real BPE token IDs, decoded strings, and tensor shapes.
          </p>
        </div>
      )}

      {/* Results Section */}
      {data && (
        <div className="space-y-6">
          {/* Top Metrics Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-xs text-slate-400 font-medium">Token Count</span>
                <div className="text-2xl font-bold text-slate-100 mt-1">{data.token_count}</div>
              </div>
              <div className="w-10 h-10 rounded-lg bg-indigo-950/80 border border-indigo-800 flex items-center justify-center text-indigo-400 font-mono text-sm">
                N={data.token_count}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-xs text-slate-400 font-medium">PyTorch Input Tensor Shape</span>
                <div className="text-2xl font-mono font-bold text-cyan-400 mt-1">
                  [{data.input_tensor_shape.join(', ')}]
                </div>
              </div>
              <div className="w-10 h-10 rounded-lg bg-cyan-950/80 border border-cyan-800 flex items-center justify-center text-cyan-400">
                <Layers className="w-5 h-5" />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-xs text-slate-400 font-medium">Batch × Sequence</span>
                <div className="text-sm font-mono text-slate-200 mt-1">
                  batch_size: <strong className="text-indigo-400">1</strong>, seq_len:{' '}
                  <strong className="text-cyan-400">{data.token_count}</strong>
                </div>
              </div>
              <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400">
                <Hash className="w-5 h-5" />
              </div>
            </div>
          </div>

          {/* Token Visualizer Chips */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Sequence Tokens (Click chip to inspect)
              </span>
              <span className="text-xs text-slate-400">
                Leading <code className="text-cyan-300">␣</code> indicates a BPE space character
              </span>
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              {data.tokens.map((tok) => {
                const isSelected = selectedTokenIdx === tok.index;
                return (
                  <button
                    key={tok.index}
                    onClick={() => setSelectedTokenIdx(isSelected ? null : tok.index)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-mono transition-all ${
                      isSelected
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-lg shadow-cyan-500/10'
                        : 'bg-slate-950 hover:bg-slate-800 border-slate-700 text-slate-200'
                    }`}
                  >
                    <span className="text-[10px] text-slate-400 font-sans font-medium">
                      #{tok.index}
                    </span>
                    <span className="font-semibold text-amber-300">{tok.token_bytes_display}</span>
                    <span className="text-[11px] px-1 py-0.2 rounded bg-slate-800 text-cyan-400">
                      ID {tok.token_id}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Token Mapping Table */}
          <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Exact Token-to-ID Vocabulary Mapping
              </span>
              <span className="text-xs text-slate-400">
                Vocabulary Size: <strong>50,257</strong>
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Sequence Position</th>
                    <th className="px-4 py-3">BPE Subword Symbol</th>
                    <th className="px-4 py-3">Decoded String</th>
                    <th className="px-4 py-3">Token Integer ID</th>
                    <th className="px-4 py-3">Hex Encoding</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {data.tokens.map((tok) => {
                    const isSelected = selectedTokenIdx === tok.index;
                    return (
                      <tr
                        key={tok.index}
                        onClick={() => setSelectedTokenIdx(isSelected ? null : tok.index)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-cyan-950/30 text-cyan-200'
                            : 'hover:bg-slate-800/40 text-slate-300'
                        }`}
                      >
                        <td className="px-4 py-3 font-semibold text-slate-400">
                          Pos {tok.index}
                        </td>
                        <td className="px-4 py-3 text-amber-300 font-semibold">
                          {tok.token_bytes_display}
                        </td>
                        <td className="px-4 py-3 text-slate-200">
                          {JSON.stringify(tok.token_str)}
                        </td>
                        <td className="px-4 py-3 text-cyan-400 font-bold">
                          {tok.token_id}
                        </td>
                        <td className="px-4 py-3 text-slate-400">
                          0x{tok.token_id.toString(16).toUpperCase()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

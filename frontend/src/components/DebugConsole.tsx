import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { DebugLogsResponse, DebugInfoResponse, RequestLogEntry } from '../types/api';
import { Badge } from './Badge';
import {
  Terminal,
  RefreshCw,
  Trash2,
  Cpu,
  Layers,
  Clock,
  AlertCircle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';

export const DebugConsole: React.FC = () => {
  const [logsData, setLogsData] = useState<DebugLogsResponse | null>(null);
  const [infoData, setInfoData] = useState<DebugInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedLogId, setExpandedLogId] = useState<number | null>(null);

  const fetchDebugData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [logs, info] = await Promise.all([
        api.getDebugLogs(),
        api.getDebugInfo(),
      ]);
      setLogsData(logs);
      setInfoData(info);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to fetch debug logs from backend');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      await api.clearDebugLogs();
      await fetchDebugData();
    } catch (err: unknown) {
      console.error('Failed to clear logs:', err);
    }
  };

  useEffect(() => {
    fetchDebugData();
    // Poll every 8 seconds
    const interval = setInterval(fetchDebugData, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Terminal className="w-5 h-5 text-cyan-400" />
            Backend Debug Console
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time request metadata ring buffer (last 50 requests), tensor shapes, dtypes, and execution latencies
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Real model output" />
          <button
            onClick={fetchDebugData}
            title="Refresh logs"
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={handleClear}
            title="Clear ring buffer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-rose-950/60 text-slate-400 hover:text-rose-300 border border-slate-800 hover:border-rose-800 transition-colors text-xs"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Logs
          </button>
        </div>
      </div>

      {/* Runtime Environment Metadata Card */}
      {infoData && (
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" /> Model Runtime Environment
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              Revision: <strong className="text-slate-200">{infoData.model_revision}</strong>
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 text-xs font-mono">
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">Model</span>
              <span className="text-cyan-400 font-bold">{infoData.model_name}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">Device / Dtype</span>
              <span className="text-slate-200">{infoData.device} / {infoData.dtype}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">Attention Impl</span>
              <span className="text-amber-400 font-bold">{infoData.attn_implementation}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">Tokenizer Class</span>
              <span className="text-indigo-400 truncate block">{infoData.tokenizer_class}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">PyTorch</span>
              <span className="text-slate-300">v{infoData.torch_version}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-500 font-sans block">Transformers</span>
              <span className="text-slate-300">v{infoData.transformers_version}</span>
            </div>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <strong className="font-semibold">Debug Error:</strong>
            <p className="mt-0.5 text-xs text-rose-200">{error}</p>
          </div>
        </div>
      )}

      {/* Request Ring Buffer Table */}
      <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden space-y-0">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            Request Ring Buffer ({logsData?.total_captured || 0} / {logsData?.capacity || 50} captured)
          </span>
          <span className="text-xs text-slate-500">
            Click any row to inspect tensor shapes, dtypes, & latency breakdown
          </span>
        </div>

        {logsData && logsData.logs.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs italic">
            Ring buffer is currently empty. Make requests across other views to inspect real telemetry.
          </div>
        ) : (
          <div className="divide-y divide-slate-800/80">
            {logsData?.logs.slice().reverse().map((log: RequestLogEntry) => {
              const isExpanded = expandedLogId === log.id;
              const isError = log.status_code >= 400 || log.errors.length > 0;

              return (
                <div key={log.id} className="hover:bg-slate-800/30 transition-colors">
                  {/* Row Summary */}
                  <div
                    onClick={() => setExpandedLogId(isExpanded ? null : log.id)}
                    className="p-3.5 flex items-center justify-between cursor-pointer text-xs font-mono"
                  >
                    <div className="flex items-center gap-3">
                      <button className="text-slate-500 hover:text-slate-300">
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-indigo-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-slate-500" />
                        )}
                      </button>

                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isError
                            ? 'bg-rose-950 text-rose-400 border border-rose-800'
                            : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        }`}
                      >
                        {log.status_code}
                      </span>

                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-semibold text-[10px]">
                        {log.method}
                      </span>

                      <span className="text-slate-200 font-semibold">{log.endpoint}</span>
                    </div>

                    <div className="flex items-center gap-4 text-slate-400 text-[11px]">
                      <span className="flex items-center gap-1 font-mono">
                        <Clock className="w-3 h-3 text-cyan-400" />
                        {log.duration_ms.toFixed(1)} ms
                      </span>

                      <span className="text-slate-500 hidden sm:inline">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>

                  {/* Expanded Detail Panel */}
                  {isExpanded && (
                    <div className="p-4 bg-slate-950/90 border-t border-slate-800/80 space-y-3 text-xs">
                      {/* Grid of properties */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <span className="text-[10px] text-slate-500 font-sans block">Execution Device</span>
                          <span className="text-slate-200">{log.device}</span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <span className="text-[10px] text-slate-500 font-sans block">Model Name</span>
                          <span className="text-cyan-400">{log.model_name}</span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <span className="text-[10px] text-slate-500 font-sans block">Layer / Head</span>
                          <span className="text-slate-200">
                            {log.layer_selected !== null ? `Layer ${log.layer_selected}` : 'N/A'} /{' '}
                            {log.head_selected !== null ? `Head ${log.head_selected}` : 'N/A'}
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 border border-slate-800">
                          <span className="text-[10px] text-slate-500 font-sans block">Gemini Status</span>
                          <span className="text-slate-200">{log.gemini_status || 'not_called'}</span>
                        </div>
                      </div>

                      {/* Tensor Shapes & Dtypes */}
                      {Object.keys(log.tensor_shapes).length > 0 && (
                        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1.5 font-mono">
                          <span className="text-[11px] font-sans font-semibold text-slate-400 block">
                            Captured Tensor Dimensions & Dtypes:
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(log.tensor_shapes).map(([tensorName, shape]) => (
                              <span
                                key={tensorName}
                                className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-[11px] text-slate-300"
                              >
                                <strong className="text-cyan-300">{tensorName}</strong>: [{shape.join(', ')}]{' '}
                                <span className="text-slate-500">({log.tensor_dtypes[tensorName] || 'float32'})</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Input Tokens */}
                      {log.input_token_ids.length > 0 && (
                        <div className="text-[11px] text-slate-400 font-mono">
                          Input Token IDs (first 16):{' '}
                          <span className="text-amber-300">[{log.input_token_ids.join(', ')}]</span>
                        </div>
                      )}

                      {/* Warnings / Errors */}
                      {log.errors.length > 0 && (
                        <div className="p-2.5 rounded bg-rose-950/40 border border-rose-900/60 text-rose-300 text-[11px]">
                          <strong>Errors:</strong> {log.errors.join('; ')}
                        </div>
                      )}
                      {log.warnings.length > 0 && (
                        <div className="p-2.5 rounded bg-amber-950/40 border border-amber-900/60 text-amber-300 text-[11px]">
                          <strong>Warnings:</strong> {log.warnings.join('; ')}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

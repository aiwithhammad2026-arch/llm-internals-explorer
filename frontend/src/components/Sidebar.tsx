import {
  FileCode,
  Binary,
  TrendingUp,
  Eye,
  Box,
  Bot,
  Terminal,
  GitCompare,
  Activity,
  CheckCircle2,
  AlertCircle,
  Clock,
  X,
  Sparkles,
  GitBranch,
} from 'lucide-react';
import type { HealthResponse } from '../types/api';

export type ActiveTab =
  | 'tokenizer'
  | 'embeddings'
  | 'predict'
  | 'attention'
  | 'rollout'
  | 'blocks'
  | 'logit-lens'
  | 'tutor'
  | 'debug'
  | 'compare';

interface SidebarProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  health,
  healthLoading,
  healthError,
  isOpenMobile = false,
  onCloseMobile,
}) => {
  const navItems = [
    {
      id: 'tokenizer' as ActiveTab,
      label: 'Tokenization Inspector',
      icon: FileCode,
      phase: 'Phase 1',
      enabled: true,
      desc: 'BPE tokens, IDs, & tensor shape',
    },
    {
      id: 'embeddings' as ActiveTab,
      label: 'Embedding Inspector',
      icon: Binary,
      phase: 'Phase 2',
      enabled: true,
      desc: 'Token (wte) + Position (wpe)',
    },
    {
      id: 'predict' as ActiveTab,
      label: 'Next-Token Prediction',
      icon: TrendingUp,
      phase: 'Phase 2',
      enabled: true,
      desc: 'Logits, Softmax & Generation',
    },
    {
      id: 'attention' as ActiveTab,
      label: 'Attention Heatmap & Q/K/V',
      icon: Eye,
      phase: 'Phase 5',
      enabled: true,
      desc: 'Causal weights & head routing',
    },
    {
      id: 'rollout' as ActiveTab,
      label: 'Attention Rollout Flow',
      icon: GitBranch,
      phase: 'Phase 11',
      enabled: true,
      desc: 'Cumulative multi-layer attribution',
    },
    {
      id: 'blocks' as ActiveTab,
      label: 'Transformer Blocks',
      icon: Box,
      phase: 'Phase 5',
      enabled: true,
      desc: 'LayerNorm, MLP & Residuals',
    },
    {
      id: 'logit-lens' as ActiveTab,
      label: 'Logit Lens Evolution',
      icon: Sparkles,
      phase: 'Phase 10',
      enabled: true,
      desc: 'Residual layer prediction emergence',
    },
    {
      id: 'tutor' as ActiveTab,
      label: 'Gemini AI Tutor',
      icon: Bot,
      phase: 'Phase 6',
      enabled: true,
      desc: 'Plain-English AI Explanations',
    },
    {
      id: 'debug' as ActiveTab,
      label: 'Backend Debug Console',
      icon: Terminal,
      phase: 'Phase 7',
      enabled: true,
      desc: 'Shapes, dtypes & latencies',
    },
    {
      id: 'compare' as ActiveTab,
      label: 'Model Comparison',
      icon: GitCompare,
      phase: 'Phase 7',
      enabled: true,
      desc: 'distilgpt2 vs gpt2 benchmarks',
    },
  ];

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-50 w-72 bg-slate-900 border-r border-slate-800 flex flex-col h-screen select-none transform transition-transform duration-200 md:relative md:translate-x-0 ${
        isOpenMobile ? 'translate-x-0 shadow-2xl' : '-translate-x-full'
      }`}
    >
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-100 tracking-tight leading-none">
              Transformer Explorer
            </h1>
            <span className="text-[10px] text-cyan-400 font-medium">Real Internals Engine</span>
          </div>
        </div>

        {/* Mobile Close Button */}
        {onCloseMobile && (
          <button
            onClick={onCloseMobile}
            className="md:hidden p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
            title="Close navigation menu"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Live Model Status Pill */}
      <div className="px-5 pt-3">
        <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-slate-400 font-medium">Model Engine:</span>
            {healthLoading ? (
              <span className="flex items-center gap-1 text-slate-400">
                <Clock className="w-3 h-3 animate-spin" /> Checking...
              </span>
            ) : health?.model_loaded ? (
              <span className="flex items-center gap-1 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" /> Online
              </span>
            ) : (
              <span className="flex items-center gap-1 text-rose-400 font-semibold">
                <AlertCircle className="w-3 h-3 text-rose-400" /> Offline
              </span>
            )}
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-300">
            <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded text-cyan-300">
              {health?.model_name || 'distilgpt2'}
            </span>
            <span className="text-slate-400 capitalize">
              Device: <strong className="text-slate-200">{health?.device || 'cpu'}</strong>
            </span>
          </div>
          {healthError && (
            <div className="mt-1 text-[10px] text-rose-400 leading-tight">
              Backend not responding
            </div>
          )}
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        <div className="px-3 pb-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          Exploration Modules
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.enabled && activeTab === item.id;

          if (item.enabled) {
            return (
              <button
                key={item.id}
                onClick={() => {
                  onTabChange(item.id as ActiveTab);
                  if (onCloseMobile) onCloseMobile();
                }}
                className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                  isActive
                    ? 'bg-indigo-600/15 border border-indigo-500/40 text-indigo-300 shadow-sm'
                    : 'text-slate-300 hover:bg-slate-800/60 hover:text-white border border-transparent'
                }`}
              >
                <Icon
                  className={`w-4 h-4 mt-0.5 shrink-0 ${
                    isActive ? 'text-indigo-400' : 'text-slate-400'
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium leading-tight truncate">{item.label}</span>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/40">
                      Live
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 truncate mt-0.5">{item.desc}</p>
                </div>
              </button>
            );
          }

          return (
            <div
              key={item.id}
              className="w-full flex items-start gap-3 px-3 py-2.5 rounded-lg text-left opacity-40 cursor-not-allowed border border-transparent"
              title={`${item.label} will be enabled in ${item.phase}`}
            >
              <Icon className="w-4 h-4 mt-0.5 shrink-0 text-slate-500" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400 truncate">{item.label}</span>
                  <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-slate-400">
                    {item.phase}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 truncate mt-0.5">{item.desc}</p>
              </div>
            </div>
          );
        })}
      </nav>

      {/* Mandatory Credit Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40 text-center">
        <div className="text-[11px] font-semibold text-slate-300">
          Real Transformer Internals Explorer
        </div>
        <div className="text-[11px] font-medium text-cyan-400 mt-0.5">
          By Hammad, AI Engineer
        </div>
      </div>
    </aside>
  );
};

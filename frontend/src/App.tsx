import React, { useState, useEffect, lazy, Suspense } from 'react';
import { Sidebar, type ActiveTab } from './components/Sidebar';
import { api } from './services/api';
import type { HealthResponse } from './types/api';
import { Terminal, RefreshCw, Cpu, Server, Loader2, X, ChevronUp, ChevronDown, Menu } from 'lucide-react';

// Lazy-loaded views for code splitting and fast initial bundle load
const TokenizationView = lazy(() =>
  import('./components/TokenizationView').then((m) => ({ default: m.TokenizationView }))
);
const EmbeddingView = lazy(() =>
  import('./components/EmbeddingView').then((m) => ({ default: m.EmbeddingView }))
);
const NextTokenView = lazy(() =>
  import('./components/NextTokenView').then((m) => ({ default: m.NextTokenView }))
);
const AttentionView = lazy(() =>
  import('./components/AttentionView').then((m) => ({ default: m.AttentionView }))
);
const BlockInspectionView = lazy(() =>
  import('./components/BlockInspectionView').then((m) => ({ default: m.BlockInspectionView }))
);
const TutorView = lazy(() =>
  import('./components/TutorView').then((m) => ({ default: m.TutorView }))
);
const DebugConsole = lazy(() =>
  import('./components/DebugConsole').then((m) => ({ default: m.DebugConsole }))
);
const ModelComparisonView = lazy(() =>
  import('./components/ModelComparisonView').then((m) => ({ default: m.ModelComparisonView }))
);
const LogitLensView = lazy(() =>
  import('./components/LogitLensView').then((m) => ({ default: m.LogitLensView }))
);

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('tokenizer');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Global collapsible debug drawer toggle (accessible from every view)
  const [debugDrawerOpen, setDebugDrawerOpen] = useState(false);

  const fetchHealth = async () => {
    try {
      const res = await api.getHealth();
      setHealth(res);
      setHealthError(null);
    } catch (err: unknown) {
      setHealth(null);
      if (err instanceof Error) {
        setHealthError(err.message);
      } else {
        setHealthError('Backend is offline');
      }
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    // Poll every 10 seconds
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans relative">
      {/* Mobile Drawer Backdrop */}
      {mobileMenuOpen && (
        <div
          onClick={() => setMobileMenuOpen(false)}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 md:hidden transition-opacity"
        />
      )}

      {/* Sidebar Navigation with Credit */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        isOpenMobile={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Top Header Bar */}
        <header className="h-14 border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-3 sm:px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            {/* Mobile Menu Hamburger Button */}
            <button
              onClick={() => setMobileMenuOpen(true)}
              className="md:hidden p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-colors"
              title="Open Navigation Menu"
            >
              <Menu className="w-4 h-4" />
            </button>

            <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 hidden sm:inline">
              Active:
            </span>
            <span className="text-xs sm:text-sm font-bold text-slate-100 truncate">
              {activeTab === 'tokenizer' && 'Tokenization Inspector'}
              {activeTab === 'embeddings' && 'Embedding Inspector'}
              {activeTab === 'predict' && 'Next-Token & Generation'}
              {activeTab === 'attention' && 'Attention & Q/K/V'}
              {activeTab === 'blocks' && 'Transformer Blocks'}
              {activeTab === 'tutor' && 'Gemini AI Tutor'}
              {activeTab === 'debug' && 'Debug Console'}
              {activeTab === 'compare' && 'Model Comparison'}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Quick Global Debug Toggle Button */}
            <button
              onClick={() => setDebugDrawerOpen(!debugDrawerOpen)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono transition-all border ${
                debugDrawerOpen
                  ? 'bg-cyan-950/80 border-cyan-500/60 text-cyan-300'
                  : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
              }`}
              title="Toggle Live Debug Drawer from any view"
            >
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              <span>Debug Console</span>
              {debugDrawerOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            </button>

            {health?.model_loaded ? (
              <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 text-xs text-slate-300 border border-slate-700">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span>
                  <strong className="text-slate-100">{health.model_name}</strong> on{' '}
                  <span className="text-cyan-400 font-mono">{health.device}</span>
                </span>
              </div>
            ) : (
              <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-rose-950/60 text-xs text-rose-300 border border-rose-800">
                <Server className="w-3.5 h-3.5 text-rose-400" />
                <span>FastAPI Disconnected</span>
              </div>
            )}

            <button
              onClick={fetchHealth}
              title="Refresh backend status"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${healthLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </header>

        {/* Global Offline Banner if Backend is down */}
        {healthError && (
          <div className="bg-amber-950/90 border-b border-amber-800 px-6 py-3 flex items-start gap-3 text-amber-200 text-xs shrink-0">
            <Terminal className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <strong className="font-semibold text-amber-300">FastAPI Backend is Offline:</strong>
              <p className="mt-0.5 text-amber-200/90">
                Start the backend server in a separate Windows PowerShell window using:
              </p>
              <code className="block mt-1 font-mono text-cyan-300 bg-slate-950 px-2 py-1 rounded border border-slate-800 select-all">
                cd "c:\Users\hamma\OneDrive\Desktop\taransfermore llm\backend"; .\venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
              </code>
            </div>
          </div>
        )}

        {/* Active View Container with Suspense Lazy Loading */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Suspense
            fallback={
              <div className="flex items-center justify-center h-64 text-slate-400 gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
                <span className="text-sm">Loading Inspector module...</span>
              </div>
            }
          >
            {activeTab === 'tokenizer' && <TokenizationView />}
            {activeTab === 'embeddings' && <EmbeddingView />}
            {activeTab === 'predict' && <NextTokenView />}
            {activeTab === 'attention' && <AttentionView />}
            {activeTab === 'blocks' && <BlockInspectionView />}
            {activeTab === 'logit-lens' && <LogitLensView />}
            {activeTab === 'tutor' && <TutorView />}
            {activeTab === 'debug' && <DebugConsole />}
            {activeTab === 'compare' && <ModelComparisonView />}
          </Suspense>
        </main>

        {/* Collapsible Global Debug Console Drawer (available from every view) */}
        {debugDrawerOpen && activeTab !== 'debug' && (
          <div className="absolute inset-x-0 bottom-0 h-96 bg-slate-950/95 backdrop-blur-lg border-t-2 border-indigo-500/60 shadow-2xl flex flex-col z-50 transition-all">
            <div className="h-10 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between shrink-0">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                Live Request Debug Drawer (Global Overlay)
              </span>
              <button
                onClick={() => setDebugDrawerOpen(false)}
                className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                title="Close Drawer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <Suspense fallback={<div className="text-xs text-slate-400 p-4">Loading Debug Console...</div>}>
                <DebugConsole />
              </Suspense>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;

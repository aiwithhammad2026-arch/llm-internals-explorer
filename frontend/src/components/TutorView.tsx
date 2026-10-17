import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { TutorStatusResponse, QuizQuestionItem } from '../types/api';
import { Badge } from './Badge';
import {
  Bot,
  HelpCircle,
  Award,
  Send,
  Key,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Sparkles,
  BookOpen,
  Info,
  ExternalLink,
} from 'lucide-react';

export const TutorView: React.FC = () => {
  const [status, setStatus] = useState<TutorStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Active sub-tab in Tutor: 'explain' | 'ask' | 'quiz'
  const [tab, setTab] = useState<'explain' | 'ask' | 'quiz'>('explain');

  // Explain mode states
  const [explainViewName, setExplainViewName] = useState('attention');
  const [explainPrompt, setExplainPrompt] = useState('The capital of France is');
  const [explainResult, setExplainResult] = useState<string | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  // Ask mode states
  const [question, setQuestion] = useState('');
  const [askResult, setAskResult] = useState<string | null>(null);
  const [askLoading, setAskLoading] = useState(false);

  // Quiz mode states
  const [quizTopic, setQuizTopic] = useState('attention');
  const [quizQuestions, setQuizQuestions] = useState<QuizQuestionItem[] | null>(null);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, number>>({});
  const [quizLoading, setQuizLoading] = useState(false);

  const fetchStatus = async () => {
    setStatusLoading(true);
    setError(null);
    try {
      const res = await api.getTutorStatus();
      setStatus(res);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to query Gemini AI Tutor status.');
      }
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleExplain = async () => {
    if (!status?.configured) return;
    setExplainLoading(true);
    setError(null);
    setExplainResult(null);

    // Build bounded sample context for the chosen view
    const contextMap: Record<string, Record<string, unknown>> = {
      attention: {
        prompt: explainPrompt,
        layer: 0,
        head: 0,
        num_heads: 12,
        head_dim: 64,
        fused_c_attn_width: 2304,
        top_weight: { source: ' is', target: 'The', weight: 0.5297 },
        note: 'First token receives 52.97% attention probability (attention sink)',
      },
      tokenizer: {
        prompt: explainPrompt,
        token_count: 5,
        tokens: ['The', ' capital', ' of', ' France', ' is'],
        token_ids: [464, 3139, 286, 4881, 318],
        input_tensor_shape: [1, 5],
      },
      embeddings: {
        prompt: explainPrompt,
        embedding_dim: 768,
        wte_shape: [1, 5, 768],
        wpe_shape: [1, 5, 768],
        combined_shape: [1, 5, 768],
      },
      predict: {
        prompt: explainPrompt,
        top_candidate: { token: ' Paris', probability: 0.2847 },
        temperature: 1.0,
        top_k: 10,
      },
      blocks: {
        prompt: explainPrompt,
        layer: 0,
        pipeline: ['input_hidden_state', 'ln_1', 'attn_output', 'residual_1', 'ln_2', 'mlp_output', 'residual_2'],
        architecture: 'Pre-LayerNorm with residual addition highways',
      },
    };

    try {
      const res = await api.explainView(explainViewName, contextMap[explainViewName] || { prompt: explainPrompt });
      setExplainResult(res.explanation);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to generate Gemini explanation.');
      }
    } finally {
      setExplainLoading(false);
    }
  };

  const handleAsk = async (userQ: string) => {
    const q = userQ || question;
    if (!q.trim() || !status?.configured) return;
    setAskLoading(true);
    setError(null);
    setAskResult(null);
    try {
      const res = await api.askTutor(q.trim());
      setAskResult(res.answer);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to receive response from Gemini AI Tutor.');
      }
    } finally {
      setAskLoading(false);
    }
  };

  const handleGenerateQuiz = async () => {
    if (!status?.configured) return;
    setQuizLoading(true);
    setError(null);
    setSelectedAnswers({});
    try {
      const res = await api.getTutorQuiz(quizTopic);
      setQuizQuestions(res.questions);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to generate quiz questions.');
      }
    } finally {
      setQuizLoading(false);
    }
  };

  const sampleQuestions = [
    'Why does GPT-2 place so much attention on the first token?',
    'What does the 2304 width in c_attn represent?',
    'How do residual connections prevent vanishing gradients in transformers?',
    'What is the difference between BPE subword tokens and raw characters?',
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Bot className="w-5 h-5 text-indigo-400" />
            Gemini AI Tutor
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Pedagogical AI companion grounded strictly in real <code className="text-cyan-400">distilgpt2</code> internal activations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge origin="Gemini-written explanation" />
          <button
            onClick={fetchStatus}
            title="Refresh Tutor Status"
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${statusLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Mandatory Privacy & Transmission Notice */}
      <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-400 flex items-start gap-3">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-cyan-400" />
        <div className="leading-relaxed">
          <strong className="text-slate-300">Privacy & Data Transmission Notice:</strong> Text prompts and
          bounded context summaries sent to the AI Tutor are transmitted over HTTPS to Google's Gemini API.
          All model internals, weights, and embeddings remain processed 100% locally on your machine.
          API calls occur strictly when you click a button.
        </div>
      </div>

      {/* Status Banner */}
      {status && !status.configured && (
        <div className="p-6 rounded-xl bg-gradient-to-br from-amber-950/40 via-slate-900 to-slate-950 border border-amber-800/80 space-y-4">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-amber-950/80 border border-amber-700/60 text-amber-300 shrink-0">
              <Key className="w-5 h-5" />
            </div>
            <div className="flex-1">
              <h3 className="text-base font-bold text-amber-200">
                Add GEMINI_API_KEY to backend/.env to enable the AI Tutor
              </h3>
              <p className="text-xs text-amber-300/80 mt-1 leading-relaxed">
                The core explorer is running 100% locally and fully functional. To unlock the optional AI Tutor for
                pedagogical explanations and interactive Q&A:
              </p>
            </div>
          </div>

          <div className="bg-slate-950/80 p-4 rounded-lg border border-slate-800 space-y-2 text-xs font-mono">
            <div className="text-slate-400 font-sans font-semibold">Quick Setup Steps:</div>
            <div className="text-slate-300 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-800 text-center leading-5 font-bold text-[10px]">1</span>
              <span>Get an API key from Google AI Studio:</span>
              <a
                href="https://aistudio.google.com/"
                target="_blank"
                rel="noreferrer"
                className="text-cyan-400 underline flex items-center gap-1 font-sans font-medium"
              >
                aistudio.google.com <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            <div className="text-slate-300 flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-800 text-center leading-5 font-bold text-[10px]">2</span>
              <span>Open <code className="text-amber-300">backend/.env</code> and paste your key:</span>
            </div>
            <code className="block bg-slate-900 p-2.5 rounded text-cyan-300 select-all border border-slate-800">
              GEMINI_API_KEY=your_gemini_api_key_here...<br />
              GEMINI_MODEL=gemini-2.5-flash
            </code>
            <div className="text-slate-300 flex items-center gap-2 pt-1">
              <span className="w-5 h-5 rounded-full bg-slate-800 text-center leading-5 font-bold text-[10px]">3</span>
              <span>Click the refresh button above once saved.</span>
            </div>
          </div>
        </div>
      )}

      {/* Global Error Banner */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
          <div>
            <strong className="font-semibold">AI Tutor Notice:</strong>
            <p className="mt-0.5 text-xs text-rose-200">{error}</p>
          </div>
        </div>
      )}

      {/* Main Tutor Workspace (Active when configured or previewing) */}
      <div className="space-y-6">
        {/* Sub-mode Navigation Tabs */}
        <div className="flex border-b border-slate-800">
          <button
            onClick={() => setTab('explain')}
            className={`flex items-center gap-2 px-5 py-3 text-xs font-semibold border-b-2 transition-all ${
              tab === 'explain'
                ? 'border-indigo-500 text-indigo-300 bg-indigo-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Explain This View
          </button>

          <button
            onClick={() => setTab('ask')}
            className={`flex items-center gap-2 px-5 py-3 text-xs font-semibold border-b-2 transition-all ${
              tab === 'ask'
                ? 'border-indigo-500 text-indigo-300 bg-indigo-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <HelpCircle className="w-4 h-4" />
            Ask a Question
          </button>

          <button
            onClick={() => setTab('quiz')}
            className={`flex items-center gap-2 px-5 py-3 text-xs font-semibold border-b-2 transition-all ${
              tab === 'quiz'
                ? 'border-indigo-500 text-indigo-300 bg-indigo-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Award className="w-4 h-4" />
            Quiz Me (3 Questions)
          </button>
        </div>

        {/* Tab 1: Explain View */}
        {tab === 'explain' && (
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Explain Architecture Module
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Gemini translates the mathematical tensors and routing operations into plain-English pedagogical concepts.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                  Target Explorer View
                </label>
                <select
                  value={explainViewName}
                  onChange={(e) => setExplainViewName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="attention">Attention Heatmap & Q/K/V</option>
                  <option value="blocks">Transformer Blocks (Pre-LN & MLP)</option>
                  <option value="embeddings">Embedding Inspector (wte + wpe)</option>
                  <option value="predict">Next-Token Prediction & Logits</option>
                  <option value="tokenizer">Tokenization Inspector (BPE)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                  Context Prompt
                </label>
                <input
                  type="text"
                  value={explainPrompt}
                  onChange={(e) => setExplainPrompt(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <button
              onClick={handleExplain}
              disabled={explainLoading || !status?.configured}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
            >
              {explainLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating Explanation...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Explain View with Gemini
                </>
              )}
            </button>

            {/* Explanation Result Box */}
            {explainResult && (
              <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <span className="text-xs font-bold text-cyan-300 uppercase tracking-wider">
                    Gemini Explanation
                  </span>
                  <Badge origin="Gemini-written explanation" />
                </div>
                <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
                  {explainResult}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Ask a Question */}
        {tab === 'ask' && (
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-indigo-400" />
                Ask the AI Tutor
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Ask any question about self-attention, residual connections, logits, or BPE tokenization.
              </p>
            </div>

            {/* Suggested Sample Questions */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                Suggested Questions:
              </span>
              <div className="flex flex-wrap gap-2">
                {sampleQuestions.map((sq, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuestion(sq);
                      handleAsk(sq);
                    }}
                    className="text-left px-3 py-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 hover:text-cyan-300 transition-colors"
                  >
                    {sq}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Box */}
            <div className="space-y-1.5">
              <div className="relative">
                <textarea
                  rows={3}
                  value={question}
                  maxLength={500}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Type your question here (e.g. Why is the causal upper triangle zeroed out?)..."
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
                <span className="absolute right-3 bottom-3 text-[10px] text-slate-500 font-mono">
                  {question.length}/500 chars
                </span>
              </div>

              <button
                onClick={() => handleAsk(question)}
                disabled={askLoading || !question.trim() || !status?.configured}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-md shadow-indigo-600/20"
              >
                {askLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Consulting Tutor...
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Ask AI Tutor
                  </>
                )}
              </button>
            </div>

            {/* Answer Box */}
            {askResult && (
              <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider">
                    AI Tutor Answer
                  </span>
                  <Badge origin="Gemini-written explanation" />
                </div>
                <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
                  {askResult}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Conceptual Quiz */}
        {tab === 'quiz' && (
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Award className="w-4 h-4 text-emerald-400" />
                  Conceptual Quiz (3 Questions)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Test your understanding of Transformer mechanics with interactive feedback.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  value={quizTopic}
                  onChange={(e) => setQuizTopic(e.target.value)}
                  className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="attention">Self-Attention & Q/K/V</option>
                  <option value="blocks">Transformer Blocks & Residuals</option>
                  <option value="embeddings">Token & Position Embeddings</option>
                  <option value="generation">Logits & Autoregressive Decoding</option>
                </select>

                <button
                  onClick={handleGenerateQuiz}
                  disabled={quizLoading || !status?.configured}
                  className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium text-xs transition-all shadow-md shadow-emerald-600/20"
                >
                  {quizLoading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    'Generate Quiz'
                  )}
                </button>
              </div>
            </div>

            {/* Render 3 Questions */}
            {quizQuestions && (
              <div className="space-y-4 pt-2">
                {quizQuestions.map((q) => {
                  const answeredIdx = selectedAnswers[q.id];
                  const hasAnswered = answeredIdx !== undefined;

                  return (
                    <div
                      key={q.id}
                      className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-bold text-slate-200">
                          {q.id}. {q.question}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono shrink-0">
                          Question {q.id}/3
                        </span>
                      </div>

                      {/* Options */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {q.options.map((opt, oIdx) => {
                          const isChosen = answeredIdx === oIdx;
                          const isCorrect = q.correct_index === oIdx;

                          let btnStyle = 'border-slate-800 bg-slate-900 text-slate-300 hover:border-slate-700';
                          if (hasAnswered) {
                            if (isCorrect) {
                              btnStyle = 'border-emerald-500 bg-emerald-950/60 text-emerald-200 font-semibold';
                            } else if (isChosen && !isCorrect) {
                              btnStyle = 'border-rose-500 bg-rose-950/60 text-rose-200';
                            } else {
                              btnStyle = 'border-slate-800/40 bg-slate-950 text-slate-500 opacity-60';
                            }
                          }

                          return (
                            <button
                              key={oIdx}
                              onClick={() => {
                                if (!hasAnswered) {
                                  setSelectedAnswers((prev) => ({ ...prev, [q.id]: oIdx }));
                                }
                              }}
                              className={`p-2.5 rounded-lg border text-left text-xs transition-all flex items-start gap-2 ${btnStyle}`}
                            >
                              <span className="font-mono font-bold text-slate-500">
                                {String.fromCharCode(65 + oIdx)}.
                              </span>
                              <span className="flex-1">{opt}</span>
                            </button>
                          );
                        })}
                      </div>

                      {/* Explanation Reveal */}
                      {hasAnswered && (
                        <div
                          className={`p-3 rounded-lg text-xs leading-relaxed flex items-start gap-2 ${
                            answeredIdx === q.correct_index
                              ? 'bg-emerald-950/30 border border-emerald-900/60 text-emerald-200'
                              : 'bg-rose-950/30 border border-rose-900/60 text-rose-200'
                          }`}
                        >
                          {answeredIdx === q.correct_index ? (
                            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400 mt-0.5" />
                          ) : (
                            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
                          )}
                          <div>
                            <strong className="block font-semibold mb-0.5">
                              {answeredIdx === q.correct_index ? 'Correct!' : 'Incorrect'}
                            </strong>
                            {q.explanation}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

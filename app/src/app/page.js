'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  createModel,
  trainModel,
  predictColor,
  saveModel,
  loadModel,
} from './utils/colorUtils';
import ColorPicker from './components/ColorPicker';
import s from './page.module.css';

/* ── Constants ──────────────────────────────────────────────── */
const MODEL_ID = 'color-predictor-v1';
const ONBOARDING_KEY = 'color-predictor-onboarding-dismissed-v1';
const SESSION_KEY = 'color-predictor-session-state-v1';

/* ── Helpers ────────────────────────────────────────────────── */
const safeParse = (raw, fallback) => {
  try {
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
};

const shouldIgnoreShortcut = (e) => {
  const tag = e.target?.tagName?.toLowerCase();
  return tag === 'input' || tag === 'textarea' || e.target?.isContentEditable || e.metaKey || e.ctrlKey || e.altKey;
};

const removeLastMatch = (items, value) => {
  const copy = [...items];
  const idx = copy.lastIndexOf(value);
  if (idx >= 0) copy.splice(idx, 1);
  return copy;
};

/* ═══════════════════════════════════════════════════════════════
   SVG Icon System
   ─────────────────────────────────────────────────────────────
   All icons use Lucide-style stroked SVGs. They inherit
   `currentColor` so they theme automatically with text color.
   ═══════════════════════════════════════════════════════════════ */
const SvgIcon = ({ children, size = 20, className, ...rest }) => (
  <svg
    className={className}
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    focusable="false"
    {...rest}
  >
    {children}
  </svg>
);

/* ── Action Icons ───────────────────────────────────────────── */
const ThumbUpIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M7 10v12" />
    <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
  </SvgIcon>
);

const ThumbDownIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M17 14V2" />
    <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
  </SvgIcon>
);

const BrainIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5V5a2 2 0 0 1 2-2h.5A2.5 2.5 0 0 1 17 5.5a2.5 2.5 0 0 1-.5 1.5 2.5 2.5 0 0 1 2 2.5c0 .83-.41 1.56-1.03 2.01.39.45.53 1.06.53 1.49a2.5 2.5 0 0 1-2.5 2.5h-.5a2 2 0 0 1-2-2v-.5a2.5 2.5 0 0 1-2.5-2.5 2.5 2.5 0 0 1 .53-1.49A2.5 2.5 0 0 1 10 7.5a2.5 2.5 0 0 1-.5-1.5A2.5 2.5 0 0 1 12 3.5" />
    <path d="M12 2v20" />
    <path d="M8 7h8" />
    <path d="M6 12h12" />
    <path d="M8 17h8" />
  </SvgIcon>
);

const SparklesIcon = (props) => (
  <SvgIcon {...props}>
    <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
  </SvgIcon>
);

const UndoIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M3 7v6h6" />
    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
  </SvgIcon>
);

const TrashIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M3 6h18" />
    <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
  </SvgIcon>
);

/* ── Hero Icon ──────────────────────────────────────────────── */
const PaletteIcon = () => (
  <SvgIcon size={28}>
    <circle cx="13.5" cy="6.5" r=".5" fill="currentColor" />
    <circle cx="17.5" cy="10.5" r=".5" fill="currentColor" />
    <circle cx="8.5" cy="7.5" r=".5" fill="currentColor" />
    <circle cx="6.5" cy="12.5" r=".5" fill="currentColor" />
    <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.93 0 1.5-.67 1.5-1.5 0-.39-.15-.74-.43-1.01-.29-.27-.57-.67-.57-1.09 0-.83.67-1.5 1.5-1.5H16c3.31 0 6-2.69 6-6 0-5.5-4.5-9.95-10-9.9Z" />
  </SvgIcon>
);

/* ── Status / Indicator Icons ───────────────────────────────── */
const AlertTriangleIcon = (props) => (
  <SvgIcon {...props}>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </SvgIcon>
);

const LightbulbIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
    <path d="M9 18h6" />
    <path d="M10 22h4" />
  </SvgIcon>
);

const BarChartIcon = (props) => (
  <SvgIcon {...props}>
    <line x1="12" x2="12" y1="20" y2="10" />
    <line x1="18" x2="18" y1="20" y2="4" />
    <line x1="6" x2="6" y1="20" y2="16" />
  </SvgIcon>
);

const CheckCircleIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <path d="m9 11 3 3L22 4" />
  </SvgIcon>
);

const CircleIcon = (props) => (
  <SvgIcon {...props}>
    <circle cx="12" cy="12" r="10" />
  </SvgIcon>
);

const ScaleIcon = (props) => (
  <SvgIcon {...props}>
    <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
    <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
    <path d="M7 21h10" />
    <path d="M12 3v18" />
    <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" />
  </SvgIcon>
);

const HeartIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
  </SvgIcon>
);

const XCircleIcon = (props) => (
  <SvgIcon {...props}>
    <circle cx="12" cy="12" r="10" />
    <path d="m15 9-6 6" />
    <path d="m9 9 6 6" />
  </SvgIcon>
);

const XIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </SvgIcon>
);

/* ═══════════════════════════════════════════════════════════════
   Main Component
   ═══════════════════════════════════════════════════════════════ */
export default function ColorPredictor() {
  /* ── State ──────────────────────────────────────────────────── */
  const [status, setStatus] = useState({ isLoading: true, isTraining: false, error: null });
  const [model, setModel] = useState(null);
  const [colors, setColors] = useState({ selected: [], unselected: [], current: '#6366f1' });
  const [prediction, setPrediction] = useState(null);
  const [toast, setToast] = useState({ visible: false, message: '', severity: 'success' });
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showClearDialog, setShowClearDialog] = useState(false);
  const [ratingHistory, setRatingHistory] = useState([]);
  const [modelStats, setModelStats] = useState({ trainedSamples: 0, accuracy: null, lastTrained: null });

  const toastTimerRef = useRef(null);

  const selectedColors = colors.selected;
  const unselectedColors = colors.unselected;
  const currentColor = colors.current;

  /* ── Computed ────────────────────────────────────────────────── */
  const stats = useMemo(() => ({
    totalSamples: selectedColors.length + unselectedColors.length,
    likedPercentage:
      selectedColors.length > 0
        ? ((selectedColors.length / (selectedColors.length + unselectedColors.length)) * 100).toFixed(1)
        : '0',
    canTrain: selectedColors.length >= 2 && unselectedColors.length >= 2,
  }), [selectedColors, unselectedColors]);

  const recentHistory = useMemo(() => ratingHistory.slice(-5).reverse(), [ratingHistory]);

  /* ── Toast helper ───────────────────────────────────────────── */
  const showToast = useCallback((message, severity = 'success') => {
    clearTimeout(toastTimerRef.current);
    setToast({ visible: true, message, severity });
    toastTimerRef.current = setTimeout(() => setToast((p) => ({ ...p, visible: false })), 2800);
  }, []);

  /* ── Persistence ────────────────────────────────────────────── */
  useEffect(() => {
    const dismissed = window.localStorage.getItem(ONBOARDING_KEY) === 'true';
    const persisted = safeParse(window.localStorage.getItem(SESSION_KEY), null);

    if (persisted) {
      setColors((prev) => ({
        ...prev,
        selected: Array.isArray(persisted.selected) ? persisted.selected : prev.selected,
        unselected: Array.isArray(persisted.unselected) ? persisted.unselected : prev.unselected,
        current: typeof persisted.current === 'string' ? persisted.current : prev.current,
      }));
      setRatingHistory(Array.isArray(persisted.ratingHistory) ? persisted.ratingHistory : []);
      if (persisted.modelStats && typeof persisted.modelStats === 'object') {
        setModelStats((prev) => ({ ...prev, ...persisted.modelStats }));
      }
    }
    setShowOnboarding(!dismissed);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      SESSION_KEY,
      JSON.stringify({ selected: selectedColors, unselected: unselectedColors, current: currentColor, ratingHistory, modelStats }),
    );
  }, [selectedColors, unselectedColors, currentColor, ratingHistory, modelStats]);

  /* ── Model init ─────────────────────────────────────────────── */
  useEffect(() => {
    (async () => {
      try {
        setStatus((p) => ({ ...p, isLoading: true }));
        try {
          const loaded = await loadModel(MODEL_ID);
          setModel(loaded);
        } catch {
          const fresh = createModel({ learningRate: 0.001, dropout: 0.2 });
          setModel(fresh);
        }
      } catch {
        setStatus((p) => ({ ...p, error: 'Failed to initialize model' }));
      } finally {
        setStatus((p) => ({ ...p, isLoading: false }));
      }
    })();
  }, []);

  /* ── Handlers ───────────────────────────────────────────────── */
  const dismissOnboarding = useCallback(() => {
    window.localStorage.setItem(ONBOARDING_KEY, 'true');
    setShowOnboarding(false);
  }, []);

  const handleColorSelect = useCallback(
    (liked) => {
      const color = currentColor;
      setColors((prev) => ({
        ...prev,
        selected: liked ? [...prev.selected, prev.current] : prev.selected,
        unselected: liked ? prev.unselected : [...prev.unselected, prev.current],
      }));
      setRatingHistory((prev) => [...prev, { color, liked, timestamp: Date.now() }]);
      showToast(liked ? 'Liked! Color sample added.' : 'Disliked! Color sample added.');
    },
    [currentColor, showToast],
  );

  const handleUndo = useCallback(() => {
    if (ratingHistory.length === 0) return;
    const last = ratingHistory[ratingHistory.length - 1];
    setRatingHistory((prev) => prev.slice(0, -1));
    setColors((prev) => ({
      ...prev,
      selected: last.liked ? removeLastMatch(prev.selected, last.color) : prev.selected,
      unselected: !last.liked ? removeLastMatch(prev.unselected, last.color) : prev.unselected,
    }));
    setPrediction(null);
    showToast(`Undid rating for ${last.color.toUpperCase()}`, 'info');
  }, [ratingHistory, showToast]);

  const handleClear = useCallback(() => {
    setColors({ selected: [], unselected: [], current: '#6366f1' });
    setRatingHistory([]);
    setPrediction(null);
    setModelStats({ trainedSamples: 0, accuracy: null, lastTrained: null });
    window.localStorage.removeItem(SESSION_KEY);
    setShowClearDialog(false);
    showToast('Session cleared.', 'info');
  }, [showToast]);

  const handleTrain = useCallback(async () => {
    if (status.isTraining || !model || !stats.canTrain) return;
    setStatus((p) => ({ ...p, isTraining: true, error: null }));
    try {
      if (!model.compiled) {
        model.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'binaryCrossentropy',
          metrics: ['accuracy'],
        });
      }
      const history = await trainModel(model, colors.selected, colors.unselected, { epochs: 50, batchSize: 32 });
      await saveModel(model, MODEL_ID);

      const accArr = history?.history?.accuracy ?? [];
      const accuracy = accArr.length > 0 ? accArr[accArr.length - 1] : 0;
      setModelStats({ trainedSamples: stats.totalSamples, accuracy: (accuracy * 100).toFixed(1), lastTrained: new Date().toLocaleString() });
      setPrediction(null);
      showToast('Model trained & saved!');
    } catch (error) {
      console.error('Training error:', error);
      setStatus((p) => ({ ...p, error: `Training failed: ${error.message || 'Unknown error'}` }));
      showToast('Training failed. Please try again.', 'error');
    } finally {
      setStatus((p) => ({ ...p, isTraining: false }));
    }
  }, [model, colors.selected, colors.unselected, stats.canTrain, stats.totalSamples, status.isTraining, showToast]);

  const handlePredict = useCallback(async () => {
    if (!model) return;
    try {
      const result = await predictColor(model, currentColor);
      setPrediction(result);
      showToast('Prediction generated!');
    } catch (error) {
      setStatus((p) => ({ ...p, error: 'Prediction failed: ' + error.message }));
      showToast('Prediction failed.', 'error');
    }
  }, [model, currentColor, showToast]);

  /* ── Keyboard Shortcuts ─────────────────────────────────────── */
  useEffect(() => {
    const onKey = (e) => {
      if (shouldIgnoreShortcut(e)) return;
      const k = e.key.toLowerCase();
      if (k === 'l') { e.preventDefault(); handleColorSelect(true); }
      else if (k === 'd') { e.preventDefault(); handleColorSelect(false); }
      else if (k === 't' && stats.canTrain && !status.isTraining) { e.preventDefault(); handleTrain(); }
      else if (k === 'p' && modelStats.lastTrained) { e.preventDefault(); handlePredict(); }
      else if (k === 'u' && ratingHistory.length > 0) { e.preventDefault(); handleUndo(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleColorSelect, handlePredict, handleTrain, handleUndo, modelStats.lastTrained, ratingHistory.length, stats.canTrain, status.isTraining]);

  /* ── Loading State ──────────────────────────────────────────── */
  if (status.isLoading) {
    return (
      <div className={s.loadingScreen}>
        <div className={s.loadingSpinner} />
        <span className={s.loadingText}>Loading Color Predictor&hellip;</span>
      </div>
    );
  }

  /* ══════════════════════════════════════════════════════════════
     Render
     ══════════════════════════════════════════════════════════════ */
  return (
    <div className={s.pageContainer}>
      <main id="main-content" aria-busy={status.isTraining}>

        {/* ── Hero ─────────────────────────────────────────────── */}
        <header className={s.hero}>
          <div className={s.heroIcon} aria-hidden="true"><PaletteIcon /></div>
          <h1 className={s.heroTitle}>Color Predictor</h1>
          <p className={s.heroSubtitle}>
            Teach a tiny neural network your color taste &mdash; right in your browser.
            No data ever leaves your device.
          </p>
        </header>

        {/* ── Error Alert ──────────────────────────────────────── */}
        {status.error && (
          <div className={s.alertError} role="alert">
            <AlertTriangleIcon size={18} className={s.alertIcon} />
            <span>{status.error}</span>
          </div>
        )}

        {/* ── Onboarding ───────────────────────────────────────── */}
        {showOnboarding && (
          <div className={s.alertInfo} role="status">
            <LightbulbIcon size={18} className={s.alertIcon} />
            <span>
              <strong>Quick start:</strong> Rate a few colors (at least 2 liked &amp; 2 disliked), then train and predict.
            </span>
            <button
              type="button"
              onClick={dismissOnboarding}
              className={s.alertCloseBtn}
              aria-label="Dismiss quick start tip"
            >
              <XIcon size={16} />
            </button>
          </div>
        )}

        {/* ── Summary Bar ──────────────────────────────────────── */}
        <div className={s.summaryCard}>
          <div className={s.summaryChips}>
            <span className={s.chip}>
              <BarChartIcon size={14} className={s.chipIconInline} />
              {stats.totalSamples} samples
            </span>
            <span className={modelStats.lastTrained ? s.chipSuccess : s.chip}>
              {modelStats.lastTrained
                ? <><CheckCircleIcon size={14} className={s.chipIconInline} /> Trained</>
                : <><CircleIcon size={14} className={s.chipIconInline} /> Not trained</>
              }
            </span>
            {prediction && (
              <span className={prediction.prediction === 'like' ? s.chipSuccess : s.chipDanger}>
                {prediction.prediction === 'like'
                  ? <><HeartIcon size={14} className={s.chipIconInline} /> Like</>
                  : <><XCircleIcon size={14} className={s.chipIconInline} /> Dislike</>
                }
              </span>
            )}
          </div>
          <button
            type="button"
            className={s.textBtn}
            onClick={() => setShowClearDialog(true)}
          >
            <TrashIcon size={16} />
            Clear
          </button>
        </div>

        {/* ── Step 1 – Pick & Rate ─────────────────────────────── */}
        <section className={s.card} aria-labelledby="step1-title">
          <div className={s.stepHeader}>
            <span className={s.stepBadge} aria-hidden="true">1</span>
            <h2 id="step1-title" className={s.stepTitle}>Choose &amp; rate a color</h2>
          </div>

          <ColorPicker
            currentColor={currentColor}
            onColorChange={(c) => setColors((prev) => ({ ...prev, current: c }))}
          />

          <div className={s.buttonRow}>
            <button type="button" className={s.btnLike} onClick={() => handleColorSelect(true)}>
              <ThumbUpIcon size={18} className={s.btnIcon} />
              I like this
            </button>
            <button type="button" className={s.btnDislike} onClick={() => handleColorSelect(false)}>
              <ThumbDownIcon size={18} className={s.btnIcon} />
              {"Don\u2019t like"}
            </button>
          </div>

          {/* Shortcut hints */}
          <div className={s.shortcutRow} aria-hidden="true">
            <span className={s.shortcutBadge}><kbd className={s.kbd}>L</kbd> like</span>
            <span className={s.shortcutBadge}><kbd className={s.kbd}>D</kbd> dislike</span>
            <span className={s.shortcutBadge}><kbd className={s.kbd}>U</kbd> undo</span>
          </div>

          {/* Recent ratings */}
          <div className={s.ratingRow}>
            <span className={s.ratingLabel}>Recent</span>
            {recentHistory.length > 0 ? (
              recentHistory.map((entry) => (
                <span
                  key={`${entry.timestamp}-${entry.color}`}
                  className={entry.liked ? s.ratingChipLike : s.ratingChipDislike}
                >
                  <span
                    className={s.ratingDot}
                    style={{ backgroundColor: entry.color }}
                  />
                  {entry.color.toUpperCase()}
                </span>
              ))
            ) : (
              <span style={{ fontSize: 13, color: 'var(--text-faint)' }}>No ratings yet</span>
            )}
            {ratingHistory.length > 0 && (
              <button type="button" className={s.textBtn} onClick={handleUndo}>
                <UndoIcon size={14} />
                Undo
              </button>
            )}
          </div>
        </section>

        {/* ── Step 2 – Train ───────────────────────────────────── */}
        <section className={s.card} aria-labelledby="step2-title">
          <div className={s.stepHeader}>
            <span className={s.stepBadge} aria-hidden="true">2</span>
            <h2 id="step2-title" className={s.stepTitle}>Train your model</h2>
          </div>

          <div className={s.statsRow}>
            <span className={s.chipSuccess}>
              <ThumbUpIcon size={14} className={s.chipIconInline} />
              {selectedColors.length} liked
            </span>
            <span className={s.chipDanger}>
              <ThumbDownIcon size={14} className={s.chipIconInline} />
              {unselectedColors.length} disliked
            </span>
            <span className={s.chip}>
              <ScaleIcon size={14} className={s.chipIconInline} />
              {stats.likedPercentage}% ratio
            </span>
          </div>

          {!stats.canTrain && (
            <p id="train-help" style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 16 }}>
              Rate at least <strong>2 liked</strong> and <strong>2 disliked</strong> colors to enable training.
            </p>
          )}

          {modelStats.lastTrained && (
            <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
              Last trained: {modelStats.lastTrained} &middot; Accuracy: {modelStats.accuracy}%
            </p>
          )}

          <div className={s.buttonRow}>
            <button
              type="button"
              className={s.btnTrain}
              onClick={handleTrain}
              disabled={!stats.canTrain || status.isTraining}
              aria-describedby="train-help"
            >
              <BrainIcon size={18} className={s.btnIcon} />
              {status.isTraining ? 'Training\u2026' : 'Train model'}
            </button>
            <button
              type="button"
              className={s.btnPredict}
              onClick={handlePredict}
              disabled={!modelStats.lastTrained}
            >
              <SparklesIcon size={18} className={s.btnIcon} />
              Predict
            </button>
          </div>

          {/* Shortcut hints */}
          <div className={s.shortcutRow} aria-hidden="true">
            <span className={s.shortcutBadge}><kbd className={s.kbd}>T</kbd> train</span>
            <span className={s.shortcutBadge}><kbd className={s.kbd}>P</kbd> predict</span>
          </div>

          {status.isTraining && (
            <div className={s.progressWrap} aria-live="polite">
              <div className={s.progressBar}>
                <div className={s.progressFill} style={{ width: '100%' }} />
              </div>
              <p className={s.progressText}>Training in progress&hellip; this usually takes a few seconds.</p>
            </div>
          )}

          <p className={s.disclaimer}>All training runs locally in your browser.</p>
        </section>

        {/* ── Step 3 – Prediction ──────────────────────────────── */}
        {prediction && (
          <section className={s.predictionCard} aria-labelledby="step3-title">
            <div className={s.stepHeader}>
              <span className={s.stepBadge} aria-hidden="true">3</span>
              <h2 id="step3-title" className={s.stepTitle}>Prediction result</h2>
            </div>
            <div className={s.predictionInner}>
              <div className={s.predictionSwatch} style={{ backgroundColor: currentColor }} />
              <div className={s.predictionBody}>
                <p className={s.predictionVerdict}>
                  {prediction.prediction === 'like' ? (
                    <><CheckCircleIcon size={18} className={s.verdictIcon} /> This color matches your taste!</>
                  ) : (
                    <><XCircleIcon size={18} className={s.verdictIcon} /> This color probably isn&apos;t for you.</>
                  )}
                </p>
                <p className={s.predictionMeta}>
                  Confidence: {(prediction.confidence * 100).toFixed(1)}% &middot; Score: {(prediction.score * 100).toFixed(1)}%
                </p>
                <div className={s.confidenceBar}>
                  <div
                    className={prediction.prediction === 'like' ? s.confidenceFillLike : s.confidenceFillDislike}
                    style={{ width: `${Math.max(4, prediction.confidence * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Footer ───────────────────────────────────────────── */}
        <footer className={s.footer}>
          <p className={s.footerText}>
            Built with TensorFlow.js &middot; Runs entirely in your browser &middot; No data leaves your device
          </p>
        </footer>

        {/* ── SR Live Region ───────────────────────────────────── */}
        <div aria-live="polite" role="status" className={s.srOnly}>
          {status.isTraining ? 'Model training in progress' : 'Model training idle'}
        </div>

        {/* ── Toast ────────────────────────────────────────────── */}
        <div
          className={toast.visible ? s.toastVisible : s.toast}
          role="status"
          aria-live="polite"
        >
          <span
            className={
              toast.severity === 'success' ? s.toastSuccess
                : toast.severity === 'error' ? s.toastError
                : s.toastInfo
            }
            style={{
              display: 'inline-block',
              padding: '12px 24px',
              borderRadius: 'var(--radius-md)',
            }}
          >
            {toast.message}
          </span>
        </div>

        {/* ── Clear Confirmation Dialog ────────────────────────── */}
        {showClearDialog && (
          <div
            className={s.overlay}
            onClick={(e) => { if (e.target === e.currentTarget) setShowClearDialog(false); }}
            role="dialog"
            aria-modal="true"
            aria-labelledby="clear-dialog-title"
            aria-describedby="clear-dialog-desc"
          >
            <div className={s.dialog}>
              <h3 id="clear-dialog-title" className={s.dialogTitle}>Clear session?</h3>
              <p id="clear-dialog-desc" className={s.dialogBody}>
                This will remove all color ratings, history, prediction output, and training stats.
                Your trained model will remain saved.
              </p>
              <div className={s.dialogActions}>
                <button
                  type="button"
                  className={s.dialogCancelBtn}
                  onClick={() => setShowClearDialog(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className={s.dialogDangerBtn}
                  onClick={handleClear}
                >
                  Clear session
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

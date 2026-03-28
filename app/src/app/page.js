'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  createModel,
  hexToRgb,
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
const THEME_KEY = 'color-predictor-theme-preference-v1';
const MIN_RATINGS_PER_CLASS = 2;
const TRAINING_EPOCHS = 50;
const TRAINING_BATCH_SIZE = 32;

const getTrainingUnlockLabel = () => `${MIN_RATINGS_PER_CLASS} liked and ${MIN_RATINGS_PER_CLASS} disliked colors`;

const getDefaultModelStats = () => ({
  trainedSamples: 0,
  accuracy: null,
  lastTrained: null,
  trainedDatasetSignature: null,
});

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

const addIfMissing = (items, value) => (items.includes(value) ? items : [...items, value]);

const getColorRating = (selectedItems, unselectedItems, color) => {
  if (selectedItems.includes(color)) return 'liked';
  if (unselectedItems.includes(color)) return 'disliked';
  return null;
};

const getAccuracyFeedback = (accuracyValue) => {
  const accuracy = Number.parseFloat(accuracyValue);

  if (!Number.isFinite(accuracy)) return null;
  if (accuracy >= 90) return 'Excellent fit';
  if (accuracy >= 75) return 'Learning your taste well';
  if (accuracy >= 60) return 'Promising start';
  return 'Needs more ratings';
};

const getPredictionConfidenceLabel = (confidence) => {
  if (confidence >= 0.85) return 'Very confident';
  if (confidence >= 0.65) return 'Fairly confident';
  if (confidence >= 0.45) return 'Still learning';
  return 'Low confidence';
};

const getPredictionExplanation = (prediction) => {
  if (!prediction) return '';

  const confidenceLabel = getPredictionConfidenceLabel(prediction.confidence);

  if (prediction.prediction === 'like') {
    return `${confidenceLabel} — the model sees this color as similar to shades you have liked before.`;
  }

  return `${confidenceLabel} — the model sees this color as closer to shades you have disliked before.`;
};

const getNextThemePreference = (currentPreference) => {
  if (currentPreference === 'system') return 'light';
  if (currentPreference === 'light') return 'dark';
  return 'system';
};

const getThemeButtonLabel = (themePreference) => {
  if (themePreference === 'system') return 'Theme: System';
  if (themePreference === 'light') return 'Theme: Light';
  return 'Theme: Dark';
};

const getErrorMessage = (error) => (error instanceof Error ? error.message : String(error || ''));

const createDatasetSignature = (selectedItems, unselectedItems) => JSON.stringify({
  selected: selectedItems
    .filter((item) => typeof item === 'string' && hexToRgb(item))
    .map((item) => item.toLowerCase())
    .sort(),
  unselected: unselectedItems
    .filter((item) => typeof item === 'string' && hexToRgb(item))
    .map((item) => item.toLowerCase())
    .sort(),
});

const sanitizeModelStats = (value) => {
  const defaults = getDefaultModelStats();

  if (!value || typeof value !== 'object') {
    return defaults;
  }

  const trainedSamples = Number(value.trainedSamples);
  const accuracyValue = value.accuracy === null || value.accuracy === undefined
    ? null
    : Number.parseFloat(value.accuracy);

  return {
    trainedSamples: Number.isFinite(trainedSamples) && trainedSamples >= 0 ? trainedSamples : defaults.trainedSamples,
    accuracy: Number.isFinite(accuracyValue) ? accuracyValue.toFixed(1) : defaults.accuracy,
    lastTrained: typeof value.lastTrained === 'string' && value.lastTrained ? value.lastTrained : defaults.lastTrained,
    trainedDatasetSignature:
      typeof value.trainedDatasetSignature === 'string' && value.trainedDatasetSignature
        ? value.trainedDatasetSignature
        : defaults.trainedDatasetSignature,
  };
};

const getInitializationErrorMessage = () => (
  'We could not prepare the model right now. Refresh the page to try again.'
);

const getTrainingErrorMessage = (error) => {
  const message = getErrorMessage(error).toLowerCase();

  if (message.includes('selected and unselected') || message.includes('contain data')) {
    return `Add more rated colors before training. In this app, training unlocks once you have at least ${getTrainingUnlockLabel()}.`;
  }

  if (message.includes('no valid colors')) {
    return 'Your saved ratings could not be read. Clear the session and rate a few colors again.';
  }

  if (message.includes('indexeddb') || message.includes('save model')) {
    return 'Training finished, but the model could not be saved on this device. Try again or keep using the session without saving.';
  }

  return 'Training could not finish this time. Try rating a few more colors and run training again.';
};

const getPredictionErrorMessage = (error) => {
  const message = getErrorMessage(error).toLowerCase();

  if (message.includes('invalid color format')) {
    return 'Pick a valid color before asking for a prediction.';
  }

  return 'The model could not make a prediction right now. Try a different color or train the model again.';
};

const getModelStatus = ({ model, status, modelStats, stats, hasTrainedModel }) => {
  if (status.isLoading) {
    return {
      label: 'Model loading',
      detail: 'Preparing the in-browser neural network.',
      tone: 'neutral',
      ready: false,
    };
  }

  if (!model) {
    return {
      label: 'Model unavailable',
      detail: 'Refresh the page to try initializing it again.',
      tone: 'danger',
      ready: false,
    };
  }

  if (status.isTraining) {
    return {
      label: 'Training in progress',
      detail: `Learning from ${stats.totalSamples} rated colors.`,
      tone: 'accent',
      ready: false,
    };
  }

  if (status.isPredicting) {
    return {
      label: 'Analyzing color',
      detail: 'Generating a prediction for the current shade.',
      tone: 'accent',
      ready: false,
    };
  }

  if (modelStats.lastTrained && !hasTrainedModel) {
    return {
      label: 'Saved model unavailable',
      detail: 'Training details were restored, but the saved model could not be loaded on this device. Train again to enable predictions.',
      tone: 'danger',
      ready: false,
    };
  }

  if (hasTrainedModel) {
    return {
      label: 'Model ready',
      detail: 'Trained and ready to predict new colors.',
      tone: 'success',
      ready: true,
    };
  }

  return {
    label: 'Ready to learn',
    detail: 'Rate a few colors to unlock training.',
    tone: 'neutral',
    ready: false,
  };
};

const getTrainingGuidance = ({ selectedCount, unselectedCount, totalSamples, canTrain, modelStats, currentDatasetSignature }) => {
  const likedNeeded = Math.max(0, MIN_RATINGS_PER_CLASS - selectedCount);
  const dislikedNeeded = Math.max(0, MIN_RATINGS_PER_CLASS - unselectedCount);

  if (!canTrain) {
    const missingGroups = [];

    if (likedNeeded > 0) {
      missingGroups.push(`${likedNeeded} more liked ${likedNeeded === 1 ? 'color' : 'colors'}`);
    }

    if (dislikedNeeded > 0) {
      missingGroups.push(`${dislikedNeeded} more disliked ${dislikedNeeded === 1 ? 'color' : 'colors'}`);
    }

    return {
      tone: 'neutral',
      title: 'More examples needed',
      detail: `Add ${missingGroups.join(' and ')} to unlock training. In this app, training unlocks once you have at least ${getTrainingUnlockLabel()}.`,
    };
  }

  if (!modelStats.lastTrained) {
    return {
      tone: 'accent',
      title: 'Ready for a first training run',
      detail: `You can train now with ${totalSamples} rated colors. That is enough for a first pass, and more varied ratings usually make future predictions steadier.`,
    };
  }

  if (!modelStats.trainedDatasetSignature) {
    return {
      tone: 'accent',
      title: 'Retraining recommended',
      detail: 'This saved model was trained before rating-change tracking was available. Train again to sync it with your current liked and disliked colors.',
    };
  }

  if (modelStats.trainedDatasetSignature !== currentDatasetSignature) {
    return {
      tone: 'accent',
      title: 'Ratings changed since training',
      detail: 'Your rated colors changed since the last training run. Retrain to include the latest labels and examples.',
    };
  }

  return {
    tone: 'neutral',
    title: 'Model is up to date',
    detail: `This model was trained on ${modelStats.trainedSamples} rated colors. Add more varied colors and retrain whenever you want a stronger signal.`,
  };
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

const MonitorIcon = (props) => (
  <SvgIcon {...props}>
    <rect x="3" y="4" width="18" height="12" rx="2" />
    <path d="M8 20h8" />
    <path d="M12 16v4" />
  </SvgIcon>
);

const SunIcon = (props) => (
  <SvgIcon {...props}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m6.34 17.66-1.41 1.41" />
    <path d="m19.07 4.93-1.41 1.41" />
  </SvgIcon>
);

const MoonIcon = (props) => (
  <SvgIcon {...props}>
    <path d="M21 12.79A9 9 0 1 1 11.21 3c0 4.97 4.03 9 9 9 .27 0 .53-.01.79-.04Z" />
  </SvgIcon>
);

/* ═══════════════════════════════════════════════════════════════
   Main Component
   ═══════════════════════════════════════════════════════════════ */
export default function ColorPredictor() {
  /* ── State ──────────────────────────────────────────────────── */
  const [status, setStatus] = useState({ isLoading: true, isTraining: false, isPredicting: false, error: null });
  const [model, setModel] = useState(null);
  const [colors, setColors] = useState({ selected: [], unselected: [], current: '#6366f1' });
  const [prediction, setPrediction] = useState(null);
  const [predictionTargetColor, setPredictionTargetColor] = useState(null);
  const [toast, setToast] = useState({ visible: false, message: '', severity: 'success' });
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showClearDialog, setShowClearDialog] = useState(false);
  const [ratingHistory, setRatingHistory] = useState([]);
  const [modelStats, setModelStats] = useState(getDefaultModelStats);
  const [hasTrainedModel, setHasTrainedModel] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState({ currentEpoch: 0, totalEpochs: TRAINING_EPOCHS, accuracy: null });
  const [showTrainingCelebration, setShowTrainingCelebration] = useState(false);
  const [themePreference, setThemePreference] = useState('system');

  const toastTimerRef = useRef(null);
  const dialogRef = useRef(null);
  const lastFocusedElementRef = useRef(null);
  const trainingCelebrationTimerRef = useRef(null);

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
    canTrain: selectedColors.length >= MIN_RATINGS_PER_CLASS && unselectedColors.length >= MIN_RATINGS_PER_CLASS,
  }), [selectedColors, unselectedColors]);

  const recentHistory = useMemo(() => ratingHistory.slice(-5).reverse(), [ratingHistory]);
  const trainingProgressPercentage = useMemo(
    () => (trainingProgress.totalEpochs > 0
      ? Math.min(100, (trainingProgress.currentEpoch / trainingProgress.totalEpochs) * 100)
      : 0),
    [trainingProgress.currentEpoch, trainingProgress.totalEpochs],
  );
  const accuracyFeedback = useMemo(() => getAccuracyFeedback(modelStats.accuracy), [modelStats.accuracy]);
  const predictionExplanation = useMemo(() => getPredictionExplanation(prediction), [prediction]);
  const displayedPredictionColor = prediction?.color ?? predictionTargetColor ?? currentColor;
  const themeButtonLabel = useMemo(() => getThemeButtonLabel(themePreference), [themePreference]);
  const currentColorRating = useMemo(
    () => getColorRating(selectedColors, unselectedColors, currentColor),
    [currentColor, selectedColors, unselectedColors],
  );
  const currentDatasetSignature = useMemo(
    () => createDatasetSignature(selectedColors, unselectedColors),
    [selectedColors, unselectedColors],
  );
  const canTrainFromShortcut = stats.canTrain && !status.isTraining && !status.isPredicting;
  const canPredict = hasTrainedModel && !status.isPredicting && !status.isTraining;
  const modelStatus = useMemo(
    () => getModelStatus({ model, status, modelStats, stats, hasTrainedModel }),
    [hasTrainedModel, model, modelStats, stats, status],
  );
  const trainingGuidance = useMemo(
    () => getTrainingGuidance({
      selectedCount: selectedColors.length,
      unselectedCount: unselectedColors.length,
      totalSamples: stats.totalSamples,
      canTrain: stats.canTrain,
      modelStats,
      currentDatasetSignature,
    }),
    [currentDatasetSignature, modelStats, selectedColors.length, stats.canTrain, stats.totalSamples, unselectedColors.length],
  );

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
    const persistedThemePreference = window.localStorage.getItem(THEME_KEY);

    if (persisted) {
      setColors((prev) => ({
        ...prev,
        selected: Array.isArray(persisted.selected) ? persisted.selected : prev.selected,
        unselected: Array.isArray(persisted.unselected) ? persisted.unselected : prev.unselected,
        current: typeof persisted.current === 'string' ? persisted.current : prev.current,
      }));
      setRatingHistory(Array.isArray(persisted.ratingHistory) ? persisted.ratingHistory : []);
      setModelStats(sanitizeModelStats(persisted.modelStats));
    }

    if (persistedThemePreference === 'light' || persistedThemePreference === 'dark' || persistedThemePreference === 'system') {
      setThemePreference(persistedThemePreference);
    }

    setShowOnboarding(!dismissed);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      SESSION_KEY,
      JSON.stringify({ selected: selectedColors, unselected: unselectedColors, current: currentColor, ratingHistory, modelStats }),
    );
  }, [selectedColors, unselectedColors, currentColor, ratingHistory, modelStats]);

  useEffect(() => {
    const root = document.documentElement;
    window.localStorage.setItem(THEME_KEY, themePreference);

    if (themePreference === 'system') {
      delete root.dataset.theme;
      return;
    }

    root.dataset.theme = themePreference;
  }, [themePreference]);

  useEffect(() => () => {
    window.clearTimeout(trainingCelebrationTimerRef.current);
  }, []);

  useEffect(() => {
    if (status.isPredicting) return;
    setPrediction(null);
    setPredictionTargetColor(null);
  }, [currentColor, status.isPredicting]);

  useEffect(() => {
    if (!showClearDialog) return;

    lastFocusedElementRef.current = document.activeElement;

    const dialog = dialogRef.current;
    if (!dialog) return;

    const focusable = dialog.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    first?.focus();

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setShowClearDialog(false);
        return;
      }

      if (event.key !== 'Tab' || focusable.length === 0) return;

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };

    dialog.addEventListener('keydown', handleKeyDown);

    return () => {
      dialog.removeEventListener('keydown', handleKeyDown);
      lastFocusedElementRef.current?.focus?.();
    };
  }, [showClearDialog]);

  /* ── Model init ─────────────────────────────────────────────── */
  useEffect(() => {
    (async () => {
      try {
        setStatus((p) => ({ ...p, isLoading: true }));
        try {
          const loaded = await loadModel(MODEL_ID);
          setModel(loaded);
          setHasTrainedModel(true);
        } catch {
          const fresh = createModel({ learningRate: 0.001, dropout: 0.2 });
          setModel(fresh);
          setHasTrainedModel(false);
        }
      } catch (error) {
        console.error('Model initialization error:', error);
        setStatus((p) => ({ ...p, error: getInitializationErrorMessage() }));
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
      const existingRating = getColorRating(selectedColors, unselectedColors, color);
      const nextRating = liked ? 'liked' : 'disliked';

      if (existingRating === nextRating) {
        showToast(`You already ${nextRating} ${color.toUpperCase()}.`, 'info');
        return;
      }

      setColors((prev) => ({
        ...prev,
        selected: liked
          ? addIfMissing(removeLastMatch(prev.selected, prev.current), prev.current)
          : removeLastMatch(prev.selected, prev.current),
        unselected: liked
          ? removeLastMatch(prev.unselected, prev.current)
          : addIfMissing(removeLastMatch(prev.unselected, prev.current), prev.current),
      }));
      setRatingHistory((prev) => [...prev, {
        color,
        liked,
        timestamp: Date.now(),
        action: existingRating ? 'switch' : 'add',
      }]);
      showToast(
        existingRating
          ? `Updated rating for ${color.toUpperCase()}.`
          : liked
            ? 'Liked! Color sample added.'
            : 'Disliked! Color sample added.',
        existingRating ? 'info' : 'success',
      );
    },
    [currentColor, selectedColors, showToast, unselectedColors],
  );

  const handleUndo = useCallback(() => {
    if (ratingHistory.length === 0) return;
    const last = ratingHistory[ratingHistory.length - 1];
    setRatingHistory((prev) => prev.slice(0, -1));
    setColors((prev) => ({
      ...prev,
      selected: last.action === 'switch'
        ? last.liked
          ? removeLastMatch(prev.selected, last.color)
          : addIfMissing(removeLastMatch(prev.selected, last.color), last.color)
        : last.liked
          ? removeLastMatch(prev.selected, last.color)
          : prev.selected,
      unselected: last.action === 'switch'
        ? last.liked
          ? addIfMissing(removeLastMatch(prev.unselected, last.color), last.color)
          : removeLastMatch(prev.unselected, last.color)
        : !last.liked
          ? removeLastMatch(prev.unselected, last.color)
          : prev.unselected,
    }));
    setPrediction(null);
    setPredictionTargetColor(null);
    showToast(
      last.action === 'switch'
        ? `Restored previous rating for ${last.color.toUpperCase()}`
        : `Undid rating for ${last.color.toUpperCase()}`,
      'info',
    );
  }, [ratingHistory, showToast]);

  const handleClear = useCallback(() => {
    window.clearTimeout(trainingCelebrationTimerRef.current);
    setColors({ selected: [], unselected: [], current: '#6366f1' });
    setRatingHistory([]);
    setPrediction(null);
    setPredictionTargetColor(null);
    setModelStats(getDefaultModelStats());
    setShowTrainingCelebration(false);
    window.localStorage.removeItem(SESSION_KEY);
    setShowClearDialog(false);
    showToast(
      hasTrainedModel
        ? 'Session cleared. Your saved model is still stored on this device.'
        : 'Session cleared.',
      'info',
    );
  }, [hasTrainedModel, showToast]);

  const handleTrain = useCallback(async () => {
    if (status.isTraining || !model || !stats.canTrain) return;
    setStatus((p) => ({ ...p, isTraining: true, error: null }));
    setTrainingProgress({ currentEpoch: 0, totalEpochs: TRAINING_EPOCHS, accuracy: null });
    setShowTrainingCelebration(false);
    try {
      if (!model.compiled) {
        model.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'binaryCrossentropy',
          metrics: ['accuracy'],
        });
      }
      const history = await trainModel(model, colors.selected, colors.unselected, {
        epochs: TRAINING_EPOCHS,
        batchSize: TRAINING_BATCH_SIZE,
        onEpochEnd: (epoch, logs) => {
          const rawAccuracy = logs?.accuracy ?? logs?.acc ?? null;
          setTrainingProgress({
            currentEpoch: epoch + 1,
            totalEpochs: TRAINING_EPOCHS,
            accuracy: typeof rawAccuracy === 'number' ? rawAccuracy * 100 : null,
          });
        },
      });
      await saveModel(model, MODEL_ID);

      const accArr = history?.history?.accuracy ?? [];
      const accuracy = accArr.length > 0 ? accArr[accArr.length - 1] : 0;
      const formattedAccuracy = (accuracy * 100).toFixed(1);
      const feedback = getAccuracyFeedback(formattedAccuracy);
      setModelStats({
        trainedSamples: stats.totalSamples,
        accuracy: formattedAccuracy,
        lastTrained: new Date().toLocaleString(),
        trainedDatasetSignature: currentDatasetSignature,
      });
      setHasTrainedModel(true);
      setPrediction(null);
      setPredictionTargetColor(null);
      setShowTrainingCelebration(true);
      window.clearTimeout(trainingCelebrationTimerRef.current);
      trainingCelebrationTimerRef.current = window.setTimeout(() => {
        setShowTrainingCelebration(false);
      }, 3200);
      showToast(feedback ? `Model trained & saved — ${feedback}.` : 'Model trained & saved!');
    } catch (error) {
      console.error('Training error:', error);
      const safeMessage = getTrainingErrorMessage(error);
      setStatus((p) => ({ ...p, error: safeMessage }));
      showToast(safeMessage, 'error');
    } finally {
      setStatus((p) => ({ ...p, isTraining: false }));
    }
  }, [currentDatasetSignature, model, colors.selected, colors.unselected, stats.canTrain, stats.totalSamples, status.isTraining, showToast]);

  const handlePredict = useCallback(async () => {
    if (!model || status.isTraining || status.isPredicting) return;

    const colorToPredict = currentColor;
    setStatus((p) => ({ ...p, isPredicting: true, error: null }));
    setPrediction(null);
    setPredictionTargetColor(colorToPredict);

    try {
      const [result] = await Promise.all([
        Promise.resolve(predictColor(model, colorToPredict)),
        new Promise((resolve) => window.setTimeout(resolve, 320)),
      ]);

      setPrediction({ ...result, color: colorToPredict });
      showToast('Prediction generated!');
    } catch (error) {
      console.error('Prediction error:', error);
      const safeMessage = getPredictionErrorMessage(error);
      setStatus((p) => ({ ...p, error: safeMessage }));
      setPredictionTargetColor(null);
      showToast(safeMessage, 'error');
    } finally {
      setStatus((p) => ({ ...p, isPredicting: false }));
    }
  }, [currentColor, model, showToast, status.isPredicting, status.isTraining]);

  const handleThemeToggle = useCallback(() => {
    setThemePreference((currentPreference) => getNextThemePreference(currentPreference));
  }, []);

  /* ── Keyboard Shortcuts ─────────────────────────────────────── */
  useEffect(() => {
    const onKey = (e) => {
      if (shouldIgnoreShortcut(e)) return;
      const k = e.key.toLowerCase();
      if (k === 'l') { e.preventDefault(); handleColorSelect(true); }
      else if (k === 'd') { e.preventDefault(); handleColorSelect(false); }
      else if (k === 't' && canTrainFromShortcut) { e.preventDefault(); handleTrain(); }
      else if (k === 'p' && canPredict) { e.preventDefault(); handlePredict(); }
      else if (k === 'u' && ratingHistory.length > 0) { e.preventDefault(); handleUndo(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [canPredict, canTrainFromShortcut, handleColorSelect, handlePredict, handleTrain, handleUndo, ratingHistory.length]);

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
              <strong>Quick start:</strong> Rate a few colors (at least {getTrainingUnlockLabel()}), then train and predict.
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
        <section className={s.summaryCard} aria-labelledby="summary-title">
          <div className={s.summaryContent}>
            <div className={s.summaryStatus}>
              <p id="summary-title" className={s.summaryLabel}>Model status</p>
              <p className={s.summaryTitle}>
                {modelStatus.ready
                  ? <><CheckCircleIcon size={16} className={s.chipIconInline} /> {modelStatus.label}</>
                  : <><CircleIcon size={16} className={s.chipIconInline} /> {modelStatus.label}</>}
              </p>
              <p className={s.summaryDescription}>{modelStatus.detail}</p>
            </div>

            <div className={s.summaryChips} role="list" aria-label="Session summary">
              <span className={s.chip}>
                <BarChartIcon size={14} className={s.chipIconInline} />
                {stats.totalSamples} samples
              </span>
              <span className={modelStatus.tone === 'success' ? s.chipSuccess : modelStatus.tone === 'danger' ? s.chipDanger : modelStatus.tone === 'accent' ? s.chipAccent : s.chip}>
                {modelStatus.ready
                  ? <><CheckCircleIcon size={14} className={s.chipIconInline} /> {modelStatus.label}</>
                  : <><CircleIcon size={14} className={s.chipIconInline} /> {modelStatus.label}</>}
              </span>
              {prediction && (
                <span className={prediction.prediction === 'like' ? s.chipSuccess : s.chipDanger}>
                  {prediction.prediction === 'like'
                    ? <><HeartIcon size={14} className={s.chipIconInline} /> Like</>
                    : <><XCircleIcon size={14} className={s.chipIconInline} /> Dislike</>}
                </span>
              )}
              {showTrainingCelebration && (
                <span className={s.chipAccent}>
                  <SparklesIcon size={14} className={s.chipIconInline} /> Freshly trained
                </span>
              )}
            </div>

            <details className={s.shortcutHelp}>
              <summary className={s.shortcutHelpSummary}>Keyboard shortcuts</summary>
              <div className={s.shortcutHelpBody}>
                <p>Use single-key shortcuts when your cursor is not inside a text field.</p>
                <ul className={s.shortcutHelpList}>
                  <li><kbd className={s.kbd}>L</kbd> like the current color</li>
                  <li><kbd className={s.kbd}>D</kbd> dislike the current color</li>
                  <li><kbd className={s.kbd}>U</kbd> undo the last rating</li>
                  <li><kbd className={s.kbd}>T</kbd> train the model</li>
                  <li><kbd className={s.kbd}>P</kbd> predict for the current color</li>
                </ul>
              </div>
            </details>
          </div>
          <button
            type="button"
            className={s.themeToggle}
            onClick={handleThemeToggle}
            aria-label={`${themeButtonLabel}. Activate to switch theme mode.`}
            title={themeButtonLabel}
          >
            <span className={s.themeToggleIcon} aria-hidden="true">
              {themePreference === 'system'
                ? <MonitorIcon size={16} />
                : themePreference === 'light'
                  ? <SunIcon size={16} />
                  : <MoonIcon size={16} />}
            </span>
            <span className={s.themeToggleText}>{themeButtonLabel}</span>
          </button>
          <button
            type="button"
            className={s.textBtn}
            onClick={() => setShowClearDialog(true)}
          >
            <TrashIcon size={16} />
            Clear
          </button>
        </section>

        {/* ── Step 1 – Pick & Rate ─────────────────────────────── */}
        <section className={s.card} aria-labelledby="step1-title">
          <div className={s.stepHeader}>
            <span className={s.stepBadge} aria-hidden="true">1</span>
            <h2 id="step1-title" className={s.stepTitle}>Choose &amp; rate a color</h2>
          </div>

          <details className={s.learnMore}>
            <summary className={s.learnMoreSummary}>What&apos;s happening in this step?</summary>
            <div className={s.learnMoreBody}>
              <p>
                Every time you rate a color, you create a labeled example for the model.
                Liked colors become positive examples, and disliked colors become negative ones.
              </p>
              <p>
                The more varied your ratings are, the easier it becomes for the neural network to spot patterns in your taste.
              </p>
            </div>
          </details>

          <ColorPicker
            currentColor={currentColor}
            onColorChange={(c) => setColors((prev) => ({ ...prev, current: c }))}
          />

          {currentColorRating && (
            <p className={s.currentRatingHint} role="status" aria-live="polite">
              This color is already marked as <strong>{currentColorRating}</strong>. Rate it again to change its label.
            </p>
          )}

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

          <details className={s.learnMore}>
            <summary className={s.learnMoreSummary}>What does training do?</summary>
            <div className={s.learnMoreBody}>
              <p>
                Training runs all of your rated colors through the neural network over several rounds called epochs.
              </p>
              <p>
                On each round, the model adjusts its internal weights so it can better separate colors you like from colors you do not.
              </p>
            </div>
          </details>

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

          <div
            id="train-guidance"
            className={trainingGuidance.tone === 'accent' ? s.trainingGuidanceAccent : s.trainingGuidanceNeutral}
          >
            <LightbulbIcon size={18} className={s.trainingGuidanceIcon} />
            <div>
              <p className={s.trainingGuidanceTitle}>{trainingGuidance.title}</p>
              <p className={s.trainingGuidanceText}>{trainingGuidance.detail}</p>
            </div>
          </div>

          {modelStats.lastTrained && (
            <p className={s.trainingMeta}>
              Last trained: {modelStats.lastTrained} &middot; Accuracy: {modelStats.accuracy}%
              {!hasTrainedModel ? ' — Saved model needs retraining on this device' : ''}
              {accuracyFeedback ? ` — ${accuracyFeedback}` : ''}
            </p>
          )}

          <div className={s.buttonRow}>
            <button
              type="button"
              className={s.btnTrain}
              onClick={handleTrain}
              disabled={!stats.canTrain || status.isTraining}
              aria-describedby="train-guidance"
            >
              <BrainIcon size={18} className={s.btnIcon} />
              {status.isTraining ? 'Training\u2026' : 'Train model'}
            </button>
            <button
              type="button"
              className={s.btnPredict}
              onClick={handlePredict}
              disabled={!canPredict}
            >
              <SparklesIcon size={18} className={s.btnIcon} />
              {status.isPredicting ? 'Analyzing…' : 'Predict'}
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
                <div className={s.progressFill} style={{ width: `${Math.max(4, trainingProgressPercentage)}%` }} />
              </div>
              <p className={s.progressText}>
                Epoch {Math.max(trainingProgress.currentEpoch, 1)} of {trainingProgress.totalEpochs}
                {typeof trainingProgress.accuracy === 'number'
                  ? ` · Current accuracy ${trainingProgress.accuracy.toFixed(1)}%`
                  : ''}
              </p>
            </div>
          )}

          {showTrainingCelebration && !status.isTraining && (
            <div className={s.trainingCelebration} role="status" aria-live="polite">
              <SparklesIcon size={18} className={s.trainingCelebrationIcon} />
              <div>
                <p className={s.trainingCelebrationTitle}>Your model just learned from {stats.totalSamples} rated colors.</p>
                <p className={s.trainingCelebrationText}>
                  {accuracyFeedback
                    ? `${accuracyFeedback}. Try a prediction or rate more colors to improve it further.`
                    : 'Try a prediction or rate more colors to improve it further.'}
                </p>
              </div>
            </div>
          )}

          <p className={s.disclaimer}>All training runs locally in your browser.</p>
        </section>

        {/* ── Step 3 – Prediction ──────────────────────────────── */}
        {(status.isPredicting || prediction) && (
          <section className={s.predictionCard} aria-labelledby="step3-title">
            <div className={s.stepHeader}>
              <span className={s.stepBadge} aria-hidden="true">3</span>
              <h2 id="step3-title" className={s.stepTitle}>{status.isPredicting ? 'Analyzing color' : 'Prediction result'}</h2>
            </div>

            {!status.isPredicting && (
              <details className={s.learnMore}>
                <summary className={s.learnMoreSummary}>How should I read this prediction?</summary>
                <div className={s.learnMoreBody}>
                  <p>
                    The model converts the current color into red, green, and blue numbers, then estimates whether that pattern matches your past ratings.
                  </p>
                  <p>
                    Confidence tells you how sure the model feels about this guess. If the confidence is low, rate more colors and train again.
                  </p>
                </div>
              </details>
            )}

            <div className={s.predictionInner}>
              <div className={s.predictionSwatch} style={{ backgroundColor: displayedPredictionColor }} />
              <div className={s.predictionBody}>
                {status.isPredicting ? (
                  <>
                    <p className={s.predictionVerdict}>Analyzing this color…</p>
                    <p className={s.predictionMeta}>
                      The model is comparing this shade with the colors you rated before.
                    </p>
                    <div className={s.predictionLoadingBar} aria-hidden="true">
                      <div className={s.predictionLoadingFill} />
                    </div>
                  </>
                ) : (
                  <>
                    <p className={s.predictionVerdict}>
                      {prediction.prediction === 'like' ? (
                        <><CheckCircleIcon size={18} className={s.verdictIcon} /> This color matches your taste!</>
                      ) : (
                        <><XCircleIcon size={18} className={s.verdictIcon} /> This color probably isn&apos;t for you.</>
                      )}
                    </p>
                    <p className={s.predictionMeta}>
                      {predictionExplanation}
                    </p>
                    <p className={s.predictionStats}>
                      Confidence: {(prediction.confidence * 100).toFixed(1)}% &middot; Model score: {(prediction.score * 100).toFixed(1)}%
                    </p>
                    <div className={s.confidenceBar}>
                      <div
                        className={prediction.prediction === 'like' ? s.confidenceFillLike : s.confidenceFillDislike}
                        style={{ width: `${Math.max(4, prediction.confidence * 100)}%` }}
                      />
                    </div>
                  </>
                )}
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
          {status.isPredicting ? ' Generating prediction for the current color.' : ''}
          {prediction
            ? ` Prediction: the model thinks you would ${prediction.prediction} this color with ${(prediction.confidence * 100).toFixed(0)} percent confidence.`
            : ''}
        </div>

        {/* ── Toast ────────────────────────────────────────────── */}
        <div
          className={`${toast.visible ? s.toastVisible : s.toast} ${
            toast.severity === 'success' ? s.toastSuccess
              : toast.severity === 'error' ? s.toastError
              : s.toastInfo
          }`}
          role="status"
          aria-live="polite"
        >
          {toast.message}
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
            <div className={s.dialog} ref={dialogRef}>
              <h3 id="clear-dialog-title" className={s.dialogTitle}>Clear session?</h3>
              <p id="clear-dialog-desc" className={s.dialogBody}>
                This will remove all color ratings, history, prediction output, and training stats.
                Any saved model stays stored on this device unless you remove your browser data.
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

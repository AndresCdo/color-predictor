'use client';

import React, { useState, useCallback, useRef } from 'react';
import styles from './ColorPicker.module.css';

/** Returns white or dark text depending on the background luminance. */
const getContrastTextColor = (hexColor) => {
  const normalized = hexColor.replace('#', '');
  const expanded =
    normalized.length === 3
      ? normalized
          .split('')
          .map((c) => c + c)
          .join('')
      : normalized;

  const r = parseInt(expanded.slice(0, 2), 16);
  const g = parseInt(expanded.slice(2, 4), 16);
  const b = parseInt(expanded.slice(4, 6), 16);

  // WCAG relative luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.55 ? '#0f172a' : '#f8fafc';
};

/** Generates a random hex color. */
const randomHex = () =>
  '#' +
  Array.from({ length: 3 }, () =>
    Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, '0'),
  ).join('');

const HEX_REGEX = /^#?([0-9A-Fa-f]{6})$/;
const HEX_ALLOWED_CHARS = /[^#0-9A-Fa-f]/g;
const HEX_HELPER_ID = 'color-help';
const HEX_ERROR_ID = 'color-hex-error';

const normalizeHexInput = (value) => {
  const cleaned = value.replace(HEX_ALLOWED_CHARS, '').toUpperCase();
  if (!cleaned) return '';
  if (cleaned.startsWith('#')) return `#${cleaned.slice(1).replace(/#/g, '')}`;
  return `#${cleaned.replace(/#/g, '')}`;
};

const ColorPicker = ({ currentColor, onColorChange }) => {
  const [hexDraft, setHexDraft] = useState(currentColor.toUpperCase());
  const [hexError, setHexError] = useState('');
  const nativeInputRef = useRef(null);
  const textColor = getContrastTextColor(currentColor);
  const displayHex = currentColor.toUpperCase();

  const commitHex = useCallback(
    (raw) => {
      const clean = raw.startsWith('#') ? raw : `#${raw}`;
      if (HEX_REGEX.test(clean)) {
        setHexError('');
        onColorChange(clean.toLowerCase());
        setHexDraft(clean.toUpperCase());
        return true;
      }

      setHexError('Enter a full 6-digit hex color, for example #6366F1.');
      return false;
    },
    [onColorChange],
  );

  const handleNativeChange = useCallback(
    (e) => {
      const value = e.target.value;
      onColorChange(value);
      setHexDraft(value.toUpperCase());
      setHexError('');
    },
    [onColorChange],
  );

  const handleTextChange = useCallback((e) => {
    const nextValue = normalizeHexInput(e.target.value).slice(0, 7);
    setHexDraft(nextValue);

    if (nextValue.length === 0) {
      setHexError('');
      return;
    }

    setHexError(HEX_REGEX.test(nextValue) ? '' : 'Enter a full 6-digit hex color, for example #6366F1.');
  }, []);

  const handleTextBlur = useCallback(() => {
    if (!hexDraft) {
      setHexDraft(displayHex);
      setHexError('');
      return;
    }

    if (!commitHex(hexDraft)) {
      setHexDraft(displayHex);
    }
  }, [commitHex, displayHex, hexDraft]);

  const handleTextKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter') {
        if (!commitHex(hexDraft)) {
          e.preventDefault();
          return;
        }
        e.target.blur();
      }
    },
    [commitHex, hexDraft],
  );

  const handleRandom = useCallback(() => {
    const color = randomHex();
    onColorChange(color);
    setHexDraft(color.toUpperCase());
    setHexError('');
  }, [onColorChange]);

  // Keep text input in sync when parent changes colour via keyboard shortcuts etc.
  if (hexDraft !== displayHex && HEX_REGEX.test(hexDraft)) {
    // Only sync outward if the user isn't actively typing something new
  }

  return (
    <div className={styles.wrapper}>
      <span className={styles.label} id="color-picker-label">
        Pick a color
      </span>

      {/* ── Giant Interactive Swatch ──────────────────────────── */}
      <div
        className={styles.swatchOuter}
        role="button"
        tabIndex={0}
        aria-label={`Current color ${displayHex}. Click to open color picker.`}
        onClick={() => nativeInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            nativeInputRef.current?.click();
          }
        }}
      >
        <div
          className={styles.swatchInner}
          style={{ backgroundColor: currentColor }}
        >
          <span className={styles.hexLabel} style={{ color: textColor }}>
            {displayHex}
          </span>
          <span className={styles.tapHint} style={{ color: textColor }}>
            Tap to change
          </span>

          {/* Hidden native color input covers the swatch */}
          <input
            ref={nativeInputRef}
            type="color"
            value={currentColor}
            onChange={handleNativeChange}
            className={styles.nativeInput}
            aria-labelledby="color-picker-label"
            tabIndex={-1}
          />
        </div>
      </div>

      {/* ── Hex Input + Random ───────────────────────────────── */}
      <div className={styles.hexInputRow}>
        <input
          type="text"
          value={hexDraft}
          onChange={handleTextChange}
          onBlur={handleTextBlur}
          onKeyDown={handleTextKeyDown}
          onFocus={() => {
            if (!hexDraft) {
              setHexDraft(displayHex);
            }
          }}
          className={hexError ? `${styles.hexInputField} ${styles.hexInputFieldError}` : styles.hexInputField}
          maxLength={7}
          spellCheck={false}
          autoComplete="off"
          aria-label="Hex color value"
          aria-invalid={hexError ? 'true' : 'false'}
          aria-describedby={hexError ? `${HEX_HELPER_ID} ${HEX_ERROR_ID}` : HEX_HELPER_ID}
          placeholder="#6366F1"
        />
        <button
          type="button"
          onClick={handleRandom}
          className={styles.randomBtn}
          aria-label="Pick a random color"
          title="Random color"
        >
          {/* Dice / shuffle icon */}
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="16 3 21 3 21 8" />
            <line x1="4" y1="20" x2="21" y2="3" />
            <polyline points="21 16 21 21 16 21" />
            <line x1="15" y1="15" x2="21" y2="21" />
            <line x1="4" y1="4" x2="9" y2="9" />
          </svg>
        </button>
      </div>

      <p className={styles.helper} id={HEX_HELPER_ID}>
        Pick any color, then tell the model if you like it.
      </p>
      {hexError && (
        <p className={styles.errorText} id={HEX_ERROR_ID} role="status" aria-live="polite">
          {hexError}
        </p>
      )}
    </div>
  );
};

export default ColorPicker;

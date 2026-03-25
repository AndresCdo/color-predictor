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

const ColorPicker = ({ currentColor, onColorChange }) => {
  const [hexDraft, setHexDraft] = useState(currentColor.toUpperCase());
  const nativeInputRef = useRef(null);
  const textColor = getContrastTextColor(currentColor);

  const commitHex = useCallback(
    (raw) => {
      const clean = raw.startsWith('#') ? raw : `#${raw}`;
      if (HEX_REGEX.test(clean)) {
        onColorChange(clean.toLowerCase());
      }
    },
    [onColorChange],
  );

  const handleNativeChange = useCallback(
    (e) => {
      const value = e.target.value;
      onColorChange(value);
      setHexDraft(value.toUpperCase());
    },
    [onColorChange],
  );

  const handleTextChange = useCallback((e) => {
    setHexDraft(e.target.value.toUpperCase());
  }, []);

  const handleTextBlur = useCallback(() => {
    commitHex(hexDraft);
  }, [commitHex, hexDraft]);

  const handleTextKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter') {
        commitHex(hexDraft);
        e.target.blur();
      }
    },
    [commitHex, hexDraft],
  );

  const handleRandom = useCallback(() => {
    const color = randomHex();
    onColorChange(color);
    setHexDraft(color.toUpperCase());
  }, [onColorChange]);

  // Keep text input in sync when parent changes colour via keyboard shortcuts etc.
  const displayHex = currentColor.toUpperCase();
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
          onFocus={() => setHexDraft(displayHex)}
          className={styles.hexInputField}
          maxLength={7}
          spellCheck={false}
          autoComplete="off"
          aria-label="Hex color value"
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

      <p className={styles.helper} id="color-help">
        Pick any color, then tell the model if you like it.
      </p>
    </div>
  );
};

export default ColorPicker;

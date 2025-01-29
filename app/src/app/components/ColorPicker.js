import React from 'react';
import PropTypes from 'prop-types';
import styles from './ColorPicker.module.css';

const ColorPicker = ({ currentColor, onColorChange }) => {
  return (
    <div className={styles.colorPickerContainer}>
      <input 
        type="color" 
        value={currentColor}
        onChange={(e) => onColorChange(e.target.value)}
        className={styles.colorInput}
        aria-label="Select color"
      />
      <div
        className={styles.colorPreview}
        style={{ backgroundColor: currentColor }}
      >
        Current Color: {currentColor}
      </div>
    </div>
  );
};

ColorPicker.propTypes = {
  currentColor: PropTypes.string.isRequired,
  onColorChange: PropTypes.func.isRequired,
};

export default ColorPicker;

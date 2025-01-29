import React from 'react';
import PropTypes from 'prop-types';
import styles from './Prediction.module.css';

const Prediction = ({ prediction }) => {
  if (prediction === null) {
    return null;
  }

  const predictionText = prediction > 0.5 ? 'Like' : 'Dislike';
  const predictionPercentage = (prediction * 100).toFixed(2);

  return (
    <div className={styles.predictionContainer}>
      <h2 className={styles.predictionTitle}>Prediction Result</h2>
      <p className={styles.predictionText}>
        The model predicts: <strong>{predictionText}</strong>
      </p>
      <p className={styles.predictionConfidence}>
        Confidence: <strong>{predictionPercentage}%</strong>
      </p>
      <div className={styles.predictionBar}>
        <div 
          className={styles.predictionFill} 
          style={{ width: `${predictionPercentage}%` }}
        />
      </div>
    </div>
  );
};

Prediction.propTypes = {
  prediction: PropTypes.number,
};

export default Prediction;

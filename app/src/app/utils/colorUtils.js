import * as tf from './tensorflow.js';

/**
 * Validates and converts a hex color string to RGB values normalized between 0 and 1.
 * @param {string} hex - The hex color string (with or without # prefix).
 * @returns {number[]|null} An array of normalized RGB values, or null if invalid.
 */
export const hexToRgb = (hex) => {
  if (!hex || typeof hex !== 'string') return null;
  
  // Support both #RRGGBB and RRGGBB formats
  const cleaned = hex.charAt(0) === '#' ? hex.substring(1) : hex;
  
  // Support both 3-digit and 6-digit hex
  const expanded = cleaned.length === 3 
    ? cleaned.split('').map(char => char + char).join('')
    : cleaned;
    
  const result = /^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(expanded);
  if (!result) return null;
  
  try {
    return [
      parseInt(result[1], 16) / 255,
      parseInt(result[2], 16) / 255,
      parseInt(result[3], 16) / 255
    ];
  } catch {
    return null;
  }
};

/**
 * Creates and compiles an improved TensorFlow.js model for color prediction.
 * @param {Object} config - Model configuration options.
 * @param {number} [config.learningRate=0.001] - Learning rate for the optimizer.
 * @param {number} [config.dropout=0.2] - Dropout rate for regularization.
 * @returns {tf.Sequential} The compiled TensorFlow.js model.
 */
export const createModel = ({ learningRate = 0.001, dropout = 0.2 } = {}) => {
  const model = tf.sequential();
  
  // Input layer with batch normalization
  model.add(tf.layers.dense({
    units: 32,
    inputShape: [3],
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: dropout }));
  
  // Hidden layer
  model.add(tf.layers.dense({
    units: 16,
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: dropout }));
  
  // Output layer
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    kernelInitializer: 'glorotNormal'
  }));
  
  const optimizer = tf.train.adam(learningRate);
  model.compile({
    loss: 'binaryCrossentropy',
    optimizer,
    metrics: ['accuracy']
  });
  
  return model;
};

/** Releases model weights and the explicitly-created optimizer's tensors. */
export const disposeModel = (model) => {
  model?.optimizer?.dispose();
  model?.dispose();
};

/**
 * Preprocesses and validates color data before training.
 * @param {string[]} selectedColors - Array of liked color hex strings.
 * @param {string[]} unselectedColors - Array of disliked color hex strings.
 * @param {number} validationSplit - Fraction held out independently from each class.
 * @returns {Object} Training tensors and optional validation tensors.
 * @throws {Error} If input data is invalid.
 */
const preprocessData = (selectedColors, unselectedColors, validationSplit) => {
  if (!Number.isFinite(validationSplit) || validationSplit < 0 || validationSplit >= 1) {
    throw new Error('validationSplit must be between 0 (inclusive) and 1 (exclusive)');
  }
  if (!Array.isArray(selectedColors) || !Array.isArray(unselectedColors)) {
    throw new Error('Color arrays must be provided');
  }
  
  if (selectedColors.length === 0 || unselectedColors.length === 0) {
    throw new Error('Both selected and unselected colors must contain data');
  }
  
  const processedSelected = selectedColors
    .map(hexToRgb)
    .filter(rgb => rgb !== null);
  const processedUnselected = unselectedColors
    .map(hexToRgb)
    .filter(rgb => rgb !== null);
    
  if (processedSelected.length === 0 || processedUnselected.length === 0) {
    throw new Error('No valid colors found after processing');
  }
  
  // Split each class before fitting. fit's validationSplit takes the tail
  // before shuffling, which would otherwise validate only disliked colors.
  // At the default 20%, fewer than five examples per class use training only.
  const useValidation = [processedSelected, processedUnselected]
    .every((items) => Math.floor(items.length * validationSplit) >= 1);
  const training = [];
  const validation = [];
  [processedSelected, processedUnselected].forEach((items, index) => {
    tf.util.shuffle(items);
    const count = useValidation ? Math.min(items.length - 1, Math.floor(items.length * validationSplit)) : 0;
    items.forEach((rgb, i) => {
      (i < count ? validation : training).push({ rgb, label: index === 0 ? 1 : 0 });
    });
  });

  return tf.tidy(() => ({
    xs: tf.tensor2d(training.map((item) => item.rgb)),
    ys: tf.tensor2d(training.map((item) => [item.label])),
    validationXs: useValidation ? tf.tensor2d(validation.map((item) => item.rgb)) : null,
    validationYs: useValidation ? tf.tensor2d(validation.map((item) => [item.label])) : null,
  }));
};

/**
 * Trains the model with the provided color data and advanced configuration options.
 * @param {tf.Sequential} model - The TensorFlow.js model to train.
 * @param {string[]} selectedColors - Array of liked color hex strings.
 * @param {string[]} unselectedColors - Array of disliked color hex strings.
 * @param {Object} config - Training configuration options.
 * @param {number} [config.epochs=50] - Number of training epochs.
 * @param {number} [config.batchSize=32] - Batch size for training.
 * @param {number} [config.validationSplit=0.2] - Per-class holdout fraction, used only when both classes can contribute a sample.
 * @param {(epoch: number, logs?: tf.Logs) => void} [config.onEpochEnd] - Optional callback fired after each epoch.
 * @returns {Promise<tf.History>} Training history with trainingSamples and validationSamples counts.
 * @throws {Error} If training fails or input data is invalid.
 */
export const trainModel = async (
  model,
  selectedColors,
  unselectedColors,
  { epochs = 50, batchSize = 32, validationSplit = 0.2, onEpochEnd } = {}
) => {
  let tensors;
  
  try {
    tensors = preprocessData(selectedColors, unselectedColors, validationSplit);
    
    // Early stopping callback
    const earlyStoppingCallback = tf.callbacks.earlyStopping({
      monitor: tensors.validationXs ? 'val_loss' : 'loss',
      minDelta: 0.001,
      patience: 5,
      mode: 'min'
    });

    const callbacks = [earlyStoppingCallback];
    if (typeof onEpochEnd === 'function') {
      callbacks.push(new tf.CustomCallback({ onEpochEnd }));
    }
    
    const history = await model.fit(tensors.xs, tensors.ys, {
      epochs,
      batchSize,
      validationData: tensors.validationXs ? [tensors.validationXs, tensors.validationYs] : undefined,
      shuffle: true,
      callbacks,
      verbose: 1
    });
    
    history.trainingSamples = tensors.xs.shape[0];
    history.validationSamples = tensors.validationXs?.shape[0] ?? 0;
    return history;
  } catch (error) {
    throw new Error(`Training failed: ${error.message}`);
  } finally {
    // Clean up tensors
    if (tensors) {
      tensors.xs.dispose();
      tensors.ys.dispose();
      tensors.validationXs?.dispose();
      tensors.validationYs?.dispose();
    }
  }
};

/**
 * Predicts a color preference with uncalibrated signal strength in the legacy confidence field.
 * @param {tf.Sequential} model - The trained TensorFlow.js model.
 * @param {string} color - The hex color string to predict.
 * @returns {Object} The prediction result with confidence score.
 * @throws {Error} If prediction fails or input is invalid.
 */
export const predictColor = (model, color) => {
  const rgb = hexToRgb(color);
  if (!rgb) {
    throw new Error('Invalid color format');
  }
  
  const input = tf.tensor2d([rgb]);
  let prediction;
  
  try {
    prediction = model.predict(input);
    const score = prediction.dataSync()[0];
    
    // Distance from the decision boundary, not calibrated probability of correctness.
    const confidence = Math.abs(score - 0.5) * 2; // Scale to 0-1
    
    return {
      score,
      confidence,
      likely: score > 0.5,
      prediction: score > 0.5 ? 'like' : 'dislike'
    };
  } finally {
    if (prediction) {
      prediction.dispose();
    }
    input.dispose();
  }
};

/**
 * Saves the trained model to localStorage or IndexedDB.
 * @param {tf.Sequential} model - The trained model to save.
 * @param {string} modelId - Unique identifier for the model.
 * @returns {Promise<void>}
 */
export const saveModel = async (model, modelId) => {
  try {
    await model.save(`indexeddb://${modelId}`);
  } catch (error) {
    throw new Error(`Failed to save model: ${error.message}`);
  }
};

/**
 * Loads a previously saved model.
 * @param {string} modelId - Unique identifier for the model.
 * @returns {Promise<tf.Sequential>} The loaded model.
 */
export const loadModel = async (modelId) => {
  try {
    return await tf.loadLayersModel(`indexeddb://${modelId}`);
  } catch (error) {
    throw new Error(`Failed to load model: ${error.message}`);
  }
};

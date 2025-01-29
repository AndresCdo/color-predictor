import * as tf from '@tensorflow/tfjs';

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
  
  // Hidden layer with residual connection
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
    metrics: ['accuracy', 'precision']
  });
  
  return model;
};

/**
 * Preprocesses and validates color data before training.
 * @param {string[]} selectedColors - Array of liked color hex strings.
 * @param {string[]} unselectedColors - Array of disliked color hex strings.
 * @returns {Object} Processed tensors and metadata.
 * @throws {Error} If input data is invalid.
 */
const preprocessData = (selectedColors, unselectedColors) => {
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
  
  return {
    xs: tf.tensor2d([...processedSelected, ...processedUnselected]),
    ys: tf.tensor2d(
      [
        ...Array(processedSelected.length).fill(1),
        ...Array(processedUnselected.length).fill(0)
      ],
      [processedSelected.length + processedUnselected.length, 1]
    ),
    totalSamples: processedSelected.length + processedUnselected.length
  };
};

/**
 * Trains the model with the provided color data and advanced configuration options.
 * @param {tf.Sequential} model - The TensorFlow.js model to train.
 * @param {string[]} selectedColors - Array of liked color hex strings.
 * @param {string[]} unselectedColors - Array of disliked color hex strings.
 * @param {Object} config - Training configuration options.
 * @param {number} [config.epochs=50] - Number of training epochs.
 * @param {number} [config.batchSize=32] - Batch size for training.
 * @param {number} [config.validationSplit=0.2] - Fraction of data to use for validation.
 * @returns {Promise<tf.History>} A promise that resolves with the training history.
 * @throws {Error} If training fails or input data is invalid.
 */
export const trainModel = async (
  model,
  selectedColors,
  unselectedColors,
  { epochs = 50, batchSize = 32, validationSplit = 0.2 } = {}
) => {
  let tensors;
  
  try {
    tensors = preprocessData(selectedColors, unselectedColors);
    
    // Early stopping callback
    const earlyStoppingCallback = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      minDelta: 0.001,
      patience: 5,
      mode: 'min'
    });
    
    const history = await model.fit(tensors.xs, tensors.ys, {
      epochs,
      batchSize,
      validationSplit,
      shuffle: true,
      callbacks: [earlyStoppingCallback],
      verbose: 1
    });
    
    return history;
  } catch (error) {
    throw new Error(`Training failed: ${error.message}`);
  } finally {
    // Clean up tensors
    if (tensors) {
      tensors.xs.dispose();
      tensors.ys.dispose();
    }
  }
};

/**
 * Predicts the likelihood of liking a color using the trained model with confidence score.
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
  
  try {
    const prediction = model.predict(input);
    const score = prediction.dataSync()[0];
    
    // Calculate confidence based on distance from decision boundary
    const confidence = Math.abs(score - 0.5) * 2; // Scale to 0-1
    
    return {
      score,
      confidence,
      likely: score > 0.5,
      prediction: score > 0.5 ? 'like' : 'dislike'
    };
  } finally {
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
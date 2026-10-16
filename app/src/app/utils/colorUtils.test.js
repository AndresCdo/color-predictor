import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from './tensorflow.js';
import { trainModel, hexToRgb, createModel, disposeModel } from './colorUtils.js';

const liked = ['#ffffff', '#eeeeee', '#dddddd', '#cccccc', '#bbbbbb'];
const disliked = ['#000000', '#111111', '#222222', '#333333', '#444444'];

test('validation contains both classes and no training samples', async () => {
  const model = { fit: async (xs, ys, options) => {
    assert.equal(options.validationSplit, undefined);
    const [vx, vy] = options.validationData;
    assert.deepEqual(new Set(vy.dataSync()), new Set([0, 1]));
    assert.deepEqual(new Set(ys.dataSync()), new Set([0, 1]));
    assert.equal(xs.shape[0], 8);
    assert.equal(vx.shape[0], 2);
    const training = new Set(xs.arraySync().map(JSON.stringify));
    for (const rgb of vx.arraySync()) assert.equal(training.has(JSON.stringify(rgb)), false);
    return { history: {} };
  } };
  const history = await trainModel(model, liked, disliked);
  assert.equal(history.validationSamples, 2);
  assert.equal(history.trainingSamples, 8);
});

test('four ratings train without claiming a validation estimate', async () => {
  const history = await trainModel({ fit: async (xs, ys, options) => {
    assert.equal(xs.shape[0], 4);
    assert.equal(options.validationData, undefined);
    assert.equal(options.validationSplit, undefined);
    return { history: {} };
  } }, liked.slice(0, 2), disliked.slice(0, 2));
  assert.equal(history.validationSamples, 0);
});

test('progress callback preserves early stopping on constant loss', async () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [3], activation: 'sigmoid', kernelInitializer: 'zeros' }));
  model.compile({ optimizer: tf.train.sgd(0), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  let updates = 0;
  try {
    const history = await trainModel(model, liked, disliked, { epochs: 50, onEpochEnd: () => { updates++; } });
    assert.equal(history.epoch.length, 6);
    assert.equal(updates, 6);
    assert.equal(history.history.acc.length, 6);
  } finally {
    model.dispose();
  }
});

test('training releases input and validation tensors when fit rejects', async () => {
  const before = tf.memory().numTensors;
  await assert.rejects(trainModel({ fit: async () => { throw new Error('fit failed'); } }, liked, disliked), /fit failed/);
  assert.equal(tf.memory().numTensors, before);
});

test('invalid inputs reject before fitting', async () => {
  const model = { fit: () => assert.fail('fit must not run') };
  await assert.rejects(trainModel(model, [], disliked), /contain data/);
  await assert.rejects(trainModel(model, ['invalid'], disliked), /No valid colors/);
  await assert.rejects(trainModel(model, liked, disliked, { validationSplit: 1 }), /validationSplit/);
});

test('hex input accepts supported formats and rejects invalid colors', () => {
  assert.deepEqual(hexToRgb('#fff'), [1, 1, 1]);
  assert.deepEqual(hexToRgb('000000'), [0, 0, 0]);
  assert.equal(hexToRgb('#gggggg'), null);
  assert.equal(hexToRgb(null), null);
});

test('replacing trained models releases optimizer state as well as weights', async () => {
  const before = tf.memory().numTensors;
  const model = createModel();
  await trainModel(model, liked, disliked, { epochs: 1 });
  disposeModel(model);
  assert.equal(tf.memory().numTensors, before);
});

test('loaded models can be disposed without leaking or double-disposal errors', async () => {
  const before = tf.memory().numTensors;
  const original = createModel();
  let artifacts;
  await original.save(tf.io.withSaveHandler(async (value) => {
    artifacts = value;
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));
  disposeModel(original);
  const loaded = await tf.loadLayersModel(tf.io.fromMemory(artifacts));
  disposeModel(loaded);
  assert.equal(tf.memory().numTensors, before);
});

let colorChoosen = [];
let colorNotChoosen = [];

async function learnLinear() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShare: [1]}))
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  const xs = tf.tensor2d(colorsChoosen,[colorsChoosen.length, 1]);
  const ys = tf.tensor2d(colorsNotChoosen, [colorsNotChoosen.length, 1]);

  await model.fit(xs, ys, {epochs: 5});
}

# Color Predictor

Teach a small neural network your color taste — entirely in your browser. No data
ever leaves your device.

**[▶ Try the live demo](https://andrescdo.github.io/color-predictor/)**

[![Deploy to GitHub Pages](https://github.com/AndresCdo/color-predictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/AndresCdo/color-predictor/actions/workflows/deploy.yml)

## How it works

1. **Rate colors.** Pick a color and mark it as liked or disliked. Each rating
   becomes a labeled training example.
2. **Train.** Once you have at least 2 liked and 2 disliked colors, training
   runs in the browser with TensorFlow.js.
3. **Predict.** The model classifies any new color as _like_ or _dislike_ and
   reports its signal strength. Predictions are experimental with few ratings.

Everything — training, inference, and storage — happens client-side. There is no
backend and no telemetry.

## The model

A small binary classifier defined in [`app/src/app/utils/colorUtils.js`](app/src/app/utils/colorUtils.js):

| Stage            | Configuration                                                             |
| ---------------- | ------------------------------------------------------------------------- |
| Input            | 3 features — RGB channels normalized to `[0, 1]`                          |
| Hidden 1         | Dense 32 (Glorot normal) → BatchNorm → ReLU → Dropout 0.2                 |
| Hidden 2         | Dense 16 (Glorot normal) → BatchNorm → ReLU → Dropout 0.2                 |
| Output           | Dense 1, sigmoid                                                          |
| Loss / optimizer | Binary cross-entropy / Adam (lr `0.001`)                                  |
| Training         | Up to 50 epochs, batch size 32, fresh model on each run                   |
| Validation       | Random 20% holdout per class, enabled with at least 5 colors per class     |
| Regularization   | Dropout plus early stopping (min delta `0.001`, patience 5)               |

Early stopping monitors `val_loss` when both classes have validation samples,
otherwise training `loss`. Training accuracy and held-out validation accuracy
are shown separately; neither a tiny holdout nor training accuracy establishes
reliable performance on new colors.

Signal strength is the distance to the decision boundary: `|score − 0.5| × 2`.
It is not calibrated confidence or the probability of a correct prediction.

These choices follow [TensorFlow.js callback and validation semantics](https://js.tensorflow.org/api/latest/#tf.LayersModel.fit),
[scikit-learn's guidance on stratification and small datasets](https://scikit-learn.org/stable/modules/cross_validation.html),
and [Guo et al. on neural-network calibration](https://proceedings.mlr.press/v70/guo17a.html).

## Persistence

- The trained model is saved to **IndexedDB** (`indexeddb://color-predictor-v1`)
  and reloaded automatically on the next visit.
- If saving fails, the trained model remains usable in the current tab. Its
  unsaved training statistics are not restored as if they belonged to a saved model.
- Ratings, rating history, theme preference, and training stats are kept in
  **localStorage**.
- _Clear session_ removes the ratings and stats; the saved model stays on the
  device until browser data is cleared.

## Features

- Color picker with a hex input, validation, and a random-color button.
- Keyboard shortcuts: <kbd>L</kbd> like, <kbd>D</kbd> dislike, <kbd>U</kbd> undo,
  <kbd>T</kbd> train, <kbd>P</kbd> predict.
- Live training progress with per-epoch accuracy.
- Light, dark, and system theme modes.
- Accessibility: ARIA live regions for status changes, a focus-trapped
  confirmation dialog, labelled controls, and swatch text contrast computed from
  relative luminance.

The layout puts rating controls first and keeps explanations in disclosures,
following [NN/g's progressive-disclosure guidance](https://www.nngroup.com/articles/progressive-disclosure/).
Text contrast follows [WCAG 2.2 AA](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

## Tech stack

- [Next.js 15](https://nextjs.org/) (App Router, static export)
- [React 19](https://react.dev/)
- [TensorFlow.js 4](https://www.tensorflow.org/js)
- CSS Modules

## Running locally

Requires Node.js 22 or newer.

```bash
git clone https://github.com/AndresCdo/color-predictor.git
cd color-predictor/app
npm ci
npm run dev
```

The dev server runs at <http://localhost:3000/color-predictor>.

Available scripts, all run from `app/`:

| Command         | Purpose                           |
| --------------- | --------------------------------- |
| `npm run dev`   | Development server with Turbopack |
| `npm run build` | Static export to `app/out/`       |
| `npm run lint`  | ESLint via `eslint-config-next`   |
| `npm test`      | Training regression tests using Node's test runner |

## Dependency security

Run `npm audit --omit=dev` from `app/`; CI requires this check to pass.
TensorFlow uses its existing core, layers, CPU, and WebGL modules directly,
excluding the unused umbrella CLI dependency that pulled in vulnerable `sprintf-js`.

The full development audit still reports
[GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
through the Next.js ESLint toolchain (`fast-glob` → `micromatch` → `braces`).
As of 2026-10-16, upstream lists no patched release. These packages process
repository-controlled lint patterns and are not shipped in the static site.
Do not pass untrusted patterns to this tooling or use `npm audit fix --force`
to downgrade the framework. Keep reviewing the full audit for an upstream fix;
the production audit does not imply that development dependencies are clean.

## Deployment

Pushing to `master` triggers
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the
static export and publishes `app/out/` to GitHub Pages. Pull requests run lint,
regression tests, the static build, and a container smoke test without publishing.

Because Pages serves the site from a subpath, `app/next.config.mjs` sets
`basePath: '/color-predictor'` alongside `output: 'export'`.

For a local production preview, build and run the static nginx container:

```bash
docker build -t color-predictor .
docker run --rm -p 3000:3000 color-predictor
```

Open <http://localhost:3000/color-predictor/>. `next start` is not compatible
with this static-export project. Unknown paths return 404.

## Project structure

```
app/
├── next.config.mjs              # Static export + basePath for GitHub Pages
└── src/app/
    ├── page.js                  # Main UI, training and prediction flow
    ├── layout.js
    ├── components/ColorPicker.js
    └── utils/colorUtils.js      # Model definition, training, prediction, persistence
.github/workflows/deploy.yml     # Build and deploy to GitHub Pages
Dockerfile                       # Container build
conf/                            # nginx configuration
```

## Contributing

Issues and pull requests are welcome. Please keep changes focused and run
`npm test`, `npm run lint`, and `npm run build` from `app/` before opening a pull request.

## License

Released under the [MIT License](LICENSE).

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
   reports its confidence.

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
| Training         | 50 epochs, batch size 32, 20% validation split, shuffled                  |
| Regularization   | Dropout plus early stopping on `val_loss` (min delta `0.001`, patience 5) |

Confidence is derived from the distance to the decision boundary:
`|score − 0.5| × 2`.

## Persistence

- The trained model is saved to **IndexedDB** (`indexeddb://color-predictor-v1`)
  and reloaded automatically on the next visit.
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

## Tech stack

- [Next.js 15](https://nextjs.org/) (App Router, static export)
- [React 19](https://react.dev/)
- [TensorFlow.js 4](https://www.tensorflow.org/js)
- CSS Modules

## Running locally

Requires Node.js 20 or newer.

```bash
git clone https://github.com/AndresCdo/color-predictor.git
cd color-predictor/app
npm install
npm run dev
```

The dev server runs at <http://localhost:3000/color-predictor>.

Available scripts, all run from `app/`:

| Command         | Purpose                           |
| --------------- | --------------------------------- |
| `npm run dev`   | Development server with Turbopack |
| `npm run build` | Static export to `app/out/`       |
| `npm run lint`  | ESLint via `eslint-config-next`   |

## Deployment

Pushing to `master` triggers
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the
static export and publishes `app/out/` to GitHub Pages.

Because Pages serves the site from a subpath, `app/next.config.mjs` sets
`basePath: '/color-predictor'` alongside `output: 'export'`.

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
`npm run lint` from `app/` before opening a pull request.

## License

Released under the [MIT License](LICENSE).

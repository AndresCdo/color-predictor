# Project decisions

- Keep the application client-only with a static export at `/color-predictor`.
  Preserve the existing IndexedDB model key and rating storage format.
- Train a fresh model on each run so validation colors remain unseen. Split
  each class independently; at the default 20%, validation starts at five
  valid colors per class. Smaller datasets use training loss for early stopping.
- Label training and validation metrics separately. The distance from a 0.5
  score is signal strength, not calibrated confidence or probability of success.
- Keep the color sample and both rating controls visible on the initial
  390×844 viewport. Use neutral surfaces and WCAG AA text contrast.
- Verify changes with `npm test`, `npm run lint`, `npm run build`, and browser
  checks of rating, undo, training, prediction, reload, reset, and save failure.
  The container serves the static export with nginx; it does not run Next.js.
- Use the aligned TensorFlow core, layers, CPU, and WebGL runtime packages.
  Do not restore the umbrella CLI dependency; CI audits production dependencies.

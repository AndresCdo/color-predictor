// Use the existing runtime modules without the umbrella package's unused CLI.
// Layers registers training gradients; both backends preserve browser fallback.
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

export * from '@tensorflow/tfjs-core';
export * from '@tensorflow/tfjs-layers';

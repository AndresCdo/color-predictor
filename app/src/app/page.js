'use client';
import { useState, useEffect, useCallback, useMemo } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Alert, AlertTitle, Button, CircularProgress, Container, Grid, Paper, Typography } from '@mui/material';
import { ErrorOutline, ThumbUp, ThumbDown, Psychology, AutoFixHigh } from '@mui/icons-material';
import {
  hexToRgb,
  createModel,
  trainModel,
  predictColor,
  saveModel,
  loadModel
} from './utils/colorUtils';
import ColorPicker from './components/ColorPicker';

const MODEL_ID = 'color-predictor-v1';

export default function ColorPredictor() {
  // State management
  const [status, setStatus] = useState({
    isLoading: true,
    isTraining: false,
    error: null
  });
  const [model, setModel] = useState(null);
  const [colors, setColors] = useState({
    selected: [],
    unselected: [],
    current: '#000000'
  });
  const [prediction, setPrediction] = useState(null);
  const [modelStats, setModelStats] = useState({
    trainedSamples: 0,
    accuracy: null,
    lastTrained: null
  });

  // Load or create model on mount
  useEffect(() => {
    const initModel = async () => {
      try {
        setStatus(prev => ({ ...prev, isLoading: true }));
        try {
          const loadedModel = await loadModel(MODEL_ID);
          setModel(loadedModel);
          setStatus(prev => ({ ...prev, isLoading: false }));
        } catch {
          const newModel = createModel({ learningRate: 0.001, dropout: 0.2 });
          setModel(newModel);
          setStatus(prev => ({ ...prev, isLoading: false }));
        }
      } catch (error) {
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          error: 'Failed to initialize model'
        }));
      }
    };
    initModel();
  }, []);

  // Memoized stats
  const stats = useMemo(() => ({
    totalSamples: colors.selected.length + colors.unselected.length,
    likedPercentage: colors.selected.length > 0
      ? (colors.selected.length / (colors.selected.length + colors.unselected.length) * 100).toFixed(1)
      : 0,
    canTrain: colors.selected.length >= 2 && colors.unselected.length >= 2
  }), [colors.selected, colors.unselected]);

  // Handlers
  const handleColorSelect = useCallback((liked) => {
    setColors(prev => ({
      ...prev,
      selected: liked ? [...prev.selected, prev.current] : prev.selected,
      unselected: liked ? prev.unselected : [...prev.unselected, prev.current]
    }));
  }, []);

  const handleTrainModel = useCallback(async () => {
    if (status.isTraining || !model || !stats.canTrain) return;
  
    setStatus(prev => ({ ...prev, isTraining: true, error: null }));
    try {
      // Verify model compilation
      if (!model.compiled) {
        model.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'binaryCrossentropy',
          metrics: ['accuracy']
        });
      }
      
      const history = await trainModel(
        model,
        colors.selected,
        colors.unselected,
        { epochs: 50, batchSize: 32 }
      );
  
      await saveModel(model, MODEL_ID);
  
      // Safely access history values with fallback
      const accuracy = history?.history?.accuracy?.[0] ?? 0;
      
      setModelStats({
        trainedSamples: stats.totalSamples,
        accuracy: (accuracy * 100).toFixed(1),
        lastTrained: new Date().toLocaleString()
      });
  
      setPrediction(null);
    } catch (error) {
      console.error('Training error:', error);
      setStatus(prev => ({
        ...prev,
        error: `Training failed: ${error.message || 'Unknown error'}`
      }));
    } finally {
      setStatus(prev => ({ ...prev, isTraining: false }));
    }
  }, [model, colors.selected, colors.unselected, stats.canTrain, stats.totalSamples]);  

  const handlePredict = useCallback(async () => {
    if (!model) return;

    try {
      const result = await predictColor(model, colors.current);
      setPrediction(result);
    } catch (error) {
      setStatus(prev => ({
        ...prev,
        error: 'Prediction failed: ' + error.message
      }));
    }
  }, [model, colors.current]);

  if (status.isLoading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading Color Predictor...</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 8 }}>
      <Typography variant="h3" component="h1" align="center" gutterBottom>Color Predictor</Typography>

      {status.error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>Error</AlertTitle>
          {status.error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>Select Colors</Typography>
            <ColorPicker
              currentColor={colors.current}
              onColorChange={(color) => setColors(prev => ({ ...prev, current: color }))}
            />
            
            <Grid container justifyContent="center" spacing={2} sx={{ mt: 3 }}>
              <Grid item>
                <Button
                  variant="contained"
                  startIcon={<ThumbUp />}
                  onClick={() => handleColorSelect(true)}
                >
                  Like
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="contained"
                  color="error"
                  startIcon={<ThumbDown />}
                  onClick={() => handleColorSelect(false)}
                >
                  Dislike
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>Model Training</Typography>
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <Typography variant="body2">Liked Colors: {colors.selected.length}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">Disliked Colors: {colors.unselected.length}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">Total Samples: {stats.totalSamples}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">Like Ratio: {stats.likedPercentage}%</Typography>
              </Grid>
            </Grid>

            {modelStats.lastTrained && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Last trained: {modelStats.lastTrained} 
                (Accuracy: {modelStats.accuracy}%)
              </Typography>
            )}

            <Grid container justifyContent="center" spacing={2}>
              <Grid item>
                <Button
                  variant="contained"
                  startIcon={<Psychology />}
                  onClick={handleTrainModel}
                  disabled={!stats.canTrain || status.isTraining}
                >
                  {status.isTraining ? 'Training...' : 'Train Model'}
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  startIcon={<AutoFixHigh />}
                  onClick={handlePredict}
                  disabled={!modelStats.lastTrained}
                >
                  Predict
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {prediction && (
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>Prediction Results</Typography>
              <Grid container alignItems="center" spacing={2}>
                <Grid item>
                  <div style={{ width: 48, height: 48, backgroundColor: colors.current, border: '1px solid #000', borderRadius: 4 }} />
                </Grid>
                <Grid item xs>
                  <Typography variant="body1">
                    {prediction.prediction === 'like' ? 'You might like this color!' : 'You might not like this color'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  );
}

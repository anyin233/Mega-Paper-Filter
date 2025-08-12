import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  Button,
  TextField,
  MenuItem,
  FormControl,
  FormLabel,
  Slider,
  Grid,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { 
  PlayArrow as PlayIcon, 
  Stop as StopIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { api, Dataset, ClusteringConfig } from '../services/api';
import { useJobStatus, useWebSocket } from '../hooks/useWebSocket';

interface ClusteringInterfaceProps {
  onClusteringComplete?: (jobId: string) => void;
  onClusteringError?: (error: string) => void;
}

const ClusteringInterface: React.FC<ClusteringInterfaceProps> = ({ 
  onClusteringComplete, 
  onClusteringError 
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [config, setConfig] = useState<ClusteringConfig>({
    dataset_name: '',
    max_features: 1000,
    max_k: 15,
    min_papers: 5,
  });
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [resultsDialogOpen, setResultsDialogOpen] = useState(false);
  const [clusteringResults, setClusteringResults] = useState<any>(null);

  const { status: jobStatus, startPolling, stopPolling } = useJobStatus();

  // WebSocket for real-time updates  
  const { isConnected } = useWebSocket(
    useCallback((data: any) => {
      if (data.type === 'job_update' && data.data.job_id === currentJobId) {
        console.log('Received clustering job update:', data.data);
        // Job status will be updated via the polling hook
      }
    }, [currentJobId]),
    useCallback((error: Event) => {
      console.error('WebSocket error during clustering:', error);
    }, [])
  );

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    // Stop polling and clear job when job is completed or failed
    if (jobStatus && (jobStatus.status === 'completed' || jobStatus.status === 'failed')) {
      if (jobStatus.status === 'completed' && jobStatus.result) {
        setClusteringResults(jobStatus.result);
        // Notify parent component of successful completion
        if (currentJobId && onClusteringComplete) {
          onClusteringComplete(currentJobId);
        }
      } else if (jobStatus.status === 'failed' && jobStatus.error) {
        // Notify parent component of error
        if (onClusteringError) {
          onClusteringError(jobStatus.error);
        }
      }
    }
  }, [jobStatus, currentJobId, onClusteringComplete, onClusteringError]);

  const loadDatasets = async () => {
    try {
      const datasetsData = await api.getDatasets();
      setDatasets(datasetsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load datasets');
    }
  };

  const handleConfigChange = (field: keyof ClusteringConfig) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig(prev => ({
      ...prev,
      [field]: event.target.type === 'number' ? Number(event.target.value) : event.target.value,
    }));
  };

  const handleSliderChange = (field: keyof ClusteringConfig) => (
    _event: Event,
    value: number | number[]
  ) => {
    setConfig(prev => ({
      ...prev,
      [field]: Array.isArray(value) ? value[0] : value,
    }));
  };

  const startClustering = async () => {
    try {
      setError(null);
      const response = await api.runClustering(config);
      setCurrentJobId(response.job_id);
      startPolling(response.job_id);
    } catch (err: any) {
      setError(err.message || 'Failed to start clustering');
    }
  };

  const stopClustering = () => {
    stopPolling();
    setCurrentJobId(null);
  };

  const viewResults = async () => {
    if (!currentJobId || !jobStatus?.result) return;
    
    try {
      const results = await api.getClusteringResults(currentJobId);
      setClusteringResults(results);
      setResultsDialogOpen(true);
    } catch (err: any) {
      setError(err.message || 'Failed to load clustering results');
    }
  };

  const isRunning = jobStatus?.status === 'running' || jobStatus?.status === 'pending';
  const isCompleted = jobStatus?.status === 'completed';
  const hasFailed = jobStatus?.status === 'failed';

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Clustering Analysis
      </Typography>

      {/* WebSocket Status */}
      {currentJobId && (
        <Alert 
          severity={isConnected ? 'success' : 'warning'} 
          sx={{ mb: 3 }}
        >
          Real-time updates: {isConnected ? 'Connected' : 'Disconnected'}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={6}>
          <MuiPaper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>

            <TextField
              select
              fullWidth
              label="Dataset"
              value={config.dataset_name || ''}
              onChange={handleConfigChange('dataset_name')}
              margin="normal"
              helperText="Select a dataset to cluster, or leave empty for all papers"
            >
              <MenuItem value="">All Papers</MenuItem>
              {datasets.map((dataset) => (
                <MenuItem key={dataset.id} value={dataset.name}>
                  {dataset.name} ({dataset.total_papers} papers)
                </MenuItem>
              ))}
            </TextField>

            <FormControl fullWidth sx={{ mt: 3, mb: 2 }}>
              <FormLabel>Maximum Features: {config.max_features}</FormLabel>
              <Slider
                value={config.max_features || 1000}
                onChange={handleSliderChange('max_features')}
                min={100}
                max={5000}
                step={100}
                marks={[
                  { value: 500, label: '500' },
                  { value: 1000, label: '1000' },
                  { value: 2000, label: '2000' },
                  { value: 5000, label: '5000' },
                ]}
              />
            </FormControl>

            <FormControl fullWidth sx={{ mt: 3, mb: 2 }}>
              <FormLabel>Maximum Clusters: {config.max_k}</FormLabel>
              <Slider
                value={config.max_k || 15}
                onChange={handleSliderChange('max_k')}
                min={2}
                max={30}
                step={1}
                marks={[
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 15, label: '15' },
                  { value: 20, label: '20' },
                  { value: 30, label: '30' },
                ]}
              />
            </FormControl>

            <TextField
              fullWidth
              type="number"
              label="Minimum Papers"
              value={config.min_papers}
              onChange={handleConfigChange('min_papers')}
              margin="normal"
              inputProps={{ min: 1, max: 100 }}
              helperText="Minimum number of papers required for clustering"
            />

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={startClustering}
                disabled={isRunning}
                startIcon={<PlayIcon />}
                fullWidth
              >
                {isRunning ? 'Running...' : 'Start Clustering'}
              </Button>
              
              {isRunning && (
                <Button
                  variant="outlined"
                  onClick={stopClustering}
                  startIcon={<StopIcon />}
                >
                  Stop
                </Button>
              )}
            </Box>
          </MuiPaper>
        </Grid>

        {/* Status Panel */}
        <Grid item xs={12} md={6}>
          <MuiPaper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Status
            </Typography>

            {!currentJobId && (
              <Alert severity="info">
                Configure your clustering parameters and click "Start Clustering" to begin.
              </Alert>
            )}

            {currentJobId && jobStatus && (
              <Box>
                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Job ID: {currentJobId}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="body2" sx={{ mr: 1 }}>
                        Status:
                      </Typography>
                      <Chip 
                        label={jobStatus.status.toUpperCase()} 
                        color={
                          jobStatus.status === 'completed' ? 'success' :
                          jobStatus.status === 'failed' ? 'error' :
                          jobStatus.status === 'running' ? 'primary' :
                          'default'
                        }
                        size="small"
                      />
                    </Box>

                    {jobStatus.message && (
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {jobStatus.message}
                      </Typography>
                    )}

                    {isRunning && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body2" gutterBottom>
                          Progress: {Math.round(jobStatus.progress * 100)}%
                        </Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={jobStatus.progress * 100} 
                        />
                      </Box>
                    )}

                    {isCompleted && jobStatus.result && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Results:
                        </Typography>
                        <Typography variant="body2">
                          • Total papers: {jobStatus.result.total_papers}<br />
                          • Clusters found: {jobStatus.result.total_clusters}<br />
                          • Silhouette score: {jobStatus.result.silhouette_score?.toFixed(3)}
                        </Typography>
                        <Button
                          variant="outlined"
                          onClick={viewResults}
                          startIcon={<ViewIcon />}
                          sx={{ mt: 2 }}
                          fullWidth
                        >
                          View Results
                        </Button>
                      </Box>
                    )}

                    {hasFailed && jobStatus.error && (
                      <Alert severity="error" sx={{ mt: 2 }}>
                        {jobStatus.error}
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </MuiPaper>
        </Grid>
      </Grid>

      {/* Results Dialog */}
      <Dialog 
        open={resultsDialogOpen} 
        onClose={() => setResultsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Clustering Results
        </DialogTitle>
        <DialogContent>
          {clusteringResults && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Summary
              </Typography>
              <Typography variant="body2" paragraph>
                Successfully clustered {clusteringResults.total_papers} papers into{' '}
                {clusteringResults.total_clusters} clusters with a silhouette score of{' '}
                {clusteringResults.silhouette_score?.toFixed(3)}.
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                Results have been saved and can be viewed in the visualization interface.
                The clustering data is now available for interactive exploration.
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResultsDialogOpen(false)}>
            Close
          </Button>
          <Button variant="contained" onClick={() => {
            setResultsDialogOpen(false);
            // Navigate to visualization page
            if (currentJobId && onClusteringComplete) {
              onClusteringComplete(currentJobId);
            }
          }}>
            View Visualization
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ClusteringInterface;
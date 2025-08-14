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
  ToggleButton,
  ToggleButtonGroup,
  Collapse,
} from '@mui/material';
import { 
  PlayArrow as PlayIcon, 
  Stop as StopIcon,
  Visibility as ViewIcon,
  Psychology as AIIcon,
  BarChart as TraditionalIcon,
  Hub as EmbeddingIcon,
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
    max_k: 25,  // Increased from 15 to 25
    min_papers: 5,
    clustering_method: 'traditional',
    // Feature extraction parameters
    feature_extraction_method: 'tfidf',
    sentence_transformer_model: 'all-MiniLM-L6-v2',
    // Traditional clustering algorithm parameters
    traditional_algorithm: 'kmeans',
    dbscan_eps: 0.5,
    dbscan_min_samples: 5,
    agglomerative_linkage: 'ward',
    spectral_assign_labels: 'discretize',
    // LLM clustering parameters
    llm_model: 'gpt-4o',
    custom_model_name: '',
    max_papers_llm: 500,
    // Embedding parameters
    embedding_model: 'text-embedding-ada-002',
    embedding_batch_size: 50,
    embedding_clustering_algorithm: 'kmeans',
    embedding_dbscan_eps: 0.5,
    embedding_dbscan_min_samples: 5,
    embedding_agglomerative_linkage: 'ward',
  });
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [resultsDialogOpen, setResultsDialogOpen] = useState(false);
  const [clusteringResults, setClusteringResults] = useState<any>(null);
  const [openAIEnabled, setOpenAIEnabled] = useState(false);
  const [embeddingEnabled, setEmbeddingEnabled] = useState(false);
  const [restoredJob, setRestoredJob] = useState(false); // Track if we restored a previous job

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
    checkOpenAISettings();
    checkForActiveJobs(); // Check for any active jobs when component mounts
  }, []);

  const checkForActiveJobs = async () => {
    try {
      // First, check localStorage for any previously tracked job
      const savedJobId = localStorage.getItem('currentClusteringJobId');
      if (savedJobId) {
        try {
          const savedJobStatus = await api.getJobStatus(savedJobId);
          if (savedJobStatus.status === 'running' || savedJobStatus.status === 'pending') {
            console.log('Restoring saved clustering job:', savedJobId);
            setCurrentJobId(savedJobId);
            startPolling(savedJobId);
            setRestoredJob(true); // Mark as restored
            return; // Found saved job, no need to check for others
          } else if (savedJobStatus.status === 'completed' || savedJobStatus.status === 'failed') {
            // Clean up completed/failed job from localStorage
            localStorage.removeItem('currentClusteringJobId');
          }
        } catch (error: any) {
          // Job no longer exists or was cleaned up, clean up localStorage
          localStorage.removeItem('currentClusteringJobId');
          console.log('Saved job no longer exists or was cleaned up:', savedJobId);
        }
      }

      // If no saved job or saved job is no longer active, check for any active jobs
      const activeJobsData = await api.getActiveJobs();
      
      // Find the most recent active job
      const activeJob = activeJobsData.jobs.find(job => 
        job.status === 'running' || job.status === 'pending'
      );
      
      if (activeJob) {
        console.log('Found active clustering job:', activeJob.job_id);
        setCurrentJobId(activeJob.job_id);
        startPolling(activeJob.job_id);
        setRestoredJob(true); // Mark as restored
        // Save to localStorage for future restoration
        localStorage.setItem('currentClusteringJobId', activeJob.job_id);
      }
    } catch (error) {
      console.error('Failed to check for active jobs:', error);
    }
  };

  const checkOpenAISettings = async () => {
    try {
      const settings = await api.getSettings();
      setOpenAIEnabled(settings.openai.enabled);
      setEmbeddingEnabled(settings.embedding?.enabled || settings.openai.enabled); // Fallback to OpenAI if embedding not configured
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  useEffect(() => {
    // Stop polling and clear job when job is completed or failed
    if (jobStatus && (jobStatus.status === 'completed' || jobStatus.status === 'failed')) {
      // Clean up localStorage when job completes or fails
      localStorage.removeItem('currentClusteringJobId');
      
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
      
      // Validate custom model name if needed
      if (config.clustering_method === 'llm' && config.llm_model === 'custom') {
        if (!config.custom_model_name || !config.custom_model_name.trim()) {
          setError('Custom model name is required when using custom model');
          return;
        }
      }
      
      const response = await api.runClustering(config);
      setCurrentJobId(response.job_id);
      startPolling(response.job_id);
      
      // Save job ID to localStorage for persistence
      localStorage.setItem('currentClusteringJobId', response.job_id);
    } catch (err: any) {
      setError(err.message || 'Failed to start clustering');
    }
  };

  const stopClustering = () => {
    stopPolling();
    setCurrentJobId(null);
    // Clean up localStorage when manually stopping
    localStorage.removeItem('currentClusteringJobId');
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

      {/* WebSocket Status and Restored Job Notice */}
      {currentJobId && (
        <>
          <Alert 
            severity={isConnected ? 'success' : 'warning'} 
            sx={{ mb: 2 }}
          >
            Real-time updates: {isConnected ? 'Connected' : 'Disconnected'}
          </Alert>
          
          {restoredJob && (
            <Alert 
              severity="info" 
              sx={{ mb: 3 }}
              onClose={() => setRestoredJob(false)}
            >
              Restored in-progress clustering job. Progress tracking will continue from where you left off.
            </Alert>
          )}
        </>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={6}>
          <MuiPaper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>

            {/* Clustering Method Selection */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <FormLabel sx={{ mb: 1 }}>Clustering Method</FormLabel>
              <ToggleButtonGroup
                value={config.clustering_method}
                exclusive
                onChange={(_, newMethod) => {
                  if (newMethod !== null) {
                    setConfig(prev => ({ ...prev, clustering_method: newMethod }));
                  }
                }}
                aria-label="clustering method"
                fullWidth
              >
                <ToggleButton value="traditional" aria-label="traditional clustering">
                  <TraditionalIcon sx={{ mr: 1 }} />
                  Traditional (TF-IDF + K-means)
                </ToggleButton>
                <ToggleButton 
                  value="llm" 
                  aria-label="llm clustering"
                  disabled={!openAIEnabled}
                >
                  <AIIcon sx={{ mr: 1 }} />
                  LLM Semantic
                </ToggleButton>
                <ToggleButton 
                  value="embedding" 
                  aria-label="embedding clustering"
                  disabled={!embeddingEnabled}
                >
                  <EmbeddingIcon sx={{ mr: 1 }} />
                  Embedding-based
                </ToggleButton>
              </ToggleButtonGroup>
              {!openAIEnabled && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  LLM clustering requires OpenAI configuration in settings
                </Typography>
              )}
              {!embeddingEnabled && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  Embedding clustering requires embedding model or OpenAI configuration in settings
                </Typography>
              )}
            </FormControl>

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

            {/* Traditional Clustering Options */}
            <Collapse in={config.clustering_method === 'traditional'}>
              <TextField
                select
                fullWidth
                label="Feature Extraction Method"
                value={config.feature_extraction_method || 'tfidf'}
                onChange={handleConfigChange('feature_extraction_method')}
                margin="normal"
                helperText="Choose how to convert text to numerical features"
              >
                <MenuItem value="tfidf">TF-IDF (Term Frequency)</MenuItem>
                <MenuItem value="sentence_transformer">Sentence Transformers (Semantic)</MenuItem>
              </TextField>

              {/* Sentence Transformer Model Selection */}
              <Collapse in={config.feature_extraction_method === 'sentence_transformer'}>
                <TextField
                  select
                  fullWidth
                  label="Sentence Transformer Model"
                  value={config.sentence_transformer_model || 'all-MiniLM-L6-v2'}
                  onChange={handleConfigChange('sentence_transformer_model')}
                  margin="normal"
                  helperText="Choose the sentence transformer model for embeddings"
                >
                  <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Recommended - Fast & Good)</MenuItem>
                  <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2 (Better Quality, Slower)</MenuItem>
                  <MenuItem value="all-distilroberta-v1">all-distilroberta-v1 (Good Balance)</MenuItem>
                  <MenuItem value="paraphrase-MiniLM-L6-v2">paraphrase-MiniLM-L6-v2 (Paraphrase Detection)</MenuItem>
                  <MenuItem value="multi-qa-MiniLM-L6-cos-v1">multi-qa-MiniLM-L6-cos-v1 (Question Answering)</MenuItem>
                </TextField>
              </Collapse>

              <TextField
                select
                fullWidth
                label="Clustering Algorithm"
                value={config.traditional_algorithm || 'kmeans'}
                onChange={handleConfigChange('traditional_algorithm')}
                margin="normal"
                helperText="Choose the clustering algorithm for traditional clustering"
              >
                <MenuItem value="kmeans">K-Means (Recommended)</MenuItem>
                <MenuItem value="agglomerative">Agglomerative (Hierarchical)</MenuItem>
                <MenuItem value="dbscan">DBSCAN (Density-based)</MenuItem>
                <MenuItem value="spectral">Spectral Clustering</MenuItem>
              </TextField>

              {/* TF-IDF Parameters */}
              <Collapse in={config.feature_extraction_method === 'tfidf'}>
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
              </Collapse>

              {/* DBSCAN Parameters */}
              <Collapse in={config.traditional_algorithm === 'dbscan'}>
                <FormControl fullWidth sx={{ mt: 2, mb: 2 }}>
                  <FormLabel>DBSCAN Epsilon (eps): {config.dbscan_eps}</FormLabel>
                  <Slider
                    value={config.dbscan_eps || 0.5}
                    onChange={handleSliderChange('dbscan_eps')}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    marks={[
                      { value: 0.3, label: '0.3' },
                      { value: 0.5, label: '0.5' },
                      { value: 1.0, label: '1.0' },
                      { value: 1.5, label: '1.5' },
                    ]}
                  />
                </FormControl>

                <FormControl fullWidth sx={{ mt: 2, mb: 2 }}>
                  <FormLabel>DBSCAN Min Samples: {config.dbscan_min_samples}</FormLabel>
                  <Slider
                    value={config.dbscan_min_samples || 5}
                    onChange={handleSliderChange('dbscan_min_samples')}
                    min={2}
                    max={20}
                    step={1}
                    marks={[
                      { value: 3, label: '3' },
                      { value: 5, label: '5' },
                      { value: 10, label: '10' },
                      { value: 15, label: '15' },
                    ]}
                  />
                </FormControl>
              </Collapse>

              {/* Agglomerative Parameters */}
              <Collapse in={config.traditional_algorithm === 'agglomerative'}>
                <TextField
                  select
                  fullWidth
                  label="Linkage Method"
                  value={config.agglomerative_linkage || 'ward'}
                  onChange={handleConfigChange('agglomerative_linkage')}
                  margin="normal"
                  helperText="Linkage criterion for agglomerative clustering"
                >
                  <MenuItem value="ward">Ward (Recommended)</MenuItem>
                  <MenuItem value="complete">Complete</MenuItem>
                  <MenuItem value="average">Average</MenuItem>
                  <MenuItem value="single">Single</MenuItem>
                </TextField>
              </Collapse>

              {/* Spectral Parameters */}
              <Collapse in={config.traditional_algorithm === 'spectral'}>
                <TextField
                  select
                  fullWidth
                  label="Label Assignment"
                  value={config.spectral_assign_labels || 'discretize'}
                  onChange={handleConfigChange('spectral_assign_labels')}
                  margin="normal"
                  helperText="Method for assigning labels in spectral clustering"
                >
                  <MenuItem value="discretize">Discretize (Recommended)</MenuItem>
                  <MenuItem value="kmeans">K-Means</MenuItem>
                </TextField>
              </Collapse>
            </Collapse>

            {/* LLM Clustering Options */}
            <Collapse in={config.clustering_method === 'llm'}>
              <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
                LLM clustering uses semantic understanding to group papers by research themes.
                This method is more intelligent but limited to fewer papers due to API costs.
              </Alert>

              <TextField
                select
                fullWidth
                label="LLM Model"
                value={config.llm_model || 'gpt-4o'}
                onChange={handleConfigChange('llm_model')}
                margin="normal"
                helperText="Choose the OpenAI model for clustering analysis"
              >
                <MenuItem value="gpt-4o">GPT-4o (Recommended)</MenuItem>
                <MenuItem value="gpt-4o-mini">GPT-4o Mini (Faster, less accurate)</MenuItem>
                <MenuItem value="gpt-4-turbo">GPT-4 Turbo</MenuItem>
                <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo (Cheapest)</MenuItem>
                <MenuItem value="custom">Custom Model...</MenuItem>
              </TextField>

              {config.llm_model === 'custom' && (
                <TextField
                  fullWidth
                  label="Custom Model Name"
                  value={config.custom_model_name || ''}
                  onChange={handleConfigChange('custom_model_name')}
                  margin="normal"
                  placeholder="e.g., gpt-4-1106-preview, claude-3-sonnet-20240229"
                  helperText="Enter the exact model name as supported by your OpenAI-compatible API"
                />
              )}

              <FormControl fullWidth sx={{ mt: 3, mb: 2 }}>
                <FormLabel>Maximum Papers for LLM: {config.max_papers_llm}</FormLabel>
                <Slider
                  value={config.max_papers_llm || 500}
                  onChange={handleSliderChange('max_papers_llm')}
                  min={10}
                  max={500}
                  step={10}
                  marks={[
                    { value: 50, label: '50' },
                    { value: 100, label: '100' },
                    { value: 250, label: '250' },
                    { value: 500, label: '500' },
                  ]}
                />
                <Typography variant="caption" color="text.secondary">
                  Higher values may increase costs and processing time
                </Typography>
              </FormControl>
            </Collapse>

            {/* Embedding Clustering Options */}
            <Collapse in={config.clustering_method === 'embedding'}>
              <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
                Embedding clustering uses semantic embeddings from language models combined with traditional clustering algorithms.
                This method provides high-quality semantic clustering with better scalability than pure LLM clustering.
              </Alert>

              <TextField
                select
                fullWidth
                label="Embedding Model"
                value={config.embedding_model || 'text-embedding-ada-002'}
                onChange={handleConfigChange('embedding_model')}
                margin="normal"
                helperText="Choose the embedding model for generating text representations"
              >
                <MenuItem value="text-embedding-ada-002">text-embedding-ada-002 (OpenAI)</MenuItem>
                <MenuItem value="text-embedding-3-small">text-embedding-3-small (OpenAI)</MenuItem>
                <MenuItem value="text-embedding-3-large">text-embedding-3-large (OpenAI)</MenuItem>
              </TextField>

              <TextField
                select
                fullWidth
                label="Clustering Algorithm"
                value={config.embedding_clustering_algorithm || 'kmeans'}
                onChange={handleConfigChange('embedding_clustering_algorithm')}
                margin="normal"
                helperText="Choose the clustering algorithm to apply to embeddings"
              >
                <MenuItem value="kmeans">K-means (Recommended)</MenuItem>
                <MenuItem value="dbscan">DBSCAN (Density-based)</MenuItem>
                <MenuItem value="agglomerative">Agglomerative (Hierarchical)</MenuItem>
              </TextField>

              <FormControl fullWidth sx={{ mt: 2 }}>
                <FormLabel>Embedding Batch Size: {config.embedding_batch_size}</FormLabel>
                <Slider
                  value={config.embedding_batch_size || 50}
                  onChange={handleSliderChange('embedding_batch_size')}
                  min={10}
                  max={200}
                  step={10}
                  marks={[
                    { value: 10, label: '10' },
                    { value: 50, label: '50' },
                    { value: 100, label: '100' },
                    { value: 200, label: '200' },
                  ]}
                />
                <Typography variant="caption" color="text.secondary">
                  Number of texts to process in each API call
                </Typography>
              </FormControl>

              {/* DBSCAN-specific parameters */}
              <Collapse in={config.embedding_clustering_algorithm === 'dbscan'}>
                <Box sx={{ mt: 2 }}>
                  <TextField
                    type="number"
                    label="DBSCAN Epsilon"
                    value={config.dbscan_eps || 0.5}
                    onChange={handleConfigChange('dbscan_eps')}
                    margin="normal"
                    inputProps={{ step: 0.1, min: 0.1, max: 2.0 }}
                    helperText="Maximum distance between two samples to be considered as in the same neighborhood"
                    sx={{ mr: 2, width: '48%' }}
                  />
                  <TextField
                    type="number"
                    label="Min Samples"
                    value={config.dbscan_min_samples || 5}
                    onChange={handleConfigChange('dbscan_min_samples')}
                    margin="normal"
                    inputProps={{ step: 1, min: 2, max: 20 }}
                    helperText="Minimum number of samples in a neighborhood for a point to be considered as a core point"
                    sx={{ width: '48%' }}
                  />
                </Box>
              </Collapse>

              {/* Agglomerative-specific parameters */}
              <Collapse in={config.embedding_clustering_algorithm === 'agglomerative'}>
                <TextField
                  select
                  fullWidth
                  label="Linkage Method"
                  value={config.agglomerative_linkage || 'ward'}
                  onChange={handleConfigChange('agglomerative_linkage')}
                  margin="normal"
                  helperText="Linkage criterion for agglomerative clustering"
                >
                  <MenuItem value="ward">Ward (Recommended)</MenuItem>
                  <MenuItem value="complete">Complete</MenuItem>
                  <MenuItem value="average">Average</MenuItem>
                  <MenuItem value="single">Single</MenuItem>
                </TextField>
              </Collapse>
            </Collapse>

            <FormControl fullWidth sx={{ mt: 3, mb: 2 }}>
              <FormLabel>Maximum Clusters: {config.max_k}</FormLabel>
              <Slider
                value={config.max_k || 25}  // Updated default from 15 to 25
                onChange={handleSliderChange('max_k')}
                min={2}
                max={30}
                step={1}
                marks={[
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 15, label: '15' },
                  { value: 20, label: '20' },
                  { value: 25, label: '25' },
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
                disabled={
                  isRunning || 
                  (config.clustering_method === 'llm' && !openAIEnabled) ||
                  (config.clustering_method === 'llm' && config.llm_model === 'custom' && (!config.custom_model_name || !config.custom_model_name.trim())) ||
                  (config.clustering_method === 'embedding' && !embeddingEnabled)
                }
                startIcon={<PlayIcon />}
                fullWidth
              >
                {isRunning ? 'Running...' : `Start ${
                  config.clustering_method === 'llm' ? 'LLM' : 
                  config.clustering_method === 'embedding' ? 'Embedding' : 
                  'Traditional'
                } Clustering`}
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
                          {jobStatus.result.clustering_method === 'traditional' && jobStatus.result.silhouette_score && (
                            <>• Silhouette score: {jobStatus.result.silhouette_score?.toFixed(3)}<br /></>
                          )}
                          {jobStatus.result.clustering_method === 'llm' && (
                            <>
                              • LLM Model: {jobStatus.result.llm_model}<br />
                              {jobStatus.result.unassigned_papers > 0 && (
                                <>• Unassigned papers: {jobStatus.result.unassigned_papers}<br /></>
                              )}
                            </>
                          )}
                          • Method: {jobStatus.result.clustering_method === 'llm' ? 'LLM Semantic' : 'Traditional K-means'}
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
import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  FormControlLabel,
  Checkbox,
  Divider,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import MergeIcon from '@mui/icons-material/Merge';
import { api, Dataset } from '../services/api';
import { useJobStatus } from '../hooks/useWebSocket';

interface FileUploadProps {
  onUploadComplete?: (jobId: string) => void;
  onError?: (error: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onError }) => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [datasetName, setDatasetName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  // Dataset options
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [uploadToExisting, setUploadToExisting] = useState(false);
  const [existingDataset, setExistingDataset] = useState('');
  
  // Dataset merging
  const [isMergeDialogOpen, setIsMergeDialogOpen] = useState(false);
  const [sourceDataset, setSourceDataset] = useState('');
  const [targetDataset, setTargetDataset] = useState('');
  const [deleteSource, setDeleteSource] = useState(true);
  const [isMerging, setIsMerging] = useState(false);
  
  // AI processing options
  const [autoProcessAI, setAutoProcessAI] = useState(false);
  const [generateSummary, setGenerateSummary] = useState(true);
  const [generateKeywords, setGenerateKeywords] = useState(true);
  const [aiJobId, setAiJobId] = useState<string | null>(null);
  
  // OpenAI settings state
  const [openAIEnabled, setOpenAIEnabled] = useState(false);
  
  const { status: jobStatus, startPolling } = useJobStatus();

  useEffect(() => {
    // Check OpenAI settings when component mounts
    checkOpenAISettings();
    loadDatasets();
  }, []);

  const checkOpenAISettings = async () => {
    try {
      const settings = await api.getSettings();
      setOpenAIEnabled(settings.openai.enabled);
      setAutoProcessAI(settings.processing.auto_generate_summary || settings.processing.auto_generate_keywords);
      setGenerateSummary(settings.processing.auto_generate_summary);
      setGenerateKeywords(settings.processing.auto_generate_keywords);
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const loadDatasets = async () => {
    try {
      const datasetsData = await api.getDatasets();
      setDatasets(datasetsData);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        onError?.('Please select a CSV file');
        return;
      }
      setSelectedFile(file);
      setIsDialogOpen(true);
    }
  }, [onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  const handleUpload = async () => {
    const finalDatasetName = uploadToExisting ? existingDataset : datasetName.trim();
    
    if (!selectedFile || !finalDatasetName) {
      return;
    }

    setIsUploading(true);
    try {
      // Upload the file first
      const response = await api.uploadPapers(
        selectedFile, 
        finalDatasetName, 
        description.trim(),
        uploadToExisting
      );
      
      // Start monitoring the upload job
      startPolling(response.job_id);
      
      onUploadComplete?.(response.job_id);
      
      // If auto-process with AI is enabled and OpenAI is configured
      if (autoProcessAI && openAIEnabled && (generateSummary || generateKeywords)) {
        // We'll start AI processing after upload completes
        // This will be handled by monitoring the upload job status
        setAiJobId(response.job_id);
      }
      
      // Reload datasets after upload
      loadDatasets();
      
      setIsDialogOpen(false);
      setSelectedFile(null);
      setDatasetName('');
      setExistingDataset('');
      setDescription('');
      setUploadToExisting(false);
    } catch (error: any) {
      onError?.(error.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleMergeDatasets = async () => {
    if (!sourceDataset || !targetDataset || sourceDataset === targetDataset) {
      return;
    }

    setIsMerging(true);
    try {
      const response = await api.mergeDatasets(sourceDataset, targetDataset, deleteSource);
      
      // Reload datasets after merge
      loadDatasets();
      
      setIsMergeDialogOpen(false);
      setSourceDataset('');
      setTargetDataset('');
      setDeleteSource(true);
      
      // Show success message
      onUploadComplete?.(`merge-${Date.now()}`);
    } catch (error: any) {
      onError?.(error.message || 'Dataset merge failed');
    } finally {
      setIsMerging(false);
    }
  };

  const handleClose = () => {
    setIsDialogOpen(false);
    setSelectedFile(null);
    setDatasetName('');
    setExistingDataset('');
    setDescription('');
    setUploadToExisting(false);
  };

  const handleMergeClose = () => {
    setIsMergeDialogOpen(false);
    setSourceDataset('');
    setTargetDataset('');
    setDeleteSource(true);
  };

  return (
    <>
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <MuiPaper
          {...getRootProps()}
          sx={{
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
            transition: 'all 0.2s ease-in-out',
            flexGrow: 1,
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'action.hover',
            },
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop the CSV file here' : 'Drag & drop a CSV file here'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            or click to select a file
          </Typography>
          <Button variant="outlined" sx={{ mt: 2 }}>
            Select File
          </Button>
        </MuiPaper>

        <MuiPaper
          sx={{
            p: 4,
            textAlign: 'center',
            minWidth: 300,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
          }}
        >
          <MergeIcon sx={{ fontSize: 48, color: 'secondary.main', mb: 2, mx: 'auto' }} />
          <Typography variant="h6" gutterBottom>
            Merge Datasets
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Combine two existing datasets with automatic deduplication
          </Typography>
          <Button
            variant="outlined"
            color="secondary"
            onClick={() => setIsMergeDialogOpen(true)}
            disabled={datasets.length < 2}
          >
            Merge Datasets
          </Button>
        </MuiPaper>
      </Box>

      {/* Upload Dialog */}
      <Dialog open={isDialogOpen} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Paper Dataset</DialogTitle>
        <DialogContent>
          {selectedFile && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Selected file: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </Alert>
          )}
          
          <FormControlLabel
            control={
              <Checkbox
                checked={uploadToExisting}
                onChange={(e) => {
                  setUploadToExisting(e.target.checked);
                  if (e.target.checked) {
                    setDatasetName('');
                  } else {
                    setExistingDataset('');
                  }
                }}
              />
            }
            label="Upload to existing dataset (with deduplication)"
            sx={{ mb: 2 }}
          />

          {uploadToExisting ? (
            <FormControl fullWidth margin="normal" required>
              <InputLabel>Existing Dataset</InputLabel>
              <Select
                value={existingDataset}
                onChange={(e) => setExistingDataset(e.target.value)}
                label="Existing Dataset"
              >
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.id} value={dataset.name}>
                    {dataset.name} ({dataset.total_papers} papers)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <TextField
              fullWidth
              label="Dataset Name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              margin="normal"
              required
              helperText="A unique name for this dataset"
            />
          )}
          
          <TextField
            fullWidth
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            margin="normal"
            multiline
            rows={3}
            helperText={uploadToExisting ? 
              "Optional description (will be appended to existing dataset)" : 
              "Optional description for this dataset"
            }
          />

          {uploadToExisting && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Papers will be checked for duplicates based on DOI and title/abstract similarity. 
              Existing papers with matching content will be skipped.
            </Alert>
          )}

          {openAIEnabled && (
            <>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                AI Processing Options
              </Typography>
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={autoProcessAI}
                    onChange={(e) => setAutoProcessAI(e.target.checked)}
                  />
                }
                label="Auto-process with AI after upload"
              />
              
              {autoProcessAI && (
                <Box sx={{ ml: 3, mt: 1 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={generateSummary}
                        onChange={(e) => setGenerateSummary(e.target.checked)}
                      />
                    }
                    label="Generate summaries"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={generateKeywords}
                        onChange={(e) => setGenerateKeywords(e.target.checked)}
                      />
                    }
                    label="Generate keywords"
                  />
                </Box>
              )}
            </>
          )}

          {jobStatus && (
            <Box sx={{ mt: 2 }}>
              <Alert severity={
                jobStatus.status === 'completed' ? 'success' :
                jobStatus.status === 'failed' ? 'error' :
                'info'
              }>
                <Typography variant="subtitle2">
                  Status: {jobStatus.status}
                </Typography>
                <Typography variant="body2">
                  {jobStatus.message}
                </Typography>
                {jobStatus.status === 'running' && (
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <CircularProgress size={16} sx={{ mr: 1 }} />
                    <Typography variant="body2">
                      Progress: {Math.round(jobStatus.progress * 100)}%
                    </Typography>
                  </Box>
                )}
              </Alert>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} disabled={isUploading}>
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            variant="contained"
            disabled={
              !selectedFile || 
              (!uploadToExisting && !datasetName.trim()) ||
              (uploadToExisting && !existingDataset) ||
              isUploading
            }
            startIcon={isUploading ? <CircularProgress size={16} /> : undefined}
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Dataset Merge Dialog */}
      <Dialog open={isMergeDialogOpen} onClose={handleMergeClose} maxWidth="sm" fullWidth>
        <DialogTitle>Merge Datasets</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            This will combine all papers from the source dataset into the target dataset. 
            Duplicate papers (based on DOI and title/abstract similarity) will be automatically removed.
          </Alert>

          <FormControl fullWidth margin="normal" required>
            <InputLabel>Source Dataset (merge from)</InputLabel>
            <Select
              value={sourceDataset}
              onChange={(e) => setSourceDataset(e.target.value)}
              label="Source Dataset (merge from)"
            >
              {datasets.map((dataset) => (
                <MenuItem 
                  key={dataset.id} 
                  value={dataset.name}
                  disabled={dataset.name === targetDataset}
                >
                  {dataset.name} ({dataset.total_papers} papers)
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal" required>
            <InputLabel>Target Dataset (merge into)</InputLabel>
            <Select
              value={targetDataset}
              onChange={(e) => setTargetDataset(e.target.value)}
              label="Target Dataset (merge into)"
            >
              {datasets.map((dataset) => (
                <MenuItem 
                  key={dataset.id} 
                  value={dataset.name}
                  disabled={dataset.name === sourceDataset}
                >
                  {dataset.name} ({dataset.total_papers} papers)
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Checkbox
                checked={deleteSource}
                onChange={(e) => setDeleteSource(e.target.checked)}
              />
            }
            label="Delete source dataset after merge"
            sx={{ mt: 2 }}
          />

          {sourceDataset && targetDataset && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              This action will move all papers from "{sourceDataset}" into "{targetDataset}".
              {deleteSource && ` The "${sourceDataset}" dataset will be deleted.`}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleMergeClose} disabled={isMerging}>
            Cancel
          </Button>
          <Button
            onClick={handleMergeDatasets}
            variant="contained"
            color="secondary"
            disabled={!sourceDataset || !targetDataset || sourceDataset === targetDataset || isMerging}
            startIcon={isMerging ? <CircularProgress size={16} /> : undefined}
          >
            {isMerging ? 'Merging...' : 'Merge Datasets'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FileUpload;
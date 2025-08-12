import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  Button,
  Checkbox,
  FormControlLabel,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TextField,
  MenuItem,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormGroup,
} from '@mui/material';
import {
  PlayArrow as ProcessIcon,
  Refresh as RefreshIcon,
  SelectAll as SelectAllIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { api, UnprocessedPaper, Dataset, AIProcessRequest } from '../services/api';
import { useJobStatus } from '../hooks/useWebSocket';

const AIProcessingPage: React.FC = () => {
  const [unprocessedPapers, setUnprocessedPapers] = useState<UnprocessedPaper[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [missingSummary, setMissingSummary] = useState(true);
  const [missingKeywords, setMissingKeywords] = useState(true);
  
  // Selection
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  
  // Processing options
  const [generateSummary, setGenerateSummary] = useState(true);
  const [generateKeywords, setGenerateKeywords] = useState(true);
  const [overwriteExisting, setOverwriteExisting] = useState(false);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  
  // Dialog
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  
  const { status: jobStatus, startPolling, stopPolling } = useJobStatus();

  useEffect(() => {
    // Only load data on initial mount
    loadData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (jobStatus && (jobStatus.status === 'completed' || jobStatus.status === 'failed')) {
      setIsProcessing(false);
      if (jobStatus.status === 'completed') {
        // Reload data to show updated papers
        loadData();
        setSelectedPapers(new Set());
      }
    }
  }, [jobStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [papersData, datasetsData] = await Promise.all([
        api.getUnprocessedPapers({
          dataset: selectedDataset || undefined,
          missing_summary: missingSummary,
          missing_keywords: missingKeywords,
          limit: 1000,
        }),
        api.getDatasets(),
      ]);
      
      setUnprocessedPapers(papersData);
      setDatasets(datasetsData);
      
    } catch (err: any) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    // Manual filter application - refreshes data with current filter settings
    loadData();
  };

  const handleSelectAll = () => {
    if (selectedPapers.size === currentPagePapers.length) {
      setSelectedPapers(new Set());
    } else {
      setSelectedPapers(new Set(currentPagePapers.map(p => p.paper_id)));
    }
  };

  const handleSelectPaper = (paperId: string) => {
    const newSelection = new Set(selectedPapers);
    if (newSelection.has(paperId)) {
      newSelection.delete(paperId);
    } else {
      newSelection.add(paperId);
    }
    setSelectedPapers(newSelection);
  };

  const startProcessing = async () => {
    if (selectedPapers.size === 0) return;
    
    try {
      setIsProcessing(true);
      setError(null);
      
      const request: AIProcessRequest = {
        paper_ids: Array.from(selectedPapers),
        generate_summary: generateSummary,
        generate_keywords: generateKeywords,
        overwrite_existing: overwriteExisting,
      };
      
      const response = await api.processWithAI(request);
      startPolling(response.job_id);
      setConfirmDialogOpen(false);
      
    } catch (err: any) {
      setError(err.message || 'Failed to start processing');
      setIsProcessing(false);
    }
  };

  const stopProcessing = () => {
    stopPolling();
    setIsProcessing(false);
  };

  // Pagination
  const currentPagePapers = unprocessedPapers.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  const handlePageChange = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleRowsPerPageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        AI Processing
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Generate summaries and keywords for papers using AI. Select papers to process and configure the processing options.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Filters and Options */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Filters & Options" />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  select
                  fullWidth
                  label="Dataset"
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  size="small"
                >
                  <MenuItem value="">All Datasets</MenuItem>
                  {datasets.map((dataset) => (
                    <MenuItem key={dataset.id} value={dataset.name}>
                      {dataset.name} ({dataset.total_papers} papers)
                    </MenuItem>
                  ))}
                </TextField>

                <Typography variant="subtitle2">Show papers missing:</Typography>
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={missingSummary}
                        onChange={(e) => setMissingSummary(e.target.checked)}
                      />
                    }
                    label="Summary"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={missingKeywords}
                        onChange={(e) => setMissingKeywords(e.target.checked)}
                      />
                    }
                    label="Keywords"
                  />
                </FormGroup>

                <Button
                  variant="outlined"
                  onClick={applyFilters}
                  disabled={loading}
                  size="small"
                  sx={{ mt: 1 }}
                >
                  Apply Filters
                </Button>

                <Typography variant="subtitle2" sx={{ mt: 2 }}>Processing Options:</Typography>
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={generateSummary}
                        onChange={(e) => setGenerateSummary(e.target.checked)}
                      />
                    }
                    label="Generate Summaries"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={generateKeywords}
                        onChange={(e) => setGenerateKeywords(e.target.checked)}
                      />
                    }
                    label="Generate Keywords"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={overwriteExisting}
                        onChange={(e) => setOverwriteExisting(e.target.checked)}
                      />
                    }
                    label="Overwrite Existing"
                  />
                </FormGroup>

                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                  <Button
                    variant="contained"
                    onClick={() => setConfirmDialogOpen(true)}
                    disabled={selectedPapers.size === 0 || isProcessing}
                    startIcon={<ProcessIcon />}
                    fullWidth
                  >
                    Process Selected ({selectedPapers.size})
                  </Button>
                </Box>

                {isProcessing && (
                  <Button
                    variant="outlined"
                    onClick={stopProcessing}
                    color="error"
                    size="small"
                  >
                    Stop Processing
                  </Button>
                )}
              </Box>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {jobStatus && isProcessing && (
            <Card sx={{ mt: 2 }}>
              <CardHeader title="Processing Status" />
              <CardContent>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Status: <Chip 
                      label={jobStatus.status.toUpperCase()} 
                      color={jobStatus.status === 'running' ? 'primary' : 'default'}
                      size="small"
                    />
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Progress: {Math.round(jobStatus.progress * 100)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={jobStatus.progress * 100} 
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {jobStatus.message}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Papers Table */}
        <Grid item xs={12} md={8}>
          <MuiPaper>
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6">
                Unprocessed Papers ({unprocessedPapers.length})
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <IconButton onClick={handleSelectAll} title="Select All">
                  <SelectAllIcon />
                </IconButton>
                <IconButton 
                  onClick={() => setSelectedPapers(new Set())} 
                  title="Clear Selection"
                  disabled={selectedPapers.size === 0}
                >
                  <ClearIcon />
                </IconButton>
                <IconButton onClick={loadData} title="Refresh">
                  <RefreshIcon />
                </IconButton>
              </Box>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedPapers.size === currentPagePapers.length && currentPagePapers.length > 0}
                        indeterminate={selectedPapers.size > 0 && selectedPapers.size < currentPagePapers.length}
                        onChange={handleSelectAll}
                      />
                    </TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Dataset</TableCell>
                    <TableCell>Missing</TableCell>
                    <TableCell>Created</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        Loading...
                      </TableCell>
                    </TableRow>
                  ) : currentPagePapers.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        No unprocessed papers found
                      </TableCell>
                    </TableRow>
                  ) : (
                    currentPagePapers.map((paper) => (
                      <TableRow key={paper.id} hover>
                        <TableCell padding="checkbox">
                          <Checkbox
                            checked={selectedPapers.has(paper.paper_id)}
                            onChange={() => handleSelectPaper(paper.paper_id)}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {paper.title}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip label={paper.source_dataset} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            {!paper.has_summary && (
                              <Chip label="Summary" size="small" color="warning" />
                            )}
                            {!paper.has_keywords && (
                              <Chip label="Keywords" size="small" color="info" />
                            )}
                          </Box>
                        </TableCell>
                        <TableCell>
                          {new Date(paper.created_at).toLocaleDateString()}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              rowsPerPageOptions={[10, 25, 50, 100]}
              component="div"
              count={unprocessedPapers.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handlePageChange}
              onRowsPerPageChange={handleRowsPerPageChange}
            />
          </MuiPaper>
        </Grid>
      </Grid>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialogOpen} onClose={() => setConfirmDialogOpen(false)}>
        <DialogTitle>
          Confirm AI Processing
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            You are about to process {selectedPapers.size} papers with the following options:
          </Typography>
          <Box sx={{ ml: 2, mt: 2 }}>
            <Typography variant="body2">
              • Generate Summaries: {generateSummary ? 'Yes' : 'No'}
            </Typography>
            <Typography variant="body2">
              • Generate Keywords: {generateKeywords ? 'Yes' : 'No'}
            </Typography>
            <Typography variant="body2">
              • Overwrite Existing: {overwriteExisting ? 'Yes' : 'No'}
            </Typography>
          </Box>
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              This process may take several minutes depending on the number of papers. 
              You can monitor progress in real-time and continue using other features.
            </Typography>
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={startProcessing} variant="contained" autoFocus>
            Start Processing
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AIProcessingPage;
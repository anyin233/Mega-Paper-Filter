import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
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
  Button,
  Skeleton,
  Alert,
} from '@mui/material';
import {
  Search as SearchIcon,
  Info as InfoIcon,
  Launch as LaunchIcon,
} from '@mui/icons-material';
import { api, Paper, Dataset } from '../services/api';

const PaperList: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  
  // Pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [totalPapers, setTotalPapers] = useState(0);
  
  // Paper details dialog
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    loadPapers();
  }, [page, rowsPerPage, searchQuery, selectedDataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadDatasets = async () => {
    try {
      const datasetsData = await api.getDatasets();
      setDatasets(datasetsData);
    } catch (err: any) {
      console.error('Failed to load datasets:', err);
    }
  };

  const loadPapers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.getPapers({
        dataset: selectedDataset || undefined,
        search: searchQuery || undefined,
        limit: rowsPerPage,
        offset: page * rowsPerPage,
      });
      
      setPapers(response.papers);
      setTotalPapers(response.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load papers');
    } finally {
      setLoading(false);
    }
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
    setPage(0); // Reset to first page when searching
  };

  const handleDatasetChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedDataset(event.target.value);
    setPage(0); // Reset to first page when filtering
  };

  const handlePageChange = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleRowsPerPageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const openPaperDetails = (paper: Paper) => {
    setSelectedPaper(paper);
    setDetailsOpen(true);
  };

  const closePaperDetails = () => {
    setDetailsOpen(false);
    setSelectedPaper(null);
  };

  if (loading && papers.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Papers
        </Typography>
        {[...Array(10)].map((_, i) => (
          <Skeleton key={i} variant="rectangular" height={60} sx={{ mb: 1 }} />
        ))}
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Papers
      </Typography>

      {/* Filters */}
      <MuiPaper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <TextField
            label="Search papers"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search by title, abstract, or keywords..."
            sx={{ flexGrow: 1, minWidth: 250 }}
            InputProps={{
              startAdornment: <SearchIcon sx={{ color: 'action.active', mr: 1 }} />,
            }}
          />
          
          <TextField
            select
            label="Dataset"
            value={selectedDataset}
            onChange={handleDatasetChange}
            sx={{ minWidth: 200 }}
          >
            <MenuItem value="">All Datasets</MenuItem>
            {datasets.map((dataset) => (
              <MenuItem key={dataset.id} value={dataset.name}>
                {dataset.name} ({dataset.total_papers} papers)
              </MenuItem>
            ))}
          </TextField>
        </Box>
      </MuiPaper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} action={
          <Button color="inherit" size="small" onClick={loadPapers}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      )}

      {/* Papers Table */}
      <MuiPaper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>Authors</TableCell>
                <TableCell>Year</TableCell>
                <TableCell>Venue</TableCell>
                <TableCell>Dataset</TableCell>
                <TableCell>Keywords</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {papers.map((paper) => (
                <TableRow key={paper.id} hover>
                  <TableCell>
                    <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                      {paper.title}
                    </Typography>
                    {paper.abstract && (
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {paper.abstract.substring(0, 100)}...
                      </Typography>
                    )}
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2">
                      {paper.authors.slice(0, 3).join(', ')}
                      {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    {paper.publication_year || 'N/A'}
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
                      {paper.venue || 'N/A'}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Chip 
                      label={paper.source_dataset} 
                      size="small" 
                      variant="outlined" 
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {paper.keywords.slice(0, 3).map((keyword, index) => (
                        <Chip 
                          key={index} 
                          label={keyword} 
                          size="small" 
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                      {paper.keywords.length > 3 && (
                        <Chip 
                          label={`+${paper.keywords.length - 3}`} 
                          size="small" 
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </TableCell>
                  
                  <TableCell align="center">
                    <IconButton 
                      onClick={() => openPaperDetails(paper)}
                      title="View Details"
                    >
                      <InfoIcon />
                    </IconButton>
                    {paper.url && (
                      <IconButton 
                        onClick={() => window.open(paper.url, '_blank')}
                        title="Open URL"
                      >
                        <LaunchIcon />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <TablePagination
          rowsPerPageOptions={[10, 25, 50, 100]}
          component="div"
          count={totalPapers}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handleRowsPerPageChange}
        />
      </MuiPaper>

      {/* Paper Details Dialog */}
      <Dialog 
        open={detailsOpen} 
        onClose={closePaperDetails} 
        maxWidth="md" 
        fullWidth
        scroll="body"
      >
        {selectedPaper && (
          <>
            <DialogTitle>
              <Typography variant="h6">
                {selectedPaper.title}
              </Typography>
            </DialogTitle>
            <DialogContent dividers>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Authors:
                </Typography>
                <Typography variant="body2">
                  {selectedPaper.authors.join(', ')}
                </Typography>
              </Box>

              {selectedPaper.abstract && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Abstract:
                  </Typography>
                  <Typography variant="body2">
                    {selectedPaper.abstract}
                  </Typography>
                </Box>
              )}

              {selectedPaper.summary && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    AI Summary:
                  </Typography>
                  <Typography variant="body2">
                    {selectedPaper.summary}
                  </Typography>
                </Box>
              )}

              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Details:
                </Typography>
                <Typography variant="body2" component="div">
                  <strong>Year:</strong> {selectedPaper.publication_year || 'N/A'}<br />
                  <strong>Venue:</strong> {selectedPaper.venue || 'N/A'}<br />
                  <strong>DOI:</strong> {selectedPaper.doi || 'N/A'}<br />
                  <strong>Dataset:</strong> {selectedPaper.source_dataset}<br />
                  <strong>Paper ID:</strong> {selectedPaper.paper_id}
                </Typography>
              </Box>

              {selectedPaper.keywords.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Keywords:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {selectedPaper.keywords.map((keyword, index) => (
                      <Chip 
                        key={index} 
                        label={keyword} 
                        size="small" 
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </DialogContent>
            <DialogActions>
              {selectedPaper.url && (
                <Button 
                  onClick={() => window.open(selectedPaper.url, '_blank')}
                  startIcon={<LaunchIcon />}
                >
                  Open URL
                </Button>
              )}
              <Button onClick={closePaperDetails}>
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default PaperList;
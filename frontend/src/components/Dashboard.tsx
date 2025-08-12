import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { api, Dataset, Statistics } from '../services/api';

const Dashboard: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [datasetsData, statsData] = await Promise.all([
        api.getDatasets(),
        api.getStatistics(),
      ]);
      
      setDatasets(datasetsData);
      setStatistics(statsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={120} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={loadDashboardData}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {statistics && (
        <>
          {/* Statistics Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total Papers
                  </Typography>
                  <Typography variant="h4">
                    {statistics.total_papers.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Datasets
                  </Typography>
                  <Typography variant="h4">
                    {statistics.total_datasets}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    With Summaries
                  </Typography>
                  <Typography variant="h4">
                    {statistics.papers_with_summary}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {statistics.total_papers > 0 
                      ? `${Math.round((statistics.papers_with_summary / statistics.total_papers) * 100)}%`
                      : '0%'
                    }
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    With Keywords
                  </Typography>
                  <Typography variant="h4">
                    {statistics.papers_with_keywords}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {statistics.total_papers > 0 
                      ? `${Math.round((statistics.papers_with_keywords / statistics.total_papers) * 100)}%`
                      : '0%'
                    }
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {/* Papers by Dataset Bar Chart */}
            <Grid item xs={12} md={8}>
              <MuiPaper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Papers by Dataset
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={statistics.datasets_breakdown}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="source_dataset" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </MuiPaper>
            </Grid>

            {/* Dataset Distribution Pie Chart */}
            <Grid item xs={12} md={4}>
              <MuiPaper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Dataset Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={statistics.datasets_breakdown}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ source_dataset, percent }) => 
                        `${source_dataset}: ${(percent * 100).toFixed(0)}%`
                      }
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {statistics.datasets_breakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </MuiPaper>
            </Grid>
          </Grid>
        </>
      )}

      {/* Datasets List */}
      <MuiPaper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Recent Datasets
        </Typography>
        <Grid container spacing={2}>
          {datasets.length === 0 ? (
            <Grid item xs={12}>
              <Alert severity="info">
                No datasets found. Upload your first CSV file to get started!
              </Alert>
            </Grid>
          ) : (
            datasets.slice(0, 6).map((dataset) => (
              <Grid item xs={12} sm={6} md={4} key={dataset.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" component="div" noWrap>
                      {dataset.name}
                    </Typography>
                    <Typography color="text.secondary" gutterBottom>
                      {dataset.total_papers} papers
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      {dataset.description || 'No description'}
                    </Typography>
                    <Chip 
                      label={new Date(dataset.created_at).toLocaleDateString()}
                      size="small"
                      variant="outlined"
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))
          )}
        </Grid>
      </MuiPaper>
    </Box>
  );
};

export default Dashboard;
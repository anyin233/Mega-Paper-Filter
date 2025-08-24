import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  Grid,
  Chip,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert,
  Skeleton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  ScatterPlot as ScatterPlotIcon,
  AccountTree as NetworkIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import * as d3 from 'd3';
import { api, SavedClusteringResult } from '../services/api';

interface ClusterData {
  papers: Array<{
    Key: string;
    Title: string;
    Author: string;
    summary: string;
    keywords: string;
    cluster_id: number;
    pca_x: number;
    pca_y: number;
    'Publication Year': string;
    DOI: string;
    Url: string;
    Venue: string;
    Abstract: string;
  }>;
  cluster_info: Record<string, {
    size: number;
    top_keywords: Array<[string, number]>;
    sample_titles: string[];
    common_paper_keywords: Record<string, number>;
    name?: string;
    description?: string;
  }>;
  total_papers: number;
  total_clusters: number;
  metadata: {
    generated_at: string;
    dataset_filter?: string;
    clustering_method: string;
    feature_extraction: string;
    total_papers_in_db: number;
    papers_used_for_clustering: number;
    pca_explained_variance: [number, number];
  };
}

interface VisualizationProps {
  jobId?: string;
  data?: ClusterData;
}

const ClusterVisualization: React.FC<VisualizationProps> = ({ jobId, data: externalData }) => {
  const [clusterData, setClusterData] = useState<ClusterData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPaper, setSelectedPaper] = useState<any>(null);
  const [paperDetailsOpen, setPaperDetailsOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'scatter' | 'network'>('scatter');
  const [showVisualization, setShowVisualization] = useState(false);
  const [loadingClusterData, setLoadingClusterData] = useState(false);
  const networkRef = useRef<SVGSVGElement>(null);
  
  // Saved clustering results
  const [savedResults, setSavedResults] = useState<SavedClusteringResult[]>([]);
  const [selectedResultId, setSelectedResultId] = useState<number | null>(null);
  const [loadingSavedResults, setLoadingSavedResults] = useState(false);

  useEffect(() => {
    if (externalData) {
      setClusterData(externalData);
      setLoading(false);
    } else if (jobId) {
      loadClusteringResults();
    } else {
      // Load saved clustering results when no specific job or data is provided
      loadSavedResults();
      setLoading(false);
    }
  }, [jobId, externalData]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadSavedResults = async () => {
    try {
      setLoadingSavedResults(true);
      const results = await api.getSavedClusteringResults();
      setSavedResults(results);
      console.log('Loaded saved clustering results:', results);
    } catch (err: any) {
      console.error('Failed to load saved results:', err);
      setError(err.message || 'Failed to load saved clustering results');
    } finally {
      setLoadingSavedResults(false);
    }
  };

  const loadSavedClusteringResult = async (resultId: number) => {
    try {
      setLoadingClusterData(true);
      setError(null);
      console.log('Loading saved clustering result:', resultId);
      const result = await api.getSavedClusteringResult(resultId);
      console.log('Saved clustering result loaded:', result);
      
      if (result.visualization_data) {
        console.log('Setting visualization data from saved result');
        setClusterData(result.visualization_data);
        setSelectedResultId(resultId);
      } else {
        setError('No visualization data found in saved result');
      }
    } catch (err: any) {
      console.error('Error loading saved clustering result:', err);
      setError(err.message || 'Failed to load saved clustering result');
    } finally {
      setLoadingClusterData(false);
    }
  };

  const loadClusteringResults = async () => {
    if (!jobId) return;
    
    try {
      setLoading(true);
      setError(null);
      console.log('Loading clustering results for job:', jobId);
      const results = await api.getClusteringResults(jobId);
      console.log('Clustering results received:', results);
      
      if (results.visualization_data) {
        console.log('Setting visualization data:', results.visualization_data);
        setClusterData(results.visualization_data);
      } else {
        console.error('No visualization data in results:', results);
        setError('No visualization data found in results');
      }
    } catch (err: any) {
      console.error('Error loading clustering results:', err);
      setError(err.message || 'Failed to load clustering results');
    } finally {
      setLoading(false);
    }
  };

  // Filter papers based on search and cluster selection
  const filteredPapers = useMemo(() => {
    if (!clusterData || !clusterData.papers) return [];
    
    let papers = clusterData.papers;
    
    // Filter by cluster
    if (selectedCluster !== 'all') {
      papers = papers.filter(paper => paper.cluster_id.toString() === selectedCluster);
    }
    
    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      papers = papers.filter(paper =>
        paper.Title.toLowerCase().includes(query) ||
        paper.Author.toLowerCase().includes(query) ||
        paper.keywords.toLowerCase().includes(query) ||
        paper.Abstract.toLowerCase().includes(query)
      );
    }
    
    return papers;
  }, [clusterData, selectedCluster, searchQuery]);

  // Generate colors for clusters
  const generateClusterColors = (numClusters: number) => {
    const colors = [];
    for (let i = 0; i < numClusters; i++) {
      const hue = (i * 360) / numClusters;
      colors.push(`hsl(${hue}, 70%, 50%)`);
    }
    return colors;
  };

  // Create scatter plot data
  const plotData = useMemo(() => {
    if (!clusterData || !clusterData.cluster_info) {
      console.log('No cluster data available for plotting');
      return [];
    }
    
    console.log('Creating plot data from cluster data:', {
      totalPapers: clusterData.total_papers || 0,
      totalClusters: clusterData.total_clusters || 0,
      filteredPapersCount: filteredPapers.length
    });
    
    const numClusters = clusterData.total_clusters || 0;
    const colors = generateClusterColors(numClusters);
    
    const traces: any[] = [];
    
    // Group papers by cluster
    const papersByCluster: Record<number, any[]> = {};
    filteredPapers.forEach(paper => {
      if (!papersByCluster[paper.cluster_id]) {
        papersByCluster[paper.cluster_id] = [];
      }
      papersByCluster[paper.cluster_id].push(paper);
    });
    
    console.log('Papers grouped by cluster:', Object.keys(papersByCluster).map(id => ({
      clusterId: id,
      count: papersByCluster[parseInt(id)].length
    })));
    
    // Create a trace for each cluster
    Object.entries(papersByCluster).forEach(([clusterId, papers]) => {
      const clusterIdNum = parseInt(clusterId);
      const clusterInfo = clusterData.cluster_info[clusterId];
      const clusterName = clusterInfo?.name ? clusterInfo.name : `Cluster ${clusterId}`;
      
      traces.push({
        x: papers.map(p => p.pca_x),
        y: papers.map(p => p.pca_y),
        mode: 'markers',
        type: 'scatter',
        name: `${clusterName} (${papers.length} papers)`,
        marker: {
          color: colors[clusterIdNum % colors.length],
          size: 8,
          opacity: 0.7,
        },
        text: papers.map(p => `${p.Title}<br>Authors: ${p.Author}`),
        hovertemplate: '%{text}<extra></extra>',
        customdata: papers,
      });
    });
    
    console.log('Created plot traces:', traces.length);
    return traces;
  }, [clusterData, filteredPapers]);

  // Create full dataset scatter plot data for visualization dialog (ignores cluster selection)
  const fullPlotData = useMemo(() => {
    if (!clusterData || !clusterData.cluster_info) {
      console.log('No cluster data available for full plotting');
      return [];
    }
    
    const numClusters = clusterData.total_clusters || 0;
    const colors = generateClusterColors(numClusters);
    
    const traces: any[] = [];
    
    // Group ALL papers by cluster (not filtered by selection)
    const allPapers = clusterData.papers || [];
    const papersByCluster: Record<number, any[]> = {};
    allPapers.forEach(paper => {
      if (!papersByCluster[paper.cluster_id]) {
        papersByCluster[paper.cluster_id] = [];
      }
      papersByCluster[paper.cluster_id].push(paper);
    });
    
    // Create a trace for each cluster
    Object.entries(papersByCluster).forEach(([clusterId, papers]) => {
      const clusterIdNum = parseInt(clusterId);
      const clusterInfo = clusterData.cluster_info[clusterId];
      const clusterName = clusterInfo?.name ? clusterInfo.name : `Cluster ${clusterId}`;
      
      traces.push({
        x: papers.map(p => p.pca_x),
        y: papers.map(p => p.pca_y),
        mode: 'markers',
        type: 'scatter',
        name: `${clusterName} (${papers.length} papers)`,
        marker: {
          color: colors[clusterIdNum % colors.length],
          size: 8,
          opacity: 0.7,
        },
        text: papers.map(p => `${p.Title}<br>Authors: ${p.Author}`),
        hovertemplate: '%{text}<extra></extra>',
        customdata: papers,
      });
    });
    
    console.log('Created full plot traces:', traces.length);
    return traces;
  }, [clusterData]);

  // Create network data for D3 visualization
  const networkData = useMemo(() => {
    if (!clusterData) return { nodes: [], links: [] };

    console.log('Creating network data');
    
    const nodes = filteredPapers.map((paper, index) => ({
      id: paper.Key || `paper_${index}`,
      title: paper.Title,
      author: paper.Author,
      cluster: paper.cluster_id,
      x: paper.pca_x * 500 + 400, // Scale and center the coordinates
      y: paper.pca_y * 500 + 300,
      paper: paper,
    }));

    const links: any[] = [];
    
    // Create links between papers in the same cluster (but limit to avoid too many connections)
    const papersByCluster: Record<number, any[]> = {};
    nodes.forEach(node => {
      if (!papersByCluster[node.cluster]) {
        papersByCluster[node.cluster] = [];
      }
      papersByCluster[node.cluster].push(node);
    });

    // For each cluster, create connections between papers
    Object.values(papersByCluster).forEach(clusterPapers => {
      if (clusterPapers.length <= 1) return;
      
      // Create a more structured network - connect each paper to a few others in the same cluster
      clusterPapers.forEach((paper, index) => {
        // Connect to next 2-3 papers in cluster (circular)
        const maxConnections = Math.min(3, clusterPapers.length - 1);
        for (let i = 1; i <= maxConnections; i++) {
          const targetIndex = (index + i) % clusterPapers.length;
          const target = clusterPapers[targetIndex];
          
          links.push({
            source: paper.id,
            target: target.id,
            cluster: paper.cluster,
          });
        }
      });
    });

    console.log('Network data created:', { nodeCount: nodes.length, linkCount: links.length });
    return { nodes, links };
  }, [clusterData, filteredPapers]);

  // Create full network data for visualization dialog (ignores cluster selection)
  const fullNetworkData = useMemo(() => {
    if (!clusterData) return { nodes: [], links: [] };

    console.log('Creating full network data');
    
    const allPapers = clusterData.papers || [];
    const nodes = allPapers.map((paper, index) => ({
      id: paper.Key || `paper_${index}`,
      title: paper.Title,
      author: paper.Author,
      cluster: paper.cluster_id,
      x: paper.pca_x * 500 + 400, // Scale and center the coordinates
      y: paper.pca_y * 500 + 300,
      paper: paper,
    }));

    const links: any[] = [];
    
    // Create links between papers in the same cluster (but limit to avoid too many connections)
    const papersByCluster: Record<number, any[]> = {};
    nodes.forEach(node => {
      if (!papersByCluster[node.cluster]) {
        papersByCluster[node.cluster] = [];
      }
      papersByCluster[node.cluster].push(node);
    });

    // For each cluster, create connections between papers
    Object.values(papersByCluster).forEach(clusterPapers => {
      if (clusterPapers.length <= 1) return;

      // Connect each paper to a few others in its cluster (star pattern or limited connections)
      clusterPapers.forEach((paper, index) => {
        // Connect to next paper in cluster (circular)
        const targetIndex = (index + 1) % clusterPapers.length;
        const target = clusterPapers[targetIndex];
        
        links.push({
          source: paper.id,
          target: target.id,
          cluster: paper.cluster,
        });
      });
    });

    console.log('Full network data created:', { nodeCount: nodes.length, linkCount: links.length });
    return { nodes, links };
  }, [clusterData]);

  // D3 Network Visualization Effect
  useEffect(() => {
    // Use full network data when in visualization dialog, otherwise use filtered data
    const currentNetworkData = showVisualization ? fullNetworkData : networkData;
    
    if (viewMode !== 'network' || !networkRef.current || currentNetworkData.nodes.length === 0) {
      return;
    }

    console.log('Rendering network visualization');
    const svg = d3.select(networkRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const width = 800;
    const height = 600;
    const colors = generateClusterColors(clusterData?.total_clusters || 10);

    // Create main group for zooming and panning
    const g = svg.append('g');

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    // Apply zoom behavior to SVG
    svg.call(zoom);

    // Create simulation
    const simulation = d3.forceSimulation(currentNetworkData.nodes as any)
      .force('link', d3.forceLink(currentNetworkData.links).id((d: any) => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(15));

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(currentNetworkData.links)
      .enter().append('line')
      .attr('stroke', (d: any) => colors[d.cluster % colors.length])
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 1.5);

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(currentNetworkData.nodes)
      .enter().append('circle')
      .attr('r', 8)
      .attr('fill', (d: any) => colors[d.cluster % colors.length])
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add hover effects and click handlers
    node
      .on('mouseover', function(event: any, d: any) {
        // Highlight connected nodes and links
        const connectedLinks = currentNetworkData.links.filter((l: any) => 
          l.source.id === d.id || l.target.id === d.id
        );
        const connectedNodeIds = new Set();
        connectedLinks.forEach((l: any) => {
          connectedNodeIds.add(l.source.id);
          connectedNodeIds.add(l.target.id);
        });

        node.style('opacity', (n: any) => connectedNodeIds.has(n.id) ? 1 : 0.3);
        link.style('opacity', (l: any) => 
          l.source.id === d.id || l.target.id === d.id ? 0.8 : 0.1
        );

        // Show tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'network-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '10px')
          .style('border-radius', '5px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <strong>${clusterData?.cluster_info[d.cluster]?.name || `Cluster ${d.cluster}`}</strong><br/>
            <strong>${d.title}</strong><br/>
            <em>${d.author}</em>
          `);
        
        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function(_) {
        node.style('opacity', 1);
        link.style('opacity', 0.3);
        d3.selectAll('.network-tooltip').remove();
      })
      .on('click', function(_, d: any) {
        setSelectedPaper(d.paper);
        setPaperDetailsOpen(true);
      });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);
    });

    // Drag functions
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Add zoom controls
    const controls = svg.append('g')
      .attr('class', 'zoom-controls')
      .style('pointer-events', 'all');

    // Zoom in button
    const zoomInButton = controls.append('g')
      .attr('transform', 'translate(10, 50)')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(300).call(
          zoom.scaleBy as any, 1.5
        );
      });

    zoomInButton.append('rect')
      .attr('width', 30)
      .attr('height', 30)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#ccc')
      .attr('rx', 4);

    zoomInButton.append('text')
      .attr('x', 15)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('+');

    // Zoom out button
    const zoomOutButton = controls.append('g')
      .attr('transform', 'translate(10, 85)')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(300).call(
          zoom.scaleBy as any, 0.67
        );
      });

    zoomOutButton.append('rect')
      .attr('width', 30)
      .attr('height', 30)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#ccc')
      .attr('rx', 4);

    zoomOutButton.append('text')
      .attr('x', 15)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('−');

    // Reset zoom button
    const resetButton = controls.append('g')
      .attr('transform', 'translate(10, 120)')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(500).call(
          zoom.transform as any,
          d3.zoomIdentity
        );
      });

    resetButton.append('rect')
      .attr('width', 30)
      .attr('height', 30)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#ccc')
      .attr('rx', 4);

    resetButton.append('text')
      .attr('x', 15)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text('⌂');

    return () => {
      simulation.stop();
    };
  }, [viewMode, networkData, fullNetworkData, showVisualization, clusterData]);

  const handleViewModeChange = (_event: React.MouseEvent<HTMLElement>, newViewMode: 'scatter' | 'network' | null) => {
    if (newViewMode !== null) {
      setViewMode(newViewMode);
    }
  };

  const handlePlotClick = (event: any) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      const paperData = point.customdata;
      setSelectedPaper(paperData);
      setPaperDetailsOpen(true);
    }
  };

  // Generate markdown content for current cluster
  const generateClusterMarkdown = () => {
    if (!clusterData || selectedCluster === 'all') return '';

    const clusterInfo = clusterData.cluster_info[selectedCluster];
    const clusterPapers = filteredPapers.filter(paper => paper.cluster_id.toString() === selectedCluster);
    
    let markdown = `# ${clusterInfo?.name || `Cluster ${selectedCluster}`}\n\n`;
    
    // Add cluster description if available
    if (clusterInfo?.description) {
      markdown += `${clusterInfo.description}\n\n`;
    }
    
    // Add cluster statistics
    markdown += `**Papers in cluster:** ${clusterInfo?.size || clusterPapers.length}\n\n`;
    
    // Add top keywords
    if (clusterInfo?.top_keywords && clusterInfo.top_keywords.length > 0) {
      markdown += `**Key topics:** ${clusterInfo.top_keywords.slice(0, 8).map(([keyword, score]) => `${keyword} (${score.toFixed(2)})`).join(', ')}\n\n`;
    }
    
    markdown += `---\n\n`;
    
    // Add each paper as a separate section
    clusterPapers.forEach((paper, index) => {
      markdown += `## ${index + 1}. ${paper.Title}\n\n`;
      
      markdown += `**Authors:** ${paper.Author}\n\n`;
      
      if (paper['Publication Year']) {
        markdown += `**Year:** ${paper['Publication Year']}\n\n`;
      }
      
      if (paper.Venue) {
        markdown += `**Venue:** ${paper.Venue}\n\n`;
      }
      
      if (paper.DOI) {
        markdown += `**DOI:** ${paper.DOI}\n\n`;
      }
      
      if (paper.Url) {
        markdown += `**URL:** [${paper.Url}](${paper.Url})\n\n`;
      }
      
      if (paper.summary) {
        markdown += `**AI Summary:**\n${paper.summary}\n\n`;
      }
      
      if (paper.Abstract) {
        markdown += `**Abstract:**\n${paper.Abstract}\n\n`;
      }
      
      if (paper.keywords) {
        markdown += `**Keywords:** ${paper.keywords}\n\n`;
      }
      
      markdown += `---\n\n`;
    });
    
    return markdown;
  };

  // Download markdown file
  const handleExportCluster = () => {
    if (!clusterData || selectedCluster === 'all') {
      setError('Please select a specific cluster to export');
      return;
    }

    const markdown = generateClusterMarkdown();
    const clusterInfo = clusterData.cluster_info[selectedCluster];
    const clusterName = clusterInfo?.name || `Cluster_${selectedCluster}`;
    const filename = `${clusterName.replace(/[^a-zA-Z0-9]/g, '_')}_papers.md`;
    
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Cluster Visualization
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Skeleton variant="rectangular" height={600} />
          </Grid>
          <Grid item xs={12} md={4}>
            <Skeleton variant="rectangular" height={400} />
          </Grid>
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Cluster Visualization
        </Typography>
        <Alert severity="error">
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>

      {/* Show message when no clustering data is available */}
      {!clusterData && (!jobId && !externalData && savedResults.length > 0) && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Please select a clustering result from the dropdown in the left panel to visualize.
        </Alert>
      )}

      {!clusterData && (jobId || externalData) && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No clustering data available. Please run clustering analysis first.
        </Alert>
      )}

      {/* Dataset Information - only show when we have data */}
      {loadingClusterData ? (
        <MuiPaper sx={{ p: 1.5, mb: 2 }}>
          <Skeleton variant="text" width="40%" height={24} sx={{ mb: 1 }} />
          <Skeleton variant="text" width="80%" height={20} />
        </MuiPaper>
      ) : clusterData && clusterData.metadata && (
        <MuiPaper sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
            Dataset Information
          </Typography>
          <Grid container spacing={1} alignItems="center">
            <Grid item>
              <Typography variant="body2" color="text.secondary">
                <strong>{clusterData.total_papers?.toLocaleString() || 0}</strong> papers
              </Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">•</Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">
                <strong>{clusterData.total_clusters || 0}</strong> clusters
              </Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">•</Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">
                Method: <strong>{clusterData.metadata?.clustering_method || 'N/A'}</strong>
              </Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">•</Typography>
            </Grid>
            <Grid item>
              <Typography variant="body2" color="text.secondary">
                Features: <strong>{clusterData.metadata?.feature_extraction || 'N/A'}</strong>
              </Typography>
            </Grid>
          </Grid>
        </MuiPaper>
      )}

      {/* Show message when no clustering data is available */}
      {!clusterData && (!jobId && !externalData && savedResults.length > 0) && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Please select a clustering result from the dropdown in the left panel to visualize.
        </Alert>
      )}

      {!clusterData && (jobId || externalData) && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No clustering data available. Please run clustering analysis first.
        </Alert>
      )}

      {/* Main Layout: Cluster List (Left) + Paper List (Right) - ALWAYS SHOW */}
      <Grid container spacing={3}>
            {/* Left Side - Cluster List */}
            <Grid item xs={12} md={4}>
              <MuiPaper sx={{ p: 2, height: 'fit-content', position: 'sticky', top: 20 }}>
                {/* Clustering Result Selector */}
                {!jobId && !externalData && (
                  <Box sx={{ mb: 2 }}>
                    {loadingSavedResults ? (
                      <Skeleton variant="rectangular" height={56} />
                    ) : savedResults.length === 0 ? (
                      <Alert severity="info" sx={{ mb: 2 }}>
                        No saved clustering results found. Run clustering analysis to create visualizations.
                      </Alert>
                    ) : (
                      <FormControl fullWidth size="small">
                        <InputLabel>Select Clustering Result</InputLabel>
                        <Select
                          value={selectedResultId || ''}
                          onChange={(e) => {
                            const resultId = Number(e.target.value);
                            if (resultId) {
                              loadSavedClusteringResult(resultId);
                              setSelectedCluster('all'); // Reset cluster selection
                            }
                          }}
                          label="Select Clustering Result"
                          disabled={loadingClusterData}
                        >
                          {savedResults.map((result) => (
                            <MenuItem key={result.id} value={result.id}>
                              <Box>
                                <Typography variant="body2">
                                  {result.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {result.total_papers} papers, {result.total_clusters} clusters
                                  {result.silhouette_score && ` • Score: ${result.silhouette_score.toFixed(3)}`}
                                </Typography>
                              </Box>
                            </MenuItem>
                          ))}
                        </Select>
                        {loadingClusterData && (
                          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                            <Skeleton variant="circular" width={20} height={20} sx={{ mr: 1 }} />
                            <Typography variant="caption" color="text.secondary">
                              Loading clustering data...
                            </Typography>
                          </Box>
                        )}
                      </FormControl>
                    )}
                    
                    <Button
                      onClick={loadSavedResults}
                      size="small"
                      sx={{ mt: 1 }}
                      disabled={loadingSavedResults || loadingClusterData}
                      fullWidth
                      variant="outlined"
                    >
                      {loadingSavedResults ? 'Refreshing...' : 'Refresh Results'}
                    </Button>
                  </Box>
                )}
                
                <Typography variant="h6" gutterBottom>
                  Clusters ({clusterData && clusterData.cluster_info ? Object.keys(clusterData.cluster_info).length : 0})
                </Typography>
                
                {/* Controls */}
                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="contained"
                    onClick={() => setShowVisualization(true)}
                    startIcon={<ScatterPlotIcon />}
                    size="small"
                    fullWidth
                    sx={{ mb: 2 }}
                    disabled={!clusterData || loadingClusterData}
                  >
                    Show Visualization
                  </Button>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                    Total Papers: {clusterData ? filteredPapers.length : 0}
                  </Typography>
                </Box>

                {/* Cluster List */}
                {loadingClusterData ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Skeleton variant="rectangular" height={80} sx={{ mb: 1 }} />
                    <Skeleton variant="rectangular" height={80} sx={{ mb: 1 }} />
                    <Skeleton variant="rectangular" height={80} sx={{ mb: 1 }} />
                    <Typography variant="caption" color="text.secondary">
                      Loading clusters...
                    </Typography>
                  </Box>
                ) : clusterData && clusterData.cluster_info ? (
                  <Box sx={{ maxHeight: 600, overflowY: 'auto' }}>
                    {Object.entries(clusterData.cluster_info).map(([clusterId, info]) => {
                      const isSelected = selectedCluster === clusterId;
                      return (
                        <Box 
                          key={clusterId}
                          sx={{ 
                            p: 2, 
                            mb: 1,
                            cursor: 'pointer',
                            border: 1,
                            borderColor: isSelected ? 'primary.main' : 'grey.300',
                            borderRadius: 1,
                            backgroundColor: isSelected ? 'primary.50' : 'background.paper',
                            '&:hover': {
                              borderColor: 'primary.main',
                              backgroundColor: 'primary.50'
                            }
                          }}
                          onClick={() => setSelectedCluster(clusterId)}
                        >
                          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                            {info.name || `Cluster ${clusterId}`}
                          </Typography>
                          
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                            {info.size} papers
                          </Typography>
                          
                          {info.description && (
                            <Typography variant="caption" sx={{ display: 'block', mb: 1, fontStyle: 'italic' }}>
                              {info.description.length > 80 
                                ? `${info.description.slice(0, 80)}...` 
                                : info.description}
                            </Typography>
                          )}
                          
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {info.top_keywords?.slice(0, 2).map(([keyword], index) => (
                              <Chip
                                key={index}
                                label={keyword}
                                size="small"
                                variant="outlined"
                                sx={{ fontSize: '0.65rem', height: '18px' }}
                              />
                            )) || []}
                            {(info.top_keywords?.length || 0) > 2 && (
                              <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
                                +{(info.top_keywords?.length || 0) - 2}
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      );
                    })}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4, border: '1px dashed', borderColor: 'grey.300', borderRadius: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Select a clustering result to view clusters
                    </Typography>
                  </Box>
                )}
              </MuiPaper>
            </Grid>

            {/* Right Side - Paper List */}
            <Grid item xs={12} md={8}>
              <MuiPaper sx={{ p: 2, minHeight: '70vh' }}>
                {loadingClusterData ? (
                  <Box sx={{ p: 2 }}>
                    <Skeleton variant="text" width="60%" height={40} sx={{ mb: 2 }} />
                    <Skeleton variant="rectangular" height={100} sx={{ mb: 2 }} />
                    <Skeleton variant="rectangular" height={120} sx={{ mb: 2 }} />
                    <Skeleton variant="rectangular" height={120} sx={{ mb: 2 }} />
                    <Box sx={{ textAlign: 'center', mt: 4 }}>
                      <Typography variant="body2" color="text.secondary">
                        Loading papers...
                      </Typography>
                    </Box>
                  </Box>
                ) : !clusterData ? (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      No Dataset Selected
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Select a clustering result from the left panel to explore papers and clusters
                    </Typography>
                  </Box>
                ) : selectedCluster === 'all' ? (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Select a Cluster
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Choose a cluster from the left panel to view its papers
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    {/* Search Bar for Papers */}
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                        <TextField
                          label="Search papers in this cluster"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          size="small"
                          sx={{ flexGrow: 1 }}
                        />
                        <Button
                          variant="outlined"
                          startIcon={<DownloadIcon />}
                          onClick={handleExportCluster}
                          size="small"
                          sx={{ minWidth: 'auto', px: 2 }}
                        >
                          Export
                        </Button>
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'right' }}>
                        Showing {filteredPapers.filter(p => p.cluster_id.toString() === selectedCluster).length} papers
                      </Typography>
                    </Box>
                    
                    {/* Cluster Header */}
                    <Box sx={{ mb: 3, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="h5" gutterBottom>
                        {clusterData?.cluster_info?.[selectedCluster]?.name || `Cluster ${selectedCluster}`}
                      </Typography>
                      
                      {clusterData?.cluster_info?.[selectedCluster]?.description && (
                        <Typography variant="body1" sx={{ mb: 2, fontStyle: 'italic' }}>
                          {clusterData.cluster_info[selectedCluster].description}
                        </Typography>
                      )}
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                        <Typography variant="body2">
                          <strong>{clusterData?.cluster_info?.[selectedCluster]?.size || 0} papers</strong>
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          ({filteredPapers.filter(p => p.cluster_id.toString() === selectedCluster).length} shown)
                        </Typography>
                      </Box>

                      {/* Top Keywords */}
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {clusterData?.cluster_info?.[selectedCluster]?.top_keywords?.slice(0, 8).map(([keyword, score], index) => (
                          <Chip
                            key={index}
                            label={`${keyword} (${score.toFixed(2)})`}
                            size="small"
                            variant="outlined"
                          />
                        )) || []}
                      </Box>
                    </Box>

                    {/* Papers List */}
                    <Typography variant="h6" gutterBottom>
                      Papers in this Cluster
                    </Typography>
                    
                    <Box sx={{ maxHeight: '60vh', overflowY: 'auto', pr: 1 }}>
                      {filteredPapers
                        .filter(paper => paper.cluster_id.toString() === selectedCluster)
                        .map((paper, index) => (
                          <Box 
                            key={paper.Key || index}
                            sx={{ 
                              p: 2, 
                              mb: 2, 
                              border: '1px solid', 
                              borderColor: 'grey.200',
                              borderRadius: 1,
                              cursor: 'pointer',
                              '&:hover': {
                                backgroundColor: 'action.hover',
                                borderColor: 'primary.main',
                                boxShadow: 1
                              }
                            }}
                            onClick={() => {
                              setSelectedPaper(paper);
                              setPaperDetailsOpen(true);
                            }}
                          >
                            <Typography variant="h6" sx={{ mb: 1, fontWeight: 'bold' }}>
                              {paper.Title}
                            </Typography>
                            
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              <strong>Authors:</strong> {paper.Author}
                            </Typography>
                            
                            <Box sx={{ display: 'flex', gap: 2, mb: 1, flexWrap: 'wrap' }}>
                              {paper['Publication Year'] && (
                                <Typography variant="caption" color="text.secondary">
                                  <strong>Year:</strong> {paper['Publication Year']}
                                </Typography>
                              )}
                              
                              {paper.Venue && (
                                <Typography variant="caption" color="text.secondary">
                                  <strong>Venue:</strong> {paper.Venue}
                                </Typography>
                              )}

                              {paper.DOI && (
                                <Typography variant="caption" color="text.secondary">
                                  <strong>DOI:</strong> {paper.DOI}
                                </Typography>
                              )}

                              {paper.Url && (
                                <Typography variant="caption" color="text.secondary">
                                  <strong>URL:</strong> 
                                  <a 
                                    href={paper.Url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    style={{ 
                                      color: 'inherit', 
                                      textDecoration: 'underline',
                                      marginLeft: '4px'
                                    }}
                                  >
                                    {paper.Url.length > 40 ? `${paper.Url.slice(0, 40)}...` : paper.Url}
                                  </a>
                                </Typography>
                              )}
                            </Box>

                            {paper.summary && (
                              <Box sx={{ mb: 1 }}>
                                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                                  AI Summary:
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', lineHeight: 1.4 }}>
                                  {paper.summary.length > 200 
                                    ? `${paper.summary.slice(0, 200)}...` 
                                    : paper.summary}
                                </Typography>
                              </Box>
                            )}
                            
                            {paper.keywords && (
                              <Box sx={{ mb: 1 }}>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                  {paper.keywords.split(',').slice(0, 6).map((keyword: string, keywordIndex: number) => (
                                    <Chip 
                                      key={keywordIndex} 
                                      label={keyword.trim()} 
                                      size="small" 
                                      variant="outlined"
                                      sx={{ fontSize: '0.7rem', height: '22px' }}
                                    />
                                  ))}
                                  {paper.keywords.split(',').length > 6 && (
                                    <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
                                      +{paper.keywords.split(',').length - 6} more
                                    </Typography>
                                  )}
                                </Box>
                              </Box>
                            )}
                            
                            {paper.Abstract && (
                              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', lineHeight: 1.4 }}>
                                {paper.Abstract.length > 200 
                                  ? `${paper.Abstract.slice(0, 200)}...` 
                                  : paper.Abstract}
                              </Typography>
                            )}
                          </Box>
                        ))
                      }
                    </Box>
                  </Box>
                )}
              </MuiPaper>
            </Grid>
          </Grid>

        {/* Visualization Modal */}
        <Dialog
          open={showVisualization}
          onClose={() => setShowVisualization(false)}
          maxWidth="xl"
          fullWidth
          PaperProps={{
            sx: { height: '90vh' }
          }}
        >
          <DialogTitle>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h6">
                Cluster Visualization
              </Typography>
              
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={handleViewModeChange}
                aria-label="visualization mode"
                size="small"
              >
                <ToggleButton value="scatter" aria-label="scatter plot">
                  <ScatterPlotIcon sx={{ mr: 1 }} />
                  Scatter Plot
                </ToggleButton>
                <ToggleButton value="network" aria-label="network view">
                  <NetworkIcon sx={{ mr: 1 }} />
                  Network View
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          </DialogTitle>
          
          <DialogContent sx={{ p: 2, height: '100%' }}>
            {viewMode === 'scatter' ? (
              <Plot
                data={fullPlotData}
                layout={{
                  title: 'Paper Clusters (PCA Projection)',
                  xaxis: {
                    title: `PC1 (${clusterData?.metadata?.pca_explained_variance?.[0] ? (clusterData.metadata.pca_explained_variance[0] * 100).toFixed(1) : '0'}% variance)`,
                  },
                  yaxis: {
                    title: `PC2 (${clusterData?.metadata?.pca_explained_variance?.[1] ? (clusterData.metadata.pca_explained_variance[1] * 100).toFixed(1) : '0'}% variance)`,
                  },
                  hovermode: 'closest',
                  height: 600,
                  margin: { t: 50, r: 20, b: 50, l: 60 },
                  showlegend: true,
                }}
                config={{
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                  responsive: true,
                }}
                onClick={handlePlotClick}
                onError={(error) => {
                  console.error('Plotly error:', error);
                  setError('Visualization error: ' + error.message);
                }}
                style={{ width: '100%', height: '600px' }}
              />
            ) : (
              <Box sx={{ position: 'relative', width: '100%', height: '600px', overflow: 'hidden' }}>
                <Typography variant="h6" sx={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
                  Paper Network (Clustered Papers Connected)
                </Typography>
                
                {/* Network Legend */}
                <Box sx={{ 
                  position: 'absolute', 
                  top: 10, 
                  right: 10, 
                  zIndex: 10,
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #e0e0e0'
                }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block' }}>
                    Legend
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                    <Box sx={{ 
                      width: 12, 
                      height: 12, 
                      borderRadius: '50%', 
                      background: '#1976d2', 
                      mr: 1 
                    }} />
                    <Typography variant="caption">Papers (nodes)</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box sx={{ 
                      width: 16, 
                      height: 2, 
                      background: '#666', 
                      mr: 1 
                    }} />
                    <Typography variant="caption">Cluster connections</Typography>
                  </Box>
                </Box>

                <svg
                  ref={networkRef}
                  width="100%"
                  height="600"
                  style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}
                />
                <Typography 
                  variant="caption" 
                  sx={{ 
                    position: 'absolute', 
                    bottom: 10, 
                    left: 10, 
                    color: 'text.secondary',
                    background: 'rgba(255, 255, 255, 0.8)',
                    padding: '4px 8px',
                    borderRadius: '4px'
                  }}
                >
                  Mouse wheel to zoom • Drag to pan • Drag nodes to rearrange • Hover to highlight • Click for details
                </Typography>
              </Box>
            )}
          </DialogContent>
          
          <DialogActions>
            <Button onClick={() => setShowVisualization(false)}>
              Close
            </Button>
          </DialogActions>
        </Dialog>

        {/* Keep for backwards compatibility - hide by default */}
        <Grid container spacing={3} sx={{ display: 'none' }}>
        {/* Main Visualization */}
        <Grid item xs={12} md={8}>
          <MuiPaper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap', alignItems: 'center' }}>
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={handleViewModeChange}
                aria-label="visualization mode"
                size="small"
              >
                <ToggleButton value="scatter" aria-label="scatter plot">
                  <ScatterPlotIcon sx={{ mr: 1 }} />
                  Scatter Plot
                </ToggleButton>
                <ToggleButton value="network" aria-label="network view">
                  <NetworkIcon sx={{ mr: 1 }} />
                  Network View
                </ToggleButton>
              </ToggleButtonGroup>

              <TextField
                label="Search papers"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                size="small"
                sx={{ minWidth: 200 }}
              />
              
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>Cluster</InputLabel>
                <Select
                  value={selectedCluster}
                  onChange={(e) => setSelectedCluster(e.target.value)}
                  label="Cluster"
                >
                  <MenuItem value="all">All Clusters</MenuItem>
                  {clusterData && clusterData.cluster_info && Object.keys(clusterData.cluster_info).map(clusterId => {
                    const clusterInfo = clusterData.cluster_info[clusterId];
                    const displayName = clusterInfo?.name 
                      ? `${clusterInfo.name} (${clusterInfo.size})`
                      : `Cluster ${clusterId} (${clusterInfo?.size})`;
                    return (
                      <MenuItem key={clusterId} value={clusterId}>
                        {displayName}
                      </MenuItem>
                    );
                  })}
                </Select>
              </FormControl>

              <Typography variant="body2" sx={{ alignSelf: 'center', ml: 'auto' }}>
                Showing {clusterData?.total_papers || clusterData?.papers?.length || 0} papers
              </Typography>
            </Box>

            {viewMode === 'scatter' ? (
              <Plot
                data={fullPlotData}
                layout={{
                  title: 'Paper Clusters (PCA Projection)',
                  xaxis: {
                    title: `PC1 (${clusterData?.metadata?.pca_explained_variance?.[0] ? (clusterData.metadata.pca_explained_variance[0] * 100).toFixed(1) : '0'}% variance)`,
                  },
                  yaxis: {
                    title: `PC2 (${clusterData?.metadata?.pca_explained_variance?.[1] ? (clusterData.metadata.pca_explained_variance[1] * 100).toFixed(1) : '0'}% variance)`,
                  },
                  hovermode: 'closest',
                  height: 600,
                  margin: { t: 50, r: 20, b: 50, l: 60 },
                  showlegend: true,
                }}
                config={{
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                  responsive: true,
                }}
                onClick={handlePlotClick}
                onError={(error) => {
                  console.error('Plotly error:', error);
                  setError('Visualization error: ' + error.message);
                }}
                style={{ width: '100%', height: '600px' }}
              />
            ) : (
              <Box sx={{ position: 'relative', width: '100%', height: '600px', overflow: 'hidden' }}>
                <Typography variant="h6" sx={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
                  Paper Network (Clustered Papers Connected)
                </Typography>
                
                {/* Network Legend */}
                <Box sx={{ 
                  position: 'absolute', 
                  top: 10, 
                  right: 10, 
                  zIndex: 10,
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #e0e0e0'
                }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block' }}>
                    Legend
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                    <Box sx={{ 
                      width: 12, 
                      height: 12, 
                      borderRadius: '50%', 
                      background: '#1976d2', 
                      mr: 1 
                    }} />
                    <Typography variant="caption">Papers (nodes)</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box sx={{ 
                      width: 16, 
                      height: 2, 
                      background: '#666', 
                      mr: 1 
                    }} />
                    <Typography variant="caption">Cluster connections</Typography>
                  </Box>
                </Box>

                <svg
                  ref={networkRef}
                  width="100%"
                  height="600"
                  style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}
                />
                <Typography 
                  variant="caption" 
                  sx={{ 
                    position: 'absolute', 
                    bottom: 10, 
                    left: 10, 
                    color: 'text.secondary',
                    background: 'rgba(255, 255, 255, 0.8)',
                    padding: '4px 8px',
                    borderRadius: '4px'
                  }}
                >
                  Mouse wheel to zoom • Drag to pan • Drag nodes to rearrange • Hover to highlight • Click for details
                </Typography>
              </Box>
            )}
          </MuiPaper>
        </Grid>

        {/* Cluster Information */}
        <Grid item xs={12} md={4} sx={{ display: 'none' }}>
          <MuiPaper sx={{ p: 2, maxHeight: 600, overflow: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Cluster Details
            </Typography>

            {selectedCluster === 'all' ? (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Select a specific cluster to view details
              </Typography>
            ) : (
              clusterData && clusterData.cluster_info && clusterData.cluster_info[selectedCluster] && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    {clusterData.cluster_info[selectedCluster]?.name || `Cluster ${selectedCluster}`}
                  </Typography>
                  
                  {clusterData.cluster_info[selectedCluster]?.description && (
                    <Typography variant="body2" color="text.secondary" gutterBottom sx={{ fontStyle: 'italic', mb: 2 }}>
                      {clusterData.cluster_info[selectedCluster].description}
                    </Typography>
                  )}
                  
                  <Typography variant="body2" gutterBottom>
                    <strong>Size:</strong> {clusterData.cluster_info[selectedCluster]?.size} papers
                  </Typography>

                  <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                    Top Keywords:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                    {clusterData.cluster_info[selectedCluster]?.top_keywords?.slice(0, 8).map(([keyword, score], index) => (
                      <Chip
                        key={index}
                        label={`${keyword} (${score.toFixed(2)})`}
                        size="small"
                        variant="outlined"
                      />
                    )) || []}
                  </Box>

                  <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                    Sample Papers:
                  </Typography>
                  <List dense>
                    {clusterData.cluster_info[selectedCluster]?.sample_titles?.slice(0, 5).map((title, index) => (
                      <ListItem key={index} sx={{ pl: 0 }}>
                        <ListItemText 
                          primary={title}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    )) || []}
                  </List>
                </Box>
              )
            )}

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" gutterBottom>
              All Clusters Summary:
            </Typography>
            <List dense>
              {clusterData && clusterData.cluster_info && Object.entries(clusterData.cluster_info).map(([clusterId, info]) => (
                <ListItem 
                  key={clusterId} 
                  onClick={() => setSelectedCluster(clusterId)}
                  sx={{ 
                    pl: 0,
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' },
                    backgroundColor: selectedCluster === clusterId ? 'action.selected' : 'transparent'
                  }}
                >
                  <ListItemText
                    primary={info?.name || `Cluster ${clusterId}`}
                    secondary={`${info?.size} papers${info?.description ? ` • ${info.description.slice(0, 60)}${info.description.length > 60 ? '...' : ''}` : ''}`}
                  />
                </ListItem>
              ))}
            </List>
          </MuiPaper>
        </Grid>
      </Grid>

      {/* Paper Details Dialog */}
      <Dialog
        open={paperDetailsOpen}
        onClose={() => setPaperDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedPaper && (
          <>
            <DialogTitle>
              <Typography variant="h6">
                {selectedPaper.Title}
              </Typography>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Authors:
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    {selectedPaper.Author}
                  </Typography>
                </Grid>

                {selectedPaper.Abstract && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                      Abstract:
                    </Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedPaper.Abstract}
                    </Typography>
                  </Grid>
                )}

                {selectedPaper.summary && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                      AI Summary:
                    </Typography>
                    <Typography variant="body2" gutterBottom sx={{ fontStyle: 'italic' }}>
                      {selectedPaper.summary}
                    </Typography>
                  </Grid>
                )}

                {selectedPaper.Url && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                      URL:
                    </Typography>
                    <Typography variant="body2" gutterBottom>
                      <a 
                        href={selectedPaper.Url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={{ 
                          color: 'inherit', 
                          textDecoration: 'underline'
                        }}
                      >
                        {selectedPaper.Url}
                      </a>
                    </Typography>
                  </Grid>
                )}

                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Cluster:
                  </Typography>
                  <Chip 
                    label={clusterData?.cluster_info[selectedPaper.cluster_id]?.name || `Cluster ${selectedPaper.cluster_id}`}
                    color="primary" 
                    size="small"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Year:
                  </Typography>
                  <Typography variant="body2">
                    {selectedPaper['Publication Year'] || 'N/A'}
                  </Typography>
                </Grid>

                {selectedPaper.keywords && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" gutterBottom>
                      Keywords:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selectedPaper.keywords.split(',').map((keyword: string, index: number) => (
                        <Chip 
                          key={index} 
                          label={keyword.trim()} 
                          size="small" 
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </Grid>
                )}
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setPaperDetailsOpen(false)}>
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default ClusterVisualization;
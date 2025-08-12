import React, { useState } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CssBaseline,
  Container,
  IconButton,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Upload as UploadIcon,
  List as ListIcon,
  ScatterPlot as ClusterIcon,
  Visibility as VisualizeIcon,
  Settings as SettingsIcon,
  AutoAwesome as AIIcon,
} from '@mui/icons-material';
import { createTheme, ThemeProvider } from '@mui/material/styles';

// Import components
import Dashboard from './components/Dashboard';
import FileUpload from './components/FileUpload';
import PaperList from './components/PaperList';
import ClusteringInterface from './components/ClusteringInterface';
import ClusterVisualization from './components/ClusterVisualization';
import SettingsPage from './components/SettingsPage';
import AIProcessingPage from './components/AIProcessingPage';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const drawerWidth = 240;

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  component: React.ReactNode;
}

const App: React.FC = () => {
  const [selectedPage, setSelectedPage] = useState('dashboard');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);
  const [clusteringJobId, setClusteringJobId] = useState<string | null>(null);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handlePageChange = (pageId: string) => {
    setSelectedPage(pageId);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const handleUploadComplete = (jobId: string) => {
    setUploadJobId(jobId);
    // Optionally switch to papers list or show notification
  };

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error);
    // You could show a snackbar or other notification here
  };

  const handleClusteringComplete = (jobId: string) => {
    setClusteringJobId(jobId);
    // Switch to visualization page when clustering is complete
    setSelectedPage('visualization');
  };

  const handleClusteringError = (error: string) => {
    console.error('Clustering error:', error);
    // You could show a snackbar or other notification here
  };

  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <DashboardIcon />,
      component: <Dashboard />,
    },
    {
      id: 'upload',
      label: 'Upload Papers',
      icon: <UploadIcon />,
      component: (
        <Box>
          <Typography variant="h4" gutterBottom>
            Upload Papers
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Upload CSV files containing academic papers. Supported formats include Zotero exports
            and other structured paper data.
          </Typography>
          <FileUpload 
            onUploadComplete={handleUploadComplete}
            onError={handleUploadError}
          />
        </Box>
      ),
    },
    {
      id: 'papers',
      label: 'Paper List',
      icon: <ListIcon />,
      component: <PaperList />,
    },
    {
      id: 'clustering',
      label: 'Clustering',
      icon: <ClusterIcon />,
      component: (
        <ClusteringInterface 
          onClusteringComplete={handleClusteringComplete}
          onClusteringError={handleClusteringError}
        />
      ),
    },
    {
      id: 'visualization',
      label: 'Visualization',
      icon: <VisualizeIcon />,
      component: <ClusterVisualization jobId={clusteringJobId || undefined} />,
    },
    {
      id: 'ai-processing',
      label: 'AI Processing',
      icon: <AIIcon />,
      component: <AIProcessingPage />,
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <SettingsIcon />,
      component: <SettingsPage />,
    },
  ];

  const currentPage = navigationItems.find(item => item.id === selectedPage);

  const drawer = (
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          Paper Labeler
        </Typography>
      </Toolbar>
      <List>
        {navigationItems.map((item) => (
          <ListItem
            button
            key={item.id}
            selected={selectedPage === item.id}
            onClick={() => handlePageChange(item.id)}
          >
            <ListItemIcon>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        
        {/* App Bar */}
        <AppBar
          position="fixed"
          sx={{
            width: { md: `calc(100% - ${drawerWidth}px)` },
            ml: { md: `${drawerWidth}px` },
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { md: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div">
              {currentPage?.label || 'Paper Labeler'}
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Navigation Drawer */}
        <Box
          component="nav"
          sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        >
          {/* Mobile drawer */}
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true, // Better open performance on mobile.
            }}
            sx={{
              display: { xs: 'block', md: 'none' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
          >
            {drawer}
          </Drawer>
          
          {/* Desktop drawer */}
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', md: 'block' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
            open
          >
            {drawer}
          </Drawer>
        </Box>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { md: `calc(100% - ${drawerWidth}px)` },
          }}
        >
          <Toolbar />
          <Container maxWidth="xl">
            {currentPage?.component || (
              <Typography variant="h4">
                Page not found
              </Typography>
            )}
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;
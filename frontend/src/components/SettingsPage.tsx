import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper as MuiPaper,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Divider,
  IconButton,
  InputAdornment,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Visibility,
  VisibilityOff,
  Science as TestIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import { api, Settings, SettingsUpdate } from '../services/api';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testingEmbedding, setTestingEmbedding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Form state - OpenAI
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [model, setModel] = useState('');
  const [enabled, setEnabled] = useState(false);
  const [autoSummary, setAutoSummary] = useState(false);
  const [autoKeywords, setAutoKeywords] = useState(false);
  
  // Form state - Embedding
  const [embeddingApiKey, setEmbeddingApiKey] = useState('');
  const [embeddingBaseUrl, setEmbeddingBaseUrl] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('');
  const [embeddingEnabled, setEmbeddingEnabled] = useState(false);
  
  // UI state
  const [showApiKey, setShowApiKey] = useState(false);
  const [showEmbeddingApiKey, setShowEmbeddingApiKey] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [embeddingTestDialogOpen, setEmbeddingTestDialogOpen] = useState(false);
  const [embeddingTestResult, setEmbeddingTestResult] = useState<any>(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const settingsData = await api.getSettings();
      setSettings(settingsData);
      
      // Populate OpenAI form
      setBaseUrl(settingsData.openai.base_url);
      setModel(settingsData.openai.model);
      setEnabled(settingsData.openai.enabled);
      setAutoSummary(settingsData.processing.auto_generate_summary);
      setAutoKeywords(settingsData.processing.auto_generate_keywords);
      
      // Populate Embedding form
      if (settingsData.embedding) {
        setEmbeddingBaseUrl(settingsData.embedding.base_url);
        setEmbeddingModel(settingsData.embedding.model);
        setEmbeddingEnabled(settingsData.embedding.enabled);
      }
      
    } catch (err: any) {
      setError(err.message || 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(null);
      
      const updateData: SettingsUpdate = {
        openai_base_url: baseUrl,
        openai_model: model,
        openai_enabled: enabled,
        auto_generate_summary: autoSummary,
        auto_generate_keywords: autoKeywords,
        embedding_base_url: embeddingBaseUrl,
        embedding_model: embeddingModel,
        embedding_enabled: embeddingEnabled,
      };
      
      // Only include API keys if they were changed
      if (apiKey.trim()) {
        updateData.openai_api_key = apiKey.trim();
      }
      if (embeddingApiKey.trim()) {
        updateData.embedding_api_key = embeddingApiKey.trim();
      }
      
      await api.updateSettings(updateData);
      setSuccess('Settings saved successfully');
      
      // Clear API key fields after saving
      setApiKey('');
      setEmbeddingApiKey('');
      
      // Reload settings to get updated masked key
      await loadSettings();
      
    } catch (err: any) {
      setError(err.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const testConnection = async () => {
    try {
      setTesting(true);
      setTestResult(null);
      
      const result = await api.testOpenAIConnection();
      setTestResult(result);
      setTestDialogOpen(true);
      
    } catch (err: any) {
      setTestResult({
        success: false,
        error: err.message || 'Connection test failed'
      });
      setTestDialogOpen(true);
    } finally {
      setTesting(false);
    }
  };

  const testEmbeddingConnection = async () => {
    try {
      setTestingEmbedding(true);
      setEmbeddingTestResult(null);
      
      const result = await api.testEmbeddingConnection();
      setEmbeddingTestResult(result);
      setEmbeddingTestDialogOpen(true);
      
    } catch (err: any) {
      setEmbeddingTestResult({
        success: false,
        error: err.message || 'Embedding connection test failed'
      });
      setEmbeddingTestDialogOpen(true);
    } finally {
      setTestingEmbedding(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SettingsIcon />
        Settings
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* OpenAI Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title="OpenAI Configuration"
              subheader="Configure AI-powered summary and keyword generation"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  fullWidth
                  label="API Key"
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={settings?.openai.api_key_masked ? 
                    `Current: ${settings.openai.api_key_masked}` : 
                    'Enter your OpenAI API key'
                  }
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowApiKey(!showApiKey)}
                          edge="end"
                        >
                          {showApiKey ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  helperText="Leave empty to keep current API key"
                />

                <TextField
                  fullWidth
                  label="Base URL"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder="https://api.openai.com/v1"
                  helperText="OpenAI API base URL (for custom endpoints)"
                />

                <TextField
                  fullWidth
                  label="Model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="gpt-4o-mini"
                  helperText="OpenAI model to use for processing"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={enabled}
                      onChange={(e) => setEnabled(e.target.checked)}
                    />
                  }
                  label="Enable OpenAI Processing"
                />

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    onClick={testConnection}
                    disabled={testing || !baseUrl || !model}
                    startIcon={testing ? <CircularProgress size={16} /> : <TestIcon />}
                  >
                    {testing ? 'Testing...' : 'Test Connection'}
                  </Button>
                  
                  <Button
                    variant="contained"
                    onClick={saveSettings}
                    disabled={saving}
                    startIcon={saving ? <CircularProgress size={16} /> : <SaveIcon />}
                  >
                    {saving ? 'Saving...' : 'Save Settings'}
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Embedding Model Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title="Embedding Model Configuration"
              subheader="Configure embedding model for advanced clustering"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  fullWidth
                  label="API Key"
                  type={showEmbeddingApiKey ? 'text' : 'password'}
                  value={embeddingApiKey}
                  onChange={(e) => setEmbeddingApiKey(e.target.value)}
                  placeholder={settings?.embedding?.api_key_masked ? 
                    `Current: ${settings.embedding.api_key_masked}` : 
                    'Enter embedding model API key (leave empty to use OpenAI)'
                  }
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowEmbeddingApiKey(!showEmbeddingApiKey)}
                          edge="end"
                        >
                          {showEmbeddingApiKey ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  helperText="Leave empty to use the same API key as OpenAI"
                />

                <TextField
                  fullWidth
                  label="Base URL"
                  value={embeddingBaseUrl}
                  onChange={(e) => setEmbeddingBaseUrl(e.target.value)}
                  placeholder="https://api.openai.com/v1"
                  helperText="Leave empty to use the same base URL as OpenAI"
                />

                <TextField
                  fullWidth
                  label="Model"
                  value={embeddingModel}
                  onChange={(e) => setEmbeddingModel(e.target.value)}
                  placeholder="text-embedding-ada-002"
                  helperText="Embedding model for generating text representations"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={embeddingEnabled}
                      onChange={(e) => setEmbeddingEnabled(e.target.checked)}
                    />
                  }
                  label="Enable Embedding Model"
                />

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    onClick={testEmbeddingConnection}
                    disabled={testingEmbedding || !embeddingModel}
                    startIcon={testingEmbedding ? <CircularProgress size={16} /> : <TestIcon />}
                  >
                    {testingEmbedding ? 'Testing...' : 'Test Connection'}
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Processing Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title="Processing Settings"
              subheader="Configure automatic AI processing behavior"
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={autoSummary}
                      onChange={(e) => setAutoSummary(e.target.checked)}
                      disabled={!enabled}
                    />
                  }
                  label="Auto-generate Summaries"
                />
                <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mt: -1 }}>
                  Automatically generate summaries for uploaded papers
                </Typography>

                <FormControlLabel
                  control={
                    <Switch
                      checked={autoKeywords}
                      onChange={(e) => setAutoKeywords(e.target.checked)}
                      disabled={!enabled}
                    />
                  }
                  label="Auto-generate Keywords"
                />
                <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mt: -1 }}>
                  Automatically extract keywords for uploaded papers
                </Typography>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" gutterBottom>
                  Current Status:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip 
                    label={enabled ? 'OpenAI Enabled' : 'OpenAI Disabled'} 
                    color={enabled ? 'success' : 'default'} 
                    size="small"
                  />
                  <Chip 
                    label={autoSummary ? 'Auto Summary ON' : 'Auto Summary OFF'} 
                    color={autoSummary && enabled ? 'primary' : 'default'} 
                    size="small"
                  />
                  <Chip 
                    label={autoKeywords ? 'Auto Keywords ON' : 'Auto Keywords OFF'} 
                    color={autoKeywords && enabled ? 'primary' : 'default'} 
                    size="small"
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Information */}
        <Grid item xs={12}>
          <MuiPaper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Information
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              The OpenAI integration allows automatic generation of summaries and keywords for your academic papers. 
              This helps in better organization and searchability of your research collection.
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              <strong>Summary Generation:</strong> Creates concise summaries that capture the main research problem and solution approach.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Keyword Extraction:</strong> Identifies 7-10 relevant keywords that represent the core concepts and themes of each paper.
            </Typography>
          </MuiPaper>
        </Grid>
      </Grid>

      {/* Test Result Dialog */}
      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Connection Test Result
        </DialogTitle>
        <DialogContent>
          {testResult && (
            <Alert severity={testResult.success ? 'success' : 'error'}>
              {testResult.success ? (
                <Box>
                  <Typography variant="body2">
                    ✅ Connection successful!
                  </Typography>
                  {testResult.model && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Model: {testResult.model}
                    </Typography>
                  )}
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2">
                    ❌ Connection failed
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Error: {testResult.error}
                  </Typography>
                </Box>
              )}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Embedding Test Result Dialog */}
      <Dialog open={embeddingTestDialogOpen} onClose={() => setEmbeddingTestDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Embedding Model Connection Test Result
        </DialogTitle>
        <DialogContent>
          {embeddingTestResult && (
            <Alert severity={embeddingTestResult.success ? 'success' : 'error'}>
              {embeddingTestResult.success ? (
                <Box>
                  <Typography variant="body2">
                    ✅ Embedding connection successful!
                  </Typography>
                  {embeddingTestResult.model && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Model: {embeddingTestResult.model}
                    </Typography>
                  )}
                  {embeddingTestResult.embedding_dimensions && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Embedding Dimensions: {embeddingTestResult.embedding_dimensions}
                    </Typography>
                  )}
                  {embeddingTestResult.fallback_to_openai && (
                    <Typography variant="body2" sx={{ mt: 1, color: 'info.main' }}>
                      ℹ️ Using OpenAI configuration (embedding config not set)
                    </Typography>
                  )}
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2">
                    ❌ Embedding connection failed
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Error: {embeddingTestResult.error}
                  </Typography>
                </Box>
              )}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEmbeddingTestDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SettingsPage;
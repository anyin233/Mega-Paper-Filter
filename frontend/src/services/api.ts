import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface Dataset {
  id: number;
  name: string;
  description: string;
  total_papers: number;
  created_at: string;
}

export interface Paper {
  id: number;
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  summary: string;
  keywords: string[];
  url: string;
  doi: string;
  publication_year?: number;
  venue: string;
  source_dataset: string;
  created_at: string;
}

export interface PaginatedPapersResponse {
  papers: Paper[];
  total: number;
  page: number;
  page_size: number;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  result?: any;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface ClusteringConfig {
  dataset_name?: string;
  max_features?: number;
  max_k?: number;
  min_papers?: number;
  clustering_method?: string;
  llm_model?: string;
  custom_model_name?: string;
  max_papers_llm?: number;
  // New embedding-specific parameters
  embedding_model?: string;
  embedding_batch_size?: number;
  embedding_clustering_algorithm?: string;
  dbscan_eps?: number;
  dbscan_min_samples?: number;
  agglomerative_linkage?: string;
}

export interface SavedClusteringResult {
  id: number;
  job_id: string;
  name: string;
  description: string;
  dataset_filter?: string;
  total_papers: number;
  total_clusters: number;
  clustering_method: string;
  feature_extraction: string;
  silhouette_score?: number;
  created_at: string;
}

export interface Statistics {
  total_papers: number;
  total_datasets: number;
  papers_with_summary: number;
  papers_with_keywords: number;
  datasets_breakdown: Array<{
    source_dataset: string;
    count: number;
  }>;
}

export interface Settings {
  openai: {
    base_url: string;
    model: string;
    enabled: boolean;
    api_key_masked: string;
  };
  embedding: {
    base_url: string;
    model: string;
    enabled: boolean;
    api_key_masked: string;
  };
  processing: {
    auto_generate_summary: boolean;
    auto_generate_keywords: boolean;
    batch_size: number;
    concurrent_requests: number;
  };
  ui: {
    theme: string;
    papers_per_page: number;
    auto_refresh: boolean;
  };
  database: {
    backup_enabled: boolean;
    backup_interval_hours: number;
    max_backups: number;
  };
}

export interface SettingsUpdate {
  openai_api_key?: string;
  openai_base_url?: string;
  openai_model?: string;
  openai_enabled?: boolean;
  auto_generate_summary?: boolean;
  auto_generate_keywords?: boolean;
  // New embedding model settings
  embedding_api_key?: string;
  embedding_base_url?: string;
  embedding_model?: string;
  embedding_enabled?: boolean;
}

export interface UnprocessedPaper {
  id: number;
  paper_id: string;
  title: string;
  source_dataset: string;
  has_summary: boolean;
  has_keywords: boolean;
  created_at: string;
}

export interface AIProcessRequest {
  paper_ids: string[];
  generate_summary: boolean;
  generate_keywords: boolean;
  overwrite_existing: boolean;
}

class PaperLabelerAPI {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await axios.get(`${this.baseURL}/api/health`);
    return response.data;
  }

  // Dataset management
  async getDatasets(): Promise<Dataset[]> {
    const response = await axios.get(`${this.baseURL}/api/datasets`);
    return response.data;
  }

  async createDataset(name: string, description: string = ''): Promise<Dataset> {
    const response = await axios.post(`${this.baseURL}/api/datasets`, {
      name,
      description,
    });
    return response.data;
  }

  async mergeDatasets(
    sourceDataset: string,
    targetDataset: string,
    deleteSource: boolean = true
  ): Promise<{ message: string; statistics: any }> {
    const response = await axios.post(`${this.baseURL}/api/datasets/merge`, {
      source_dataset: sourceDataset,
      target_dataset: targetDataset,
      delete_source: deleteSource,
    });
    return response.data;
  }

  // Paper management
  async getPapers(params: {
    dataset?: string;
    search?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<PaginatedPapersResponse> {
    const response = await axios.get(`${this.baseURL}/api/papers`, { params });
    return response.data;
  }

  async uploadPapers(
    file: File,
    datasetName: string,
    description: string = '',
    uploadToExisting: boolean = false
  ): Promise<{ job_id: string; message: string; status: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams({
      dataset_name: datasetName,
      description: description,
      upload_to_existing: uploadToExisting.toString()
    });
    
    const response = await axios.post(
      `${this.baseURL}/api/papers/upload?${params.toString()}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  // Clustering
  async runClustering(config: ClusteringConfig): Promise<{ job_id: string; message: string; status: string }> {
    const response = await axios.post(`${this.baseURL}/api/clustering/run`, config);
    return response.data;
  }

  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await axios.get(`${this.baseURL}/api/clustering/status/${jobId}`);
    return response.data;
  }

  async getActiveJobs(): Promise<{
    jobs: Array<{
      job_id: string;
      status: string;
      progress: number;
      message: string;
      created_at: string;
      updated_at: string;
      has_result: boolean;
      has_error: boolean;
    }>;
    active_count: number;
    total_count: number;
  }> {
    const response = await axios.get(`${this.baseURL}/api/clustering/active-jobs`);
    return response.data;
  }

  async getClusteringResults(jobId: string): Promise<any> {
    const response = await axios.get(`${this.baseURL}/api/clustering/results/${jobId}`);
    return response.data;
  }

  async getSavedClusteringResults(): Promise<SavedClusteringResult[]> {
    const response = await axios.get(`${this.baseURL}/api/clustering/saved-results`);
    return response.data;
  }

  async getSavedClusteringResult(resultId: number): Promise<any> {
    const response = await axios.get(`${this.baseURL}/api/clustering/saved-results/${resultId}`);
    return response.data;
  }

  async deleteSavedClusteringResult(resultId: number): Promise<{ message: string }> {
    const response = await axios.delete(`${this.baseURL}/api/clustering/saved-results/${resultId}`);
    return response.data;
  }

  async getClusteringResultsByDataset(datasetName: string): Promise<SavedClusteringResult[]> {
    const response = await axios.get(`${this.baseURL}/api/clustering/saved-results/by-dataset/${encodeURIComponent(datasetName)}`);
    return response.data;
  }

  // Statistics
  async getStatistics(): Promise<Statistics> {
    const response = await axios.get(`${this.baseURL}/api/statistics`);
    return response.data;
  }

  // Settings management
  async getSettings(): Promise<Settings> {
    const response = await axios.get(`${this.baseURL}/api/settings`);
    return response.data;
  }

  async updateSettings(settings: SettingsUpdate): Promise<{ message: string }> {
    const response = await axios.post(`${this.baseURL}/api/settings`, settings);
    return response.data;
  }

  async testOpenAIConnection(): Promise<{ success: boolean; message?: string; model?: string }> {
    const response = await axios.post(`${this.baseURL}/api/settings/test-openai`);
    return response.data;
  }

  async testEmbeddingConnection(): Promise<{ 
    success: boolean; 
    message?: string; 
    model?: string; 
    embedding_dimensions?: number;
    fallback_to_openai?: boolean;
  }> {
    const response = await axios.post(`${this.baseURL}/api/settings/test-embedding`);
    return response.data;
  }

  // AI Processing
  async processWithAI(request: AIProcessRequest): Promise<{ job_id: string; message: string; status: string }> {
    const response = await axios.post(`${this.baseURL}/api/ai/process`, request);
    return response.data;
  }

  async getUnprocessedPapers(params: {
    dataset?: string;
    missing_summary?: boolean;
    missing_keywords?: boolean;
    limit?: number;
  } = {}): Promise<UnprocessedPaper[]> {
    const response = await axios.get(`${this.baseURL}/api/papers/unprocessed`, { params });
    return response.data;
  }

  // WebSocket connection for real-time updates
  createWebSocket(onMessage: (data: any) => void, onError?: (error: Event) => void): WebSocket {
    const wsURL = this.baseURL.replace(/^http/, 'ws') + '/ws/progress';
    const ws = new WebSocket(wsURL);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };

    return ws;
  }
}

export const api = new PaperLabelerAPI();
export default api;
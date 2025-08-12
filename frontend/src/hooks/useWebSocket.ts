import { useState, useEffect, useCallback, useRef } from 'react';
import { api, JobStatus } from '../services/api';

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (message: string) => void;
  disconnect: () => void;
}

export const useWebSocket = (
  onMessage?: (data: any) => void,
  onError?: (error: Event) => void
): UseWebSocketReturn => {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  useEffect(() => {
    let websocket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      try {
        websocket = api.createWebSocket(
          (data) => {
            setLastMessage(data);
            if (onMessageRef.current) {
              onMessageRef.current(data);
            }
          },
          (error) => {
            console.error('WebSocket error:', error);
            if (onErrorRef.current) {
              onErrorRef.current(error);
            }
            setIsConnected(false);
            // Attempt to reconnect after 3 seconds
            reconnectTimeout = setTimeout(connect, 3000);
          }
        );

        websocket.onopen = () => {
          console.log('WebSocket connected');
          setIsConnected(true);
        };

        websocket.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds if not manually closed
          if (event.code !== 1000) {
            reconnectTimeout = setTimeout(connect, 3000);
          }
        };

        setWs(websocket);
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        setIsConnected(false);
        // Retry connection after 5 seconds
        reconnectTimeout = setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (websocket) {
        websocket.close(1000, 'Component unmounted');
      }
    };
  }, []); // Empty dependency array to prevent recreation

  const sendMessage = useCallback((message: string) => {
    if (ws && isConnected) {
      ws.send(message);
    }
  }, [ws, isConnected]);

  const disconnect = useCallback(() => {
    if (ws) {
      ws.close(1000, 'Manual disconnect');
      setWs(null);
      setIsConnected(false);
    }
  }, [ws]);

  return { isConnected, lastMessage, sendMessage, disconnect };
};

interface UseJobStatusReturn {
  status: JobStatus | null;
  isLoading: boolean;
  error: string | null;
  startPolling: (jobId: string) => void;
  stopPolling: () => void;
}

export const useJobStatus = (): UseJobStatusReturn => {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);

  const pollStatus = useCallback(async (id: string) => {
    try {
      setError(null);
      const statusData = await api.getJobStatus(id);
      setStatus(statusData);
      
      // Stop polling if job is completed or failed
      if (statusData.status === 'completed' || statusData.status === 'failed') {
        setIsLoading(false);
        if (intervalId) {
          clearInterval(intervalId);
          setIntervalId(null);
        }
      }
    } catch (err: any) {
      // Check if the job was cleaned up (finished)
      if (err.response?.status === 404 && err.response?.data?.detail?.includes('cleaned up')) {
        // Job was cleaned up because it finished - this is normal
        console.log(`Job ${id} was cleaned up (completed)`);
        setIsLoading(false);
        if (intervalId) {
          clearInterval(intervalId);
          setIntervalId(null);
        }
      } else {
        // Other errors - set error state
        setError(err.message || 'Failed to fetch job status');
        setIsLoading(false);
        if (intervalId) {
          clearInterval(intervalId);
          setIntervalId(null);
        }
      }
    }
  }, [intervalId]);

  const startPolling = useCallback((id: string) => {
    setJobId(id);
    setIsLoading(true);
    setError(null);
    
    // Clear any existing interval
    if (intervalId) {
      clearInterval(intervalId);
    }
    
    // Poll immediately and then every 2 seconds
    pollStatus(id);
    const newIntervalId = setInterval(() => pollStatus(id), 2000);
    setIntervalId(newIntervalId);
  }, [pollStatus, intervalId]);

  const stopPolling = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setIsLoading(false);
  }, [intervalId]);

  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

  return { status, isLoading, error, startPolling, stopPolling };
};
import { PriceData } from '../types/PriceData';

const WS_URL = 'ws://localhost:8000/ws/prices';

export type OnPriceDataCallback = (data: PriceData) => void;

export const connectWebSocket = (onData: OnPriceDataCallback) => {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      // Validar que el objeto recibido tenga la estructura esperada
      if (typeof data.bid === 'number' && typeof data.timestamp === 'string') {
        onData(data as PriceData);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };

  return ws;
}; 
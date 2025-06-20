export interface PriceData {
  bid: number;
  ask: number;
  will_go_up: number; // 1 = sube, 0 = baja
  earnings: number;
  operations: number;
  accuracy: number;
  timestamp: string;
} 
import { create } from 'zustand';
import { PriceData } from '../types/PriceData';

interface PriceStoreState {
  priceHistory: PriceData[];
  latestPrice: PriceData | null;
  addPriceData: (data: PriceData) => void;
  clearHistory: () => void;
}

const MAX_HISTORY_LENGTH = 500;

export const usePriceStore = create<PriceStoreState>((set) => ({
  priceHistory: [],
  latestPrice: null,
  addPriceData: (data) =>
    set((state) => {
      const newHistory = [...state.priceHistory, data];
      if (newHistory.length > MAX_HISTORY_LENGTH) {
        newHistory.shift(); // Elimina el elemento más antiguo
      }
      return {
        priceHistory: newHistory,
        latestPrice: data,
      };
    }),
  clearHistory: () => set({ priceHistory: [], latestPrice: null }),
})); 
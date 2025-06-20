import React, { useEffect } from 'react';
import './Dashboard.scss';
import { usePriceStore } from '../../store/usePriceStore';
import { connectWebSocket } from '../../services/websocketService';
import RealTimeChart from '../../components/Chart';
import { ThemeSwitcher } from '../../components/ThemeSwitcher';

const Dashboard: React.FC = () => {
  const { priceHistory, latestPrice, addPriceData, clearHistory } = usePriceStore();

  useEffect(() => {
    // Limpia el historial al montar el componente
    clearHistory();
    // Conecta al WebSocket y le pasa la función para añadir datos al store
    const ws = connectWebSocket(addPriceData);

    // Se asegura de cerrar la conexión al desmontar el componente
    return () => {
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const chartData = priceHistory.map(d => ({
    ...d,
    mid: Number(((d.bid + d.ask) / 2).toFixed(2)),
  }));

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null) return '$--';
    return `$${value.toFixed(2)}`;
  };

  const formatPercentage = (value: number | undefined) => {
    if (value === undefined || value === null) return '--%';
    return `${Math.round(value * 100)}%`;
  };
  
  const midPrice = latestPrice ? ((latestPrice.bid + latestPrice.ask) / 2).toFixed(2) : '--';
  const signal = latestPrice ? (latestPrice.will_go_up === 1 ? 'BUY' : 'SELL') : '--';
  const signalArrow = latestPrice ? (latestPrice.will_go_up === 1 ? '↑' : '↓') : '';

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <div className="dashboard__company-info">
          <h2 className="dashboard__symbol">A</h2>
          <div className="dashboard__company">Agile Technologies Inc.</div>
        </div>
        <div className="dashboard__header-right">
          <div className="dashboard__datetime">
            <div className="dashboard__date">{latestPrice ? new Date(latestPrice.timestamp).toLocaleDateString() : '--'}</div>
            <div className="dashboard__time">{latestPrice ? new Date(latestPrice.timestamp).toLocaleTimeString() : '--:--:--'}</div>
          </div>
          <ThemeSwitcher />
        </div>
      </header>
      <main className="dashboard__main">
        <section className="dashboard__chart-section">
          {chartData.length > 0 ? (
            <RealTimeChart data={chartData} />
          ) : (
            <div className="dashboard__chart-placeholder">Esperando datos del servidor...</div>
          )}
        </section>
        <section className="dashboard__info-panels">
          <div className="dashboard__panel">
            <div className="dashboard__panel-label">Bid</div>
            <div className="dashboard__panel-value bid">{latestPrice?.bid.toFixed(2) ?? '--'}</div>
          </div>
          <div className="dashboard__panel">
            <div className="dashboard__panel-label">Mid Price</div>
            <div className="dashboard__panel-value mid">{midPrice}</div>
          </div>
          <div className="dashboard__panel">
            <div className="dashboard__panel-label">Ask</div>
            <div className="dashboard__panel-value ask">{latestPrice?.ask.toFixed(2) ?? '--'}</div>
          </div>
        </section>
        <section className="dashboard__summary-panels">
          <div className="dashboard__summary">
            <div className="dashboard__summary-label">Profits</div>
            <div className="dashboard__summary-value profits">{formatCurrency(latestPrice?.earnings)}</div>
          </div>
          <div className="dashboard__summary">
            <div className="dashboard__summary-label">Operaciones</div>
            <div className="dashboard__summary-value operations">{latestPrice?.operations ?? '--'}</div>
          </div>
          <div className="dashboard__summary">
            <div className="dashboard__summary-label">Tasa de Acierto</div>
            <div className="dashboard__summary-value accuracy">{formatPercentage(latestPrice?.accuracy)}</div>
          </div>
          <div className="dashboard__summary">
             <div className={`dashboard__signal ${signal.toLowerCase()}`}>{signal} {signalArrow}</div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard; 
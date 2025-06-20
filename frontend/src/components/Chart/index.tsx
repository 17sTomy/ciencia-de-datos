import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { PriceData } from '../../types/PriceData';
import { useThemeStore } from '../../store/useThemeStore';

interface RealTimeChartProps {
  data: (PriceData & { mid: number })[];
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({ data }) => {
  const { theme } = useThemeStore();

  const gridColor = theme === 'dark' ? '#3e4a5f' : '#e0e0e0';
  const textColor = theme === 'dark' ? '#ecf0f1' : '#2c3e50';
  const tooltipBg = theme === 'dark' ? '#283344' : '#ffffff';

  const formatXAxis = (tickItem: string) => {
    return new Date(tickItem).toLocaleTimeString([], {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatXAxis}
          stroke={textColor}
          tick={{ fill: textColor }}
        />
        <YAxis
          dataKey="mid"
          domain={['dataMin', 'dataMax']}
          stroke={textColor}
          tick={{ fill: textColor }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: tooltipBg,
            borderColor: gridColor,
            color: textColor,
          }}
          labelStyle={{ color: textColor }}
        />
        <Legend wrapperStyle={{ color: textColor }} />
        <Line
          type="monotone"
          dataKey="mid"
          name="Mid Price"
          stroke="#f1c40f"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default RealTimeChart; 
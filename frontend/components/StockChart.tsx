'use client';

import { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

interface StockChartProps {
  data: any;
}

export default function StockChart({ data }: StockChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!chartRef.current || !data || !data.data || data.data.length === 0) {
      console.log('Chart data not ready:', { hasRef: !!chartRef.current, hasData: !!data, dataLength: data?.data?.length });
      return;
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) {
      console.error('Failed to get canvas context');
      return;
    }

    // Destroy previous chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Process dates and prices with validation
    const dates = data.data.map((d: any) => {
      const dateStr = d.date || d.Date;
      return dateStr ? dateStr.split('T')[0] : 'Unknown';
    });
    
    const prices = data.data.map((d: any) => {
      const close = d.close || d.Close || 0;
      return typeof close === 'number' ? close : parseFloat(close) || 0;
    });
    
    console.log('Chart data prepared:', { 
      symbol: data.symbol, 
      points: dates.length, 
      priceRange: [Math.min(...prices), Math.max(...prices)] 
    });

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: `${data.symbol} Closing Price`,
            data: prices,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#e5e7eb',
              font: {
                size: 14,
                weight: 'bold',
              },
            },
          },
          title: {
            display: true,
            text: `${data.symbol} Price History (Last ${data.data.length} days)`,
            color: '#f3f4f6',
            font: {
              size: 18,
              weight: 'bold',
            },
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function (context) {
                const value = context.parsed?.y;
                return value !== null && value !== undefined ? `₹${value.toFixed(2)}` : 'N/A';
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: false,
            grid: {
              color: 'rgba(75, 85, 99, 0.3)',
            },
            ticks: {
              color: '#9ca3af',
              callback: function (value) {
                return '₹' + value;
              },
            },
          },
          x: {
            grid: {
              color: 'rgba(75, 85, 99, 0.3)',
            },
            ticks: {
              color: '#9ca3af',
              maxTicksLimit: 10,
            },
          },
        },
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false,
        },
      },
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data]);

  return (
    <div className="bg-gray-800 rounded-2xl shadow-2xl p-6 mb-8 border border-gray-700">
      <div className="h-96">
        <canvas ref={chartRef}></canvas>
      </div>
    </div>
  );
}

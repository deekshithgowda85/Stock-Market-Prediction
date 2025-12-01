'use client';

import { useState, useEffect } from 'react';
import StockSelector from '@/components/StockSelector';
import StockAnalysis from '@/components/StockAnalysis';
import StockChart from '@/components/StockChart';
import PredictionPanel from '@/components/PredictionPanel';
import PreprocessingLogs from '@/components/PreprocessingLogs';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export default function Home() {
  const [stocks, setStocks] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [chartData, setChartData] = useState<any>(null);
  const [predictions, setPredictions] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const [dataInfo, setDataInfo] = useState<any>(null);

  useEffect(() => {
    loadStocks();
  }, []);

  const loadStocks = async () => {
    try {
      const response = await fetch(`${API_BASE}/stocks`);
      const data = await response.json();
      setStocks(data.stocks);
    } catch (error) {
      showMessage('Failed to load stocks', 'error');
    }
  };

  const showMessage = (text: string, type: 'success' | 'error') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 5000);
  };

  const analyzeStock = async (symbol: string) => {
    setLoading(true);
    setSelectedStock(symbol);
    // Don't clear predictions - keep them visible
    setDataInfo(null);

    try {
      // Check data freshness first
      try {
        const infoRes = await fetch(`${API_BASE}/data-info/${symbol}`);
        if (infoRes.ok) {
          const info = await infoRes.json();
          setDataInfo(info);
          if (!info.csv_fresh && info.days_old > 7) {
            showMessage(` Data is ${info.days_old} days old. Auto-fetching fresh data...`, 'success');
          }
        }
      } catch (e) {
        console.log('Data info check skipped');
      }

      // Fetch data from CSV by default (force_update=false)
      const [analysisRes, dataRes] = await Promise.all([
        fetch(`${API_BASE}/analyze/${symbol}?auto_update=false`),
        fetch(`${API_BASE}/data/${symbol}?limit=100&force_update=false`)
      ]);

      if (!analysisRes.ok || !dataRes.ok) {
        throw new Error('API request failed');
      }

      const analysis = await analysisRes.json();
      const stockData = await dataRes.json();

      setAnalysisData(analysis);
      setChartData(stockData);
      
      const dataSource = analysis.data_source || 'CSV data';
      showMessage(` Analysis complete! ${dataSource}`, 'success');
    } catch (error) {
      showMessage('Analysis failed. Please try again.', 'error');
      console.error('Analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchLiveData = async () => {
    if (!selectedStock) {
      showMessage(' Please select a stock first', 'error');
      return;
    }

    setLoading(true);
    setMessage({ text: ' Fetching live data from YFinance...', type: 'success' });
    
    try {
      // Use force_update=true to trigger YFinance fetch
      const response = await fetch(`${API_BASE}/data/${selectedStock}?limit=100&force_update=true`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch live data');
      }
      
      const data = await response.json();
      
      // Update chart with new data
      setChartData(data);
      
      // Show appropriate message based on data source
      if (data.data_source.includes('YFinance')) {
        showMessage(`✅ Live data fetched! Updated with YFinance data`, 'success');
      } else {
        showMessage(`⚠️ YFinance unavailable. Using CSV data (last updated: ${data.last_date || 'N/A'})`, 'error');
      }
      
      // Re-analyze with fresh data
      if (data.data_source.includes('YFinance')) {
        setTimeout(() => analyzeStock(selectedStock), 500);
      }
    } catch (error) {
      showMessage('❌ Failed to fetch live data. Using CSV fallback.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const generatePrediction = async () => {
    if (!selectedStock) {
      showMessage('⚠️ Please select a stock first', 'error');
      return;
    }

    setLoading(true);
    try {
      // First ensure we have fresh data analyzed
      if (!analysisData) {
        showMessage('⚠️ Please analyze the stock first', 'error');
        setLoading(false);
        return;
      }

      setMessage({ text: ' Generating predictions...', type: 'success' });
      
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock,
          days: 30,
          model_type: 'auto'
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }
      
      setPredictions(data);
      showMessage(`✅ Predictions generated using ${data.model_used.toUpperCase()} model`, 'success');
    } catch (error: any) {
      showMessage(`❌ Prediction failed: ${error.message || 'Unknown error'}`, 'error');
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Message */}
      {message && (
        <div className={`rounded-xl p-4 mb-6 shadow-sm border-2 ${
          message.type === 'success' 
            ? 'bg-white dark:bg-gray-900 text-green-600 dark:text-green-400 border-green-500 dark:border-green-600' 
            : 'bg-white dark:bg-gray-900 text-red-600 dark:text-red-400 border-red-500 dark:border-red-600'
        }`}>
          <div className="font-medium">
            {message.text}
          </div>
        </div>
      )}

      {/* Controls */}
      <StockSelector
        stocks={stocks}
        selectedStock={selectedStock}
        onAnalyze={analyzeStock}
        onFetchLive={fetchLiveData}
        onPredict={generatePrediction}
        loading={loading}
      />

      {/* Preprocessing Logs Button */}
      {selectedStock && (
        <div className="mb-6 flex justify-center">
          <PreprocessingLogs symbol={selectedStock} apiBase={API_BASE} />
        </div>
      )}

      {/* Data Info Badge */}
      {dataInfo && selectedStock && (
        <div className="mb-6">
          <div className={`rounded-xl p-4 border-2 shadow-sm ${
            dataInfo.csv_fresh 
              ? 'bg-white dark:bg-gray-900 border-green-500 dark:border-green-600 text-green-600 dark:text-green-400' 
              : 'bg-white dark:bg-gray-900 border-yellow-500 dark:border-yellow-600 text-yellow-600 dark:text-yellow-400'
          }`}>
            <div className="flex items-center justify-between">
              <div className="font-medium">
                <span className="font-bold">Data Status: </span>
                {dataInfo.csv_fresh ? '✅ Fresh' : '⚠️ Outdated'}
                <span className="ml-3 text-sm opacity-90">
                  Last Update: {dataInfo.last_date}
                </span>
              </div>
              <div className="text-sm font-medium opacity-90">
                Age: {dataInfo.days_old} days
                {!dataInfo.csv_fresh && ' (Auto-updated from YFinance)'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500"></div>
          <p className="text-gray-800 dark:text-gray-300 text-xl mt-4 font-semibold">Loading...</p>
        </div>
      )}

      {/* Analysis Results */}
      {analysisData && !loading && (
        <>
          <StockAnalysis data={analysisData} />
          {chartData && <StockChart data={chartData} />}
          {predictions && <PredictionPanel data={predictions} />}
        </>
      )}
    </div>
  );
}

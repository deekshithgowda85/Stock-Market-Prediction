'use client';

import { useState, useEffect } from 'react';
import PreprocessingLogs from '@/components/PreprocessingLogs';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface Message {
  type: 'success' | 'error' | 'info';
  text: string;
}

interface PipelineStep {
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  message?: string;
  data?: any;
}

export default function PipelinePage() {
  const [stocks, setStocks] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [message, setMessage] = useState<Message | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Pipeline steps
  const [steps, setSteps] = useState<PipelineStep[]>([
    { name: '1. Data Loading', status: 'pending' },
    { name: '2. Preprocessing', status: 'pending' },
    { name: '3. Analysis', status: 'pending' },
    { name: '4. Prediction', status: 'pending' },
  ]);

  // Configuration
  const [config, setConfig] = useState({
    fetchLive: false,
    saveLiveData: false,
    modelType: 'auto',
    predictionDays: 30,
    preprocess: true,
  });

  // Results
  const [results, setResults] = useState<any>(null);

  useEffect(() => {
    loadStocks();
  }, []);

  const loadStocks = async () => {
    try {
      const response = await fetch(`${API_BASE}/stocks`);
      const data = await response.json();
      setStocks(data.stocks || []);
    } catch (error) {
      console.error('Failed to load stocks:', error);
    }
  };

  const showMessage = (text: string, type: Message['type'] = 'info') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 5000);
  };

  const updateStep = (index: number, status: PipelineStep['status'], message?: string, data?: any) => {
    setSteps(prev => prev.map((step, i) => 
      i === index ? { ...step, status, message, data } : step
    ));
  };

  const resetPipeline = () => {
    setSteps([
      { name: '1. Data Loading', status: 'pending' },
      { name: '2. Preprocessing', status: 'pending' },
      { name: '3. Analysis', status: 'pending' },
      { name: '4. Prediction', status: 'pending' },
    ]);
    setResults(null);
  };

  const executePipeline = async () => {
    if (!selectedStock) {
      showMessage('Please select a stock first', 'error');
      return;
    }

    setLoading(true);
    resetPipeline();

    try {
      // Step 1: Data Loading
      updateStep(0, 'running');
      
      let dataResponse;
      if (config.fetchLive) {
        // Try to fetch live data with force_update
        dataResponse = await fetch(
          `${API_BASE}/data/${selectedStock}?limit=100&force_update=true`
        );
      } else {
        // Load from CSV
        dataResponse = await fetch(`${API_BASE}/data/${selectedStock}?limit=100&force_update=false`);
      }

      if (!dataResponse.ok) {
        throw new Error('Data loading failed');
      }

      const dataResult = await dataResponse.json();
      updateStep(0, 'success', `Loaded ${dataResult.records_fetched || dataResult.total_records} records`, dataResult);

      // Step 2: Preprocessing
      updateStep(1, 'running');
      const prepResponse = await fetch(`${API_BASE}/preprocess/${selectedStock}`);
      
      if (!prepResponse.ok) {
        throw new Error('Preprocessing failed');
      }

      const prepResult = await prepResponse.json();
      updateStep(1, 'success', `Added ${prepResult.features_added} features`, prepResult);

      // Step 3: Analysis
      updateStep(2, 'running');
      const analysisResponse = await fetch(
        `${API_BASE}/analyze/${selectedStock}?preprocess=${config.preprocess}&auto_update=false`
      );

      if (!analysisResponse.ok) {
        throw new Error('Analysis failed');
      }

      const analysisResult = await analysisResponse.json();
      updateStep(2, 'success', `Latest price: ₹${analysisResult.summary.latest_price.toFixed(2)}`, analysisResult);

      // Step 4: Prediction
      updateStep(3, 'running');
      const predResponse = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock,
          days: config.predictionDays,
          model_type: config.modelType,
        }),
      });

      if (!predResponse.ok) {
        throw new Error('Prediction failed');
      }

      const predResult = await predResponse.json();
      updateStep(3, 'success', `Generated ${predResult.forecast_days}-day forecast using ${predResult.model_used.toUpperCase()}`, predResult);

      setResults({
        data: dataResult,
        preprocessing: prepResult,
        analysis: analysisResult,
        prediction: predResult,
      });

      showMessage('Pipeline executed successfully!', 'success');
    } catch (error: any) {
      const currentStep = steps.findIndex(s => s.status === 'running');
      if (currentStep >= 0) {
        updateStep(currentStep, 'error', error.message);
      }
      showMessage(`Pipeline failed: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const executeFullPipeline = async () => {
    if (!selectedStock) {
      showMessage('Please select a stock first', 'error');
      return;
    }

    setLoading(true);
    resetPipeline();

    try {
      updateStep(0, 'running');
      updateStep(1, 'running');
      updateStep(2, 'running');
      updateStep(3, 'running');

      const response = await fetch(`${API_BASE}/pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: selectedStock,
          days: config.predictionDays,
          model_type: config.modelType,
          fetch_live: config.fetchLive,
          save_live: config.saveLiveData,
        }),
      });

      if (!response.ok) {
        throw new Error('Full pipeline execution failed');
      }

      const result = await response.json();

      updateStep(0, 'success', 'Data loaded');
      updateStep(1, 'success', 'Preprocessing complete');
      updateStep(2, 'success', `Latest price: ₹${result.analysis.latest_price.toFixed(2)}`);
      updateStep(3, 'success', `${result.predictions.length} predictions generated`);

      setResults(result);
      showMessage('Full pipeline executed successfully!', 'success');
    } catch (error: any) {
      showMessage(`Pipeline failed: ${error.message}`, 'error');
      steps.forEach((_, i) => {
        if (steps[i].status === 'running') {
          updateStep(i, 'error', 'Failed');
        }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 py-12 px-4">
      <div className="container mx-auto max-w-7xl">
        {/* Header with Logs Button */}
        <div className="text-center mb-12 relative">
          <h1 className="text-5xl font-bold text-white mb-4">
            🔧 Pipeline Control Center
          </h1>
          <p className="text-gray-400 text-lg">
            Complete control over data processing, analysis, and predictions
          </p>
          
          {/* Logs Button in Top Right Corner */}
          {selectedStock && (
            <div className="absolute top-0 right-0">
              <PreprocessingLogs symbol={selectedStock} apiBase={API_BASE} />
            </div>
          )}
        </div>

        {/* Message */}
        {message && (
          <div className={`rounded-xl p-4 mb-6 ${
            message.type === 'success' ? 'bg-green-900/50 text-green-300 border border-green-700' :
            message.type === 'error' ? 'bg-red-900/50 text-red-300 border border-red-700' :
            'bg-blue-900/50 text-blue-300 border border-blue-700'
          }`}>
            {message.text}
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left: Configuration */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700 mb-6">
              <h2 className="text-2xl font-bold text-white mb-6">Configuration</h2>

              {/* Stock Selection */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-300 mb-2">
                  Stock Symbol
                </label>
                <select
                  value={selectedStock}
                  onChange={(e) => setSelectedStock(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 text-white border-2 border-gray-600 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-500/50 transition-all"
                  disabled={loading}
                >
                  <option value="">-- Select Stock --</option>
                  {stocks.map((stock) => (
                    <option key={stock} value={stock}>
                      {stock}
                    </option>
                  ))}
                </select>
              </div>

              {/* Data Source */}
              <div className="mb-6">
                <label className="flex items-center text-gray-300 mb-2">
                  <input
                    type="checkbox"
                    checked={config.fetchLive}
                    onChange={(e) => setConfig({...config, fetchLive: e.target.checked})}
                    className="mr-2 w-4 h-4"
                    disabled={loading}
                  />
                  Fetch Live Data from YFinance
                </label>
                {config.fetchLive && (
                  <label className="flex items-center text-gray-400 ml-6">
                    <input
                      type="checkbox"
                      checked={config.saveLiveData}
                      onChange={(e) => setConfig({...config, saveLiveData: e.target.checked})}
                      className="mr-2 w-4 h-4"
                      disabled={loading}
                    />
                    Save to CSV
                  </label>
                )}
              </div>

              {/* Preprocessing */}
              <div className="mb-6">
                <label className="flex items-center text-gray-300">
                  <input
                    type="checkbox"
                    checked={config.preprocess}
                    onChange={(e) => setConfig({...config, preprocess: e.target.checked})}
                    className="mr-2 w-4 h-4"
                    disabled={loading}
                  />
                  Apply Preprocessing Pipeline
                </label>
              </div>

              {/* Model Selection */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-300 mb-2">
                  Model Type
                </label>
                <select
                  value={config.modelType}
                  onChange={(e) => setConfig({...config, modelType: e.target.value})}
                  className="w-full px-4 py-3 bg-gray-900 text-white border-2 border-gray-600 rounded-lg focus:border-blue-500 transition-all"
                  disabled={loading}
                >
                  <option value="auto">Auto (Best Model)</option>
                  <option value="arima">ARIMA</option>
                  <option value="prophet">Prophet</option>
                </select>
              </div>

              {/* Prediction Days */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-300 mb-2">
                  Prediction Days: {config.predictionDays}
                </label>
                <input
                  type="range"
                  min="7"
                  max="90"
                  value={config.predictionDays}
                  onChange={(e) => setConfig({...config, predictionDays: parseInt(e.target.value)})}
                  className="w-full"
                  disabled={loading}
                />
              </div>

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={executePipeline}
                  disabled={!selectedStock || loading}
                  className="w-full bg-white text-gray-900 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg"
                >
                  {loading ? 'Running...' : 'Execute Step-by-Step'}
                </button>
                <button
                  onClick={executeFullPipeline}
                  disabled={!selectedStock || loading}
                  className="w-full bg-transparent border-2 border-white text-white px-6 py-3 rounded-lg font-semibold hover:bg-white hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
                >
                  Execute Full Pipeline
                </button>
                <button
                  onClick={resetPipeline}
                  disabled={loading}
                  className="w-full bg-transparent border-2 border-gray-600 text-gray-300 px-6 py-3 rounded-lg font-semibold hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          {/* Right: Pipeline Steps & Results */}
          <div className="lg:col-span-2">
            {/* Pipeline Steps */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700 mb-6">
              <h2 className="text-2xl font-bold text-white mb-6">Pipeline Status</h2>
              <div className="space-y-4">
                {steps.map((step, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      step.status === 'pending' ? 'border-gray-700 bg-gray-900/50' :
                      step.status === 'running' ? 'border-blue-500 bg-blue-900/20 animate-pulse' :
                      step.status === 'success' ? 'border-green-500 bg-green-900/20' :
                      'border-red-500 bg-red-900/20'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          step.status === 'pending' ? 'bg-gray-700' :
                          step.status === 'running' ? 'bg-blue-500' :
                          step.status === 'success' ? 'bg-green-500' :
                          'bg-red-500'
                        }`}>
                          {step.status === 'pending' && '⏸️'}
                          {step.status === 'running' && '⚡'}
                          {step.status === 'success' && '✓'}
                          {step.status === 'error' && '✗'}
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{step.name}</h3>
                          {step.message && (
                            <p className="text-sm text-gray-400">{step.message}</p>
                          )}
                        </div>
                      </div>
                      <span className={`text-xs font-semibold uppercase ${
                        step.status === 'pending' ? 'text-gray-500' :
                        step.status === 'running' ? 'text-blue-400' :
                        step.status === 'success' ? 'text-green-400' :
                        'text-red-400'
                      }`}>
                        {step.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Results */}
            {results && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-2xl font-bold text-white mb-6">Results Summary</h2>
                
                {/* Analysis Results */}
                {results.analysis && (
                  <div className="mb-6 p-4 bg-gray-900/50 rounded-lg">
                    <h3 className="text-lg font-semibold text-blue-400 mb-3">📊 Analysis</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Latest Price:</span>
                        <span className="text-white font-bold ml-2">
                          ₹{results.analysis.summary?.latest_price?.toFixed(2)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Total Return:</span>
                        <span className={`font-bold ml-2 ${
                          results.analysis.performance?.total_return_percent >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {results.analysis.performance?.total_return_percent?.toFixed(2)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Volatility:</span>
                        <span className="text-white font-bold ml-2">
                          {results.analysis.performance?.volatility_percent?.toFixed(2)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Data Points:</span>
                        <span className="text-white font-bold ml-2">
                          {results.analysis.summary?.total_records}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Prediction Results */}
                {results.prediction && (
                  <div className="p-4 bg-gray-900/50 rounded-lg">
                    <h3 className="text-lg font-semibold text-purple-400 mb-3">🔮 Predictions</h3>
                    <div className="mb-3">
                      <span className="text-gray-400">Model Used:</span>
                      <span className="text-white font-bold ml-2 uppercase">
                        {results.prediction.model_used}
                      </span>
                    </div>
                    <div className="grid grid-cols-5 gap-2">
                      {results.prediction.predictions?.slice(0, 5).map((pred: any, idx: number) => (
                        <div key={idx} className="bg-gray-800 p-2 rounded text-center">
                          <div className="text-xs text-gray-400">Day {pred.day}</div>
                          <div className="text-sm font-bold text-blue-400">
                            ₹{pred.predicted_close?.toFixed(2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

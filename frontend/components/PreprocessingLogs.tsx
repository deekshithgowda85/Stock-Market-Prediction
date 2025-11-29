'use client';

import { useState } from 'react';

interface LogEntry {
  step: string;
  message: string;
  status: 'running' | 'success' | 'error' | 'info';
  timestamp: string;
  data?: any;
}

interface PreprocessingLogsProps {
  symbol: string;
  apiBase: string;
}

export default function PreprocessingLogs({ symbol, apiBase }: PreprocessingLogsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<any>(null);

  const fetchLogs = async () => {
    if (!symbol) {
      alert('Please select a stock first');
      return;
    }

    setLoading(true);
    setIsOpen(true);

    try {
      const response = await fetch(`${apiBase}/preprocess/${symbol}?add_features=true&handle_outliers=true`);
      const data = await response.json();

      if (response.ok) {
        setLogs(data.logs || []);
        setSummary(data.summary || null);
      } else {
        alert('Failed to fetch preprocessing logs');
      }
    } catch (error) {
      console.error('Error fetching logs:', error);
      alert('Failed to fetch preprocessing logs');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'running': return 'text-yellow-400';
      case 'info': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return '✓';
      case 'error': return '✗';
      case 'running': return '⟳';
      case 'info': return 'ℹ';
      default: return '•';
    }
  };

  return (
    <div>
      {/* Logs Button */}
      <button
        onClick={fetchLogs}
        disabled={!symbol || loading}
        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
      >
        <span>📊</span>
        <span>{loading ? 'Loading...' : 'View Preprocessing Logs'}</span>
      </button>

      {/* Logs Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="p-6 border-b border-gray-700 flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-white">Preprocessing Pipeline Logs</h2>
                <p className="text-gray-400 mt-1">Stock: {symbol}</p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>

            {/* Summary */}
            {summary && (
              <div className="p-4 bg-gray-900 border-b border-gray-700">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">Initial Rows</div>
                    <div className="text-white font-bold">{summary.initial_rows}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Final Rows</div>
                    <div className="text-white font-bold">{summary.final_rows}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Initial Columns</div>
                    <div className="text-white font-bold">{summary.initial_columns}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Final Columns</div>
                    <div className="text-white font-bold">{summary.final_columns}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Features Added</div>
                    <div className="text-green-400 font-bold">+{summary.features_added}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Logs Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {logs.length === 0 ? (
                <div className="text-center text-gray-400 py-8">
                  {loading ? 'Loading logs...' : 'No logs available'}
                </div>
              ) : (
                logs.map((log, index) => (
                  <div
                    key={index}
                    className="bg-gray-900 rounded-lg p-4 border border-gray-700"
                  >
                    <div className="flex items-start gap-3">
                      <span className={`text-xl ${getStatusColor(log.status)}`}>
                        {getStatusIcon(log.status)}
                      </span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold text-white">{log.step}</h3>
                          <span className="text-xs text-gray-500">
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-gray-300 mt-1">{log.message}</p>
                        
                        {/* Additional Data */}
                        {log.data && Object.keys(log.data).length > 0 && (
                          <div className="mt-3 bg-gray-800 rounded p-3 text-sm">
                            <div className="grid grid-cols-2 gap-2">
                              {Object.entries(log.data).map(([key, value]) => {
                                if (typeof value === 'object') return null;
                                return (
                                  <div key={key} className="flex justify-between">
                                    <span className="text-gray-400">{key}:</span>
                                    <span className="text-white font-mono">
                                      {typeof value === 'number' ? value.toLocaleString() : String(value)}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-700 flex justify-end">
              <button
                onClick={() => setIsOpen(false)}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

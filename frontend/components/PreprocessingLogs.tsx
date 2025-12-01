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
      case 'success': return 'text-green-600 dark:text-green-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      case 'running': return 'text-yellow-600 dark:text-yellow-400';
      case 'info': return 'text-blue-600 dark:text-blue-400';
      default: return 'text-gray-600 dark:text-gray-400';
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
        className="px-4 py-2 bg-gray-900 text-white dark:bg-white dark:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors shadow-sm flex items-center gap-2"
      >
        <span>📊</span>
        <span>{loading ? 'Loading...' : 'View Preprocessing Logs'}</span>
      </button>

      {/* Logs Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-[hsl(var(--card))] rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col">
            {/* Header */}
            <div className="p-6 border-b border-[hsl(var(--border))] flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-[hsl(var(--card-foreground))]">Preprocessing Pipeline Logs</h2>
                <p className="text-[hsl(var(--muted-foreground))] mt-1">Stock: {symbol}</p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] text-2xl"
              >
                ×
              </button>
            </div>

            {/* Summary */}
            {summary && (
              <div className="p-4 bg-[hsl(var(--muted))] border-b border-[hsl(var(--border))]">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                  <div>
                    <div className="text-[hsl(var(--muted-foreground))]">Initial Rows</div>
                    <div className="text-[hsl(var(--foreground))] font-bold">{summary.initial_rows}</div>
                  </div>
                  <div>
                    <div className="text-[hsl(var(--muted-foreground))]">Final Rows</div>
                    <div className="text-[hsl(var(--foreground))] font-bold">{summary.final_rows}</div>
                  </div>
                  <div>
                    <div className="text-[hsl(var(--muted-foreground))]">Initial Columns</div>
                    <div className="text-[hsl(var(--foreground))] font-bold">{summary.initial_columns}</div>
                  </div>
                  <div>
                    <div className="text-[hsl(var(--muted-foreground))]">Final Columns</div>
                    <div className="text-[hsl(var(--foreground))] font-bold">{summary.final_columns}</div>
                  </div>
                  <div>
                    <div className="text-[hsl(var(--muted-foreground))]">Features Added</div>
                    <div className="text-green-600 font-bold">+{summary.features_added}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Logs Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {logs.length === 0 ? (
                <div className="text-center text-[hsl(var(--muted-foreground))] py-8">
                  {loading ? 'Loading logs...' : 'No logs available'}
                </div>
              ) : (
                logs.map((log, index) => (
                  <div
                    key={index}
                    className="bg-[hsl(var(--muted))] rounded-lg p-4 border border-[hsl(var(--border))]"
                  >
                    <div className="flex items-start gap-3">
                      <span className={`text-xl ${getStatusColor(log.status)}`}>
                        {getStatusIcon(log.status)}
                      </span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold text-[hsl(var(--foreground))]">{log.step}</h3>
                          <span className="text-xs text-[hsl(var(--muted-foreground))]">
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-[hsl(var(--card-foreground))] mt-1">{log.message}</p>
                        
                        {/* Additional Data */}
                        {log.data && Object.keys(log.data).length > 0 && (
                          <div className="mt-3 bg-[hsl(var(--background))] rounded p-3 text-sm border border-[hsl(var(--border))]">
                            <div className="grid grid-cols-2 gap-2">
                              {Object.entries(log.data).map(([key, value]) => {
                                if (typeof value === 'object') return null;
                                return (
                                  <div key={key} className="flex justify-between">
                                    <span className="text-[hsl(var(--muted-foreground))]">{key}:</span>
                                    <span className="text-[hsl(var(--foreground))] font-mono">
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
            <div className="p-4 border-t border-[hsl(var(--border))] flex justify-end">
              <button
                onClick={() => setIsOpen(false)}
                className="px-6 py-2 bg-gray-900 text-white dark:bg-white dark:text-gray-900 hover:opacity-90 rounded-lg transition-colors"
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

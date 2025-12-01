interface StockSelectorProps {
  stocks: string[];
  selectedStock: string;
  onAnalyze: (symbol: string) => void;
  onFetchLive: () => void;
  onPredict: () => void;
  loading: boolean;
}

export default function StockSelector({
  stocks,
  selectedStock,
  onAnalyze,
  onFetchLive,
  onPredict,
  loading
}: StockSelectorProps) {
  return (
    <div className="bg-[hsl(var(--card))] rounded-xl shadow-md p-6 mb-6 border border-[hsl(var(--border))]">
      <div className="flex flex-wrap items-end gap-3">
        {/* Stock Selector */}
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-semibold text-[hsl(var(--card-foreground))] mb-1.5">
            Select Stock
          </label>
          <select
            value={selectedStock}
            onChange={(e) => onAnalyze(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-[hsl(var(--background))] text-[hsl(var(--foreground))] border border-[hsl(var(--border))] rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-all"
            disabled={loading}
          >
            <option value="">-- Choose Stock --</option>
            {stocks.map((stock) => (
              <option key={stock} value={stock}>
                {stock}
              </option>
            ))}
          </select>
        </div>

        {/* Analyze Button */}
        <button
          onClick={() => selectedStock && onAnalyze(selectedStock)}
          disabled={!selectedStock || loading}
          className="px-5 py-2 text-sm bg-gray-900 text-white dark:bg-white dark:text-gray-900 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
        >
          Analyze
        </button>

        {/* Fetch Live Data Button */}
        <div className="flex items-end">
          <button
            onClick={onFetchLive}
            disabled={!selectedStock || loading}
            className="w-full bg-gray-900 text-white dark:bg-white dark:text-gray-900 px-6 py-2 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            Fetch Live Data
          </button>
        </div>

        {/* Predict Button */}
        <div className="flex items-end">
          <button
            onClick={onPredict}
            disabled={!selectedStock || loading}
            className="w-full bg-gray-900 text-white dark:bg-white dark:text-gray-900 px-6 py-2 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}

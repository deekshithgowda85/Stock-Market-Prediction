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
    <div className="bg-gray-800 rounded-xl shadow-lg p-4 mb-6 border border-gray-700">
      <div className="flex flex-wrap items-end gap-3">
        {/* Stock Selector */}
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-semibold text-gray-300 mb-1.5">
            Select Stock
          </label>
          <select
            value={selectedStock}
            onChange={(e) => onAnalyze(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-blue-400 focus:ring-1 focus:ring-blue-400 transition-all"
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
          className="px-5 py-2 text-sm bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm border border-blue-700"
        >
          Analyze
        </button>

        {/* Fetch Live Data Button */}
        <div className="flex items-end">
          <button
            onClick={onFetchLive}
            disabled={!selectedStock || loading}
            className="w-full bg-transparent border-2 border-emerald-500 text-emerald-400 px-6 py-3 rounded-xl font-semibold hover:bg-emerald-600 hover:text-white transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Fetch Live Data
          </button>
        </div>

        {/* Predict Button */}
        <div className="flex items-end">
          <button
            onClick={onPredict}
            disabled={!selectedStock || loading}
            className="w-full bg-transparent border-2 border-purple-500 text-purple-400 px-6 py-3 rounded-xl font-semibold hover:bg-purple-600 hover:text-white transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}

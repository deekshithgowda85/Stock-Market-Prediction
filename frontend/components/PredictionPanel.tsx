interface PredictionPanelProps {
  data: any;
}

export default function PredictionPanel({ data }: PredictionPanelProps) {
  if (!data || !data.predictions || data.predictions.length === 0) {
    console.log('No prediction data available');
    return null;
  }
  const predictions = data.predictions.slice(0, 10);

  return (
    <div className="bg-gray-800 rounded-2xl shadow-2xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-gray-100">
          🔮 Price Predictions
        </h3>
        <span className="px-4 py-2 bg-blue-900/50 border border-blue-700 text-blue-300 rounded-full font-semibold text-sm">
          {data.model_used.toUpperCase()} Model
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {predictions.map((pred: any, idx: number) => (
          <div
            key={idx}
            className="bg-gray-900 rounded-xl p-4 border-2 border-gray-700 hover:border-blue-500 hover:shadow-lg transition-all transform hover:-translate-y-1"
          >
            <div className="text-sm text-gray-400 font-medium mb-1">
              Day {idx + 1}
            </div>
            <div className="text-xs text-gray-500 mb-2">
              {pred.date ? pred.date.split('T')[0] : 'N/A'}
            </div>
            <div className="text-2xl font-bold text-blue-400">
              ₹{pred.predicted_close ? pred.predicted_close.toFixed(2) : 'N/A'}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-gray-900/50 border border-gray-700 rounded-xl">
        <p className="text-sm text-gray-400">
          <span className="font-semibold text-gray-300">Note:</span> These predictions are generated using AI models and should be used for informational purposes only. Always consult with financial advisors before making investment decisions.
        </p>
      </div>
    </div>
  );
}

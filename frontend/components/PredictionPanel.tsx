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
    <div className="bg-[hsl(var(--card))] rounded-2xl shadow-lg p-6 border border-[hsl(var(--border))]">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-[hsl(var(--card-foreground))]">
          🔮 Price Predictions
        </h3>
        <span className="px-4 py-2 bg-blue-100 dark:bg-blue-900/50 border border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300 rounded-full font-semibold text-sm">
          {data.model_used.toUpperCase()} Model
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {predictions.map((pred: any, idx: number) => (
          <div
            key={idx}
            className="bg-[hsl(var(--background))] rounded-xl p-4 border-2 border-[hsl(var(--border))] hover:border-blue-500 hover:shadow-lg transition-all transform hover:-translate-y-1"
          >
            <div className="text-sm text-[hsl(var(--muted-foreground))] font-medium mb-1">
              Day {idx + 1}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500 mb-2">
              {pred.date ? pred.date.split('T')[0] : 'N/A'}
            </div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              ₹{pred.predicted_close ? pred.predicted_close.toFixed(2) : 'N/A'}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-[hsl(var(--muted))] border border-[hsl(var(--border))] rounded-xl">
        <p className="text-sm text-[hsl(var(--muted-foreground))]">
          <span className="font-semibold text-[hsl(var(--foreground))]">Note:</span> These predictions are generated using AI models and should be used for informational purposes only. Always consult with financial advisors before making investment decisions.
        </p>
      </div>
    </div>
  );
}

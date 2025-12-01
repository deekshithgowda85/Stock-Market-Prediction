interface StockAnalysisProps {
  data: any;
}

export default function StockAnalysis({ data }: StockAnalysisProps) {
  if (!data) {
    console.log('No analysis data provided');
    return null;
  }
  
  if (!data.summary || !data.performance || !data.technical) {
    console.log('Missing required analysis fields:', {
      hasSummary: !!data.summary,
      hasPerformance: !!data.performance,
      hasTechnical: !!data.technical
    });
    return null;
  }
  
  const { summary, performance, technical } = data;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {/* Summary Card */}
      <div className="bg-[hsl(var(--card))] rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow border border-[hsl(var(--border))]">
        <h3 className="text-xl font-bold text-blue-600 mb-4 border-b-2 border-blue-600 pb-2">
          📊 Summary
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Symbol</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">{data.symbol}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Latest Price</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold text-xl">₹{summary.latest_price.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Date</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold text-sm">{summary.latest_date.split('T')[0]}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Records</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">{summary.total_records.toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Performance Card */}
      <div className="bg-[hsl(var(--card))] rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow border border-[hsl(var(--border))]">
        <h3 className="text-xl font-bold text-green-600 mb-4 border-b-2 border-green-600 pb-2">
          📈 Performance
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Total Return</span>
            <span className={`font-bold text-lg ${performance.total_return_percent >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {performance.total_return_percent.toFixed(2)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">20-Day Return</span>
            <span className={`font-bold ${performance.recent_20d_return_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {performance.recent_20d_return_percent.toFixed(2)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Volatility</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">{performance.volatility_percent.toFixed(2)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Avg Volume</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold text-sm">{performance.average_volume.toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Technical Indicators Card */}
      <div className="bg-[hsl(var(--card))] rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow border border-[hsl(var(--border))]">
        <h3 className="text-xl font-bold text-purple-600 mb-4 border-b-2 border-purple-600 pb-2">
          🔧 Technical
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">20-Day MA</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">₹{technical.ma_20 ? technical.ma_20.toFixed(2) : 'N/A'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">50-Day MA</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">₹{technical.ma_50 ? technical.ma_50.toFixed(2) : 'N/A'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">vs MA20</span>
            <span className={`font-bold ${technical.current_vs_ma20 === 'Above' ? 'text-green-600' : 'text-red-600'}`}>
              {technical.current_vs_ma20}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">vs MA50</span>
            <span className={`font-bold ${technical.current_vs_ma50 === 'Above' ? 'text-green-600' : 'text-red-600'}`}>
              {technical.current_vs_ma50}
            </span>
          </div>
        </div>
      </div>

      {/* Price Range Card */}
      <div className="bg-[hsl(var(--card))] rounded-2xl shadow-xl p-6 hover:shadow-2xl transition-shadow border border-[hsl(var(--border))]">
        <h3 className="text-xl font-bold text-orange-600 mb-4 border-b-2 border-orange-600 pb-2">
          💰 Price Range
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Highest</span>
            <span className="text-green-600 font-bold">₹{performance.highest_price.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Lowest</span>
            <span className="text-red-600 font-bold">₹{performance.lowest_price.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">Range</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">₹{(performance.highest_price - performance.lowest_price).toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[hsl(var(--muted-foreground))] font-medium">% Range</span>
            <span className="text-[hsl(var(--card-foreground))] font-bold">
              {((performance.highest_price - performance.lowest_price) / performance.lowest_price * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

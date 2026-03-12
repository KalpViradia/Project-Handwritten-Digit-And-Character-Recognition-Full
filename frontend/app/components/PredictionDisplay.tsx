"use client";

interface PredictionResult {
    predictedIndex: number;
    confidence: number;
    probabilities: number[];
}

interface PredictionDisplayProps {
    prediction: PredictionResult | null;
    isLoading: boolean;
    onVisualize?: () => void;
}

export default function PredictionDisplay({ prediction, isLoading, onVisualize }: PredictionDisplayProps) {
    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center h-64">
                <div className="spinner mb-4"></div>
                <p className="text-gray-400">Analyzing digit...</p>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <div className="text-6xl mb-4 opacity-30">🔮</div>
                <p>Draw or upload a digit to see predictions</p>
            </div>
        );
    }

    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const circumference = 2 * Math.PI * 52; // radius = 52
    const strokeDashoffset = circumference - (prediction.confidence * circumference);

    return (
        <div className="space-y-6">
            {/* Main Prediction */}
            <div className="text-center">
                <div className="prediction-digit mb-2">
                    {prediction.predictedIndex}
                </div>
                <p className="text-lg text-gray-400">Predicted Digit</p>
            </div>

            {/* Confidence Ring */}
            <div className="flex justify-center">
                <div className="relative w-[120px] h-[120px]">
                    <svg viewBox="0 0 120 120" className="w-full h-full">
                        <defs>
                            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#6366f1" />
                                <stop offset="50%" stopColor="#8b5cf6" />
                                <stop offset="100%" stopColor="#06b6d4" />
                            </linearGradient>
                        </defs>
                        {/* Background circle */}
                        <circle
                            fill="none"
                            stroke="rgba(99, 102, 241, 0.2)"
                            strokeWidth="8"
                            cx="60"
                            cy="60"
                            r="52"
                        />
                        {/* Progress circle */}
                        <circle
                            fill="none"
                            stroke="url(#gradient)"
                            strokeWidth="8"
                            strokeLinecap="round"
                            cx="60"
                            cy="60"
                            r="52"
                            style={{
                                strokeDasharray: circumference,
                                strokeDashoffset: strokeDashoffset,
                                transform: 'rotate(-90deg)',
                                transformOrigin: '60px 60px',
                                transition: 'stroke-dashoffset 0.5s ease-out'
                            }}
                        />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                            <span className="text-2xl font-bold text-white">{confidencePercent}%</span>
                            <p className="text-xs text-gray-400">Confidence</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Top Predictions */}
            <div className="space-y-2">
                <p className="text-sm text-gray-400 mb-3">Top Predictions</p>
                {prediction.probabilities
                    .map((prob, digit) => ({ digit, prob }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, 10)
                    .map(({ digit, prob }) => (
                        <div key={digit} className="flex items-center gap-3">
                            <span className={`w-6 text-center font-mono text-lg ${digit === prediction.predictedIndex
                                ? "text-indigo-400 font-bold"
                                : "text-gray-400"
                                }`}>
                                {digit}
                            </span>
                            <div className="flex-1 prob-bar">
                                <div
                                    className={`prob-bar-fill ${digit === prediction.predictedIndex ? 'highlight' : ''}`}
                                    style={{
                                        width: `${prob * 100}%`,
                                        background: digit === prediction.predictedIndex
                                            ? 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)'
                                            : undefined
                                    }}
                                />
                            </div>
                            <span className={`w-14 text-right text-sm ${digit === prediction.predictedIndex
                                ? "text-indigo-400 font-semibold"
                                : "text-gray-500"
                                }`}>
                                {(prob * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
            </div>

            {/* Visualize Button */}
            {onVisualize && (
                <div className="flex justify-center pt-2">
                    <button className="btn-visualize" onClick={onVisualize}>
                        🎥 Visualize How It Works
                    </button>
                </div>
            )}
        </div>
    );
}

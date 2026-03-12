"use client";

// Character labels A-Z
const CHAR_LABELS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

interface CharacterPrediction {
    predictedCharacter: string;
    predictedIndex: number;
    confidence: number;
    probabilities: number[];
    warning?: string | null;
}

interface CharacterPredictionDisplayProps {
    prediction: CharacterPrediction | null;
    isLoading: boolean;
    onVisualize?: () => void;
}

export default function CharacterPredictionDisplay({ prediction, isLoading, onVisualize }: CharacterPredictionDisplayProps) {
    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center h-64">
                <div className="spinner mb-4"></div>
                <p className="text-gray-400">Analyzing character...</p>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <div className="text-6xl mb-4 opacity-30">🔤</div>
                <p>Draw or upload a letter to see predictions</p>
            </div>
        );
    }

    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const circumference = 2 * Math.PI * 52;
    const strokeDashoffset = circumference - (prediction.confidence * circumference);

    // Get top 5 predictions for character display
    const topPredictions = prediction.probabilities
        .map((prob, idx) => ({ char: CHAR_LABELS[idx], prob, idx }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 10);

    return (
        <div className="space-y-6">
            {/* Main Prediction */}
            <div className="text-center">
                <div className="text-8xl font-bold mb-2 bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent"
                    style={{ textShadow: '0 0 60px rgba(16, 185, 129, 0.5)' }}>
                    {prediction.predictedCharacter}
                </div>
                <p className="text-lg text-gray-400">Predicted Character</p>

                {/* Low confidence warning */}
                {prediction.warning && (
                    <div className="mt-3 p-2 bg-amber-500/20 border border-amber-500/50 rounded-lg text-amber-300 text-sm">
                        ⚠️ {prediction.warning}
                    </div>
                )}
            </div>

            {/* Confidence Ring */}
            <div className="flex justify-center">
                <div className="relative w-[120px] h-[120px]">
                    <svg viewBox="0 0 120 120" className="w-full h-full">
                        <defs>
                            <linearGradient id="charGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#10b981" />
                                <stop offset="50%" stopColor="#14b8a6" />
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
                            stroke="url(#charGradient)"
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
                {topPredictions.map(({ char, prob, idx }) => (
                    <div key={idx} className="flex items-center gap-3">
                        <span className={`w-8 text-center font-mono text-lg ${char === prediction.predictedCharacter
                            ? "text-emerald-400 font-bold"
                            : "text-gray-400"
                            }`}>
                            {char}
                        </span>
                        <div className="flex-1 prob-bar">
                            <div
                                className={`prob-bar-fill ${char === prediction.predictedCharacter ? 'highlight' : ''}`}
                                style={{
                                    width: `${prob * 100}%`,
                                    background: char === prediction.predictedCharacter
                                        ? 'linear-gradient(90deg, #10b981 0%, #14b8a6 100%)'
                                        : undefined
                                }}
                            />
                        </div>
                        <span className={`w-14 text-right text-sm ${char === prediction.predictedCharacter
                            ? "text-emerald-400 font-semibold"
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

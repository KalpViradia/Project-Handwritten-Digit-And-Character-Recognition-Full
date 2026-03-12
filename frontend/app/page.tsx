"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import DrawingCanvas from "./components/DrawingCanvas";
import FileUpload from "./components/FileUpload";
import PredictionDisplay from "./components/PredictionDisplay";
import ApiStatusBanner from "./components/ApiStatusBanner";

// Lazy-load the visualizer — only loaded when modal opens
const ForwardPassVisualizer = dynamic(
  () => import("./components/ForwardPassVisualizer"),
  { ssr: false }
);

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PredictionResult {
  predictedIndex: number;
  confidence: number;
  probabilities: number[];
  imageGrid?: number[][];
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"draw" | "upload">("draw");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiReady, setApiReady] = useState(false);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const canvasRef = useRef<{ getImageData: () => string; clearCanvas: () => void } | null>(null);

  // Wake up the Render API on page load
  const warmUpApi = useCallback(() => {
    fetch(`${API_BASE_URL}/health`).catch(() => { });
  }, []);

  useEffect(() => {
    warmUpApi();
  }, [warmUpApi]);

  const handlePredict = async (imageData: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/canvas`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const result = await response.json();
      console.log("API RESPONSE:", result);
      setPrediction({
        predictedIndex: result.predictedDigit ?? result.predicted_digit ?? result.predictedIndex,
        confidence: result.confidence,
        probabilities: result.probabilities,
        imageGrid: result.imageGrid ?? result.image_grid,
      });
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err instanceof Error ? err.message : "Failed to connect to the API. Please ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const result = await response.json();
      console.log("API RESPONSE:", result);
      setPrediction({
        predictedIndex: result.predictedDigit ?? result.predicted_digit ?? result.predictedIndex,
        confidence: result.confidence,
        probabilities: result.probabilities,
        imageGrid: result.imageGrid ?? result.image_grid,
      });
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err instanceof Error ? err.message : "Failed to connect to the API. Please ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCanvasSubmit = () => {
    if (canvasRef.current) {
      const imageData = canvasRef.current.getImageData();
      handlePredict(imageData);
    }
  };

  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
    setPrediction(null);
    setError(null);
  };

  // Digit labels 0-9
  const digitLabels = Array.from({ length: 10 }, (_, i) => String(i));

  return (
    <div className="min-h-screen flex flex-col">
      {/* API Status Banner */}
      <ApiStatusBanner onApiReady={() => setApiReady(true)} />

      {/* Header */}
      <header className="py-6 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-2">
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Digit Recognizer
            </span>
          </h1>
          <p className="text-gray-400 text-lg">
            AI-Powered Handwriting Recognition using CNN
          </p>
          <a
            href="/characters"
            className="inline-block mt-3 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            Try Character Recognition (A-Z) →
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 px-4 pb-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

            {/* Input Section */}
            <div className="glass-card p-6">
              {/* Tabs */}
              <div className="flex justify-center mb-6 border-b border-gray-700">
                <button
                  className={`tab-btn ${activeTab === "draw" ? "active" : ""}`}
                  onClick={() => setActiveTab("draw")}
                >
                  ✏️ Draw
                </button>
                <button
                  className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
                  onClick={() => setActiveTab("upload")}
                >
                  📁 Upload
                </button>
              </div>

              {/* Content */}
              {activeTab === "draw" ? (
                <div className="space-y-4">
                  <DrawingCanvas ref={canvasRef} />

                  <div className="flex gap-4 justify-center">
                    <button
                      className="btn-primary"
                      onClick={handleCanvasSubmit}
                      disabled={isLoading || !apiReady}
                      title={!apiReady ? "Waiting for AI model to load…" : undefined}
                    >
                      {isLoading ? (
                        <span className="flex items-center gap-2">
                          <span className="spinner w-5 h-5"></span>
                          Processing...
                        </span>
                      ) : !apiReady ? (
                        <span className="flex items-center gap-2">
                          <span className="spinner w-4 h-4 !border-2"></span>
                          Loading Model…
                        </span>
                      ) : (
                        "Predict"
                      )}
                    </button>
                    <button
                      className="btn-secondary"
                      onClick={handleClear}
                    >
                      Clear
                    </button>
                  </div>

                  <p className="text-center text-gray-500 text-sm">
                    Draw a digit (0-9)
                  </p>
                </div>
              ) : (
                <FileUpload
                  onFileSelect={handleFileUpload}
                  isLoading={isLoading}
                />
              )}

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-center">
                  {error}
                </div>
              )}
            </div>

            {/* Results Section */}
            <div className="glass-card p-6">
              <h2 className="text-xl font-semibold text-center mb-6 text-gray-300">
                Prediction Results
              </h2>

              <PredictionDisplay
                prediction={prediction}
                isLoading={isLoading}
                onVisualize={prediction ? () => setShowVisualizer(true) : undefined}
              />
            </div>
          </div>

          {/* Info Section */}
          <div className="mt-8 glass-card p-6">
            <h3 className="text-lg font-semibold text-indigo-300 mb-4">How it works</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-gray-400">
              <div className="text-center">
                <div className="text-3xl mb-2">🎨</div>
                <h4 className="font-medium text-white mb-1">Draw or Upload</h4>
                <p className="text-sm">Draw a digit on the canvas or upload an image of a handwritten number (0-9)</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🧠</div>
                <h4 className="font-medium text-white mb-1">CNN Analysis</h4>
                <p className="text-sm">Our Convolutional Neural Network processes the image with 99%+ accuracy</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">✨</div>
                <h4 className="font-medium text-white mb-1">Instant Results</h4>
                <p className="text-sm">Get the predicted digit along with confidence scores for all classes</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 text-center text-gray-500 text-sm">
        <p>Built with ❤️ using TensorFlow, FastAPI & Next.js</p>
      </footer>

      {/* Forward Pass Visualizer Modal */}
      {showVisualizer && prediction && (
        <ForwardPassVisualizer
          probabilities={prediction.probabilities}
          predictedIndex={prediction.predictedIndex}
          labels={digitLabels}
          mode="digit"
          imageGrid={prediction.imageGrid}
          onClose={() => setShowVisualizer(false)}
        />
      )}
    </div>
  );
}

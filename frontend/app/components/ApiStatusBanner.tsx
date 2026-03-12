"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ApiStatusBannerProps {
  onApiReady?: () => void;
}

export default function ApiStatusBanner({ onApiReady }: ApiStatusBannerProps) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [visible, setVisible] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

  const checkHealth = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);

      const res = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (res.ok) {
        const data = await res.json();
        if (data.digit_model_loaded || data.char_model_loaded) {
          setStatus("ready");
          onApiReady?.();
          // Auto-hide after 4 seconds
          setTimeout(() => setVisible(false), 4000);
          return;
        }
      }
      throw new Error("Models not loaded");
    } catch {
      if (retryCount < 3) {
        setRetryCount((c) => c + 1);
        // Retry after 5 seconds
        setTimeout(() => checkHealth(), 5000);
      } else {
        setStatus("error");
      }
    }
  }, [retryCount, onApiReady]);

  useEffect(() => {
    checkHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRetry = () => {
    setStatus("loading");
    setRetryCount(0);
    checkHealth();
  };

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="overflow-hidden"
        >
          <div
            className={`api-banner ${
              status === "ready"
                ? "api-banner-ready"
                : status === "error"
                ? "api-banner-error"
                : "api-banner-loading"
            }`}
          >
            <div className="max-w-6xl mx-auto px-4 py-2.5 flex items-center justify-center gap-3 text-sm">
              {status === "loading" && (
                <>
                  <span className="api-banner-pulse">⏳</span>
                  <span>
                    Warming up the AI model on Render — this may take ~30s on first
                    visit…
                  </span>
                  <div className="spinner w-4 h-4 !border-2"></div>
                </>
              )}
              {status === "ready" && (
                <>
                  <span>✅</span>
                  <span>AI model loaded — you&apos;re good to go!</span>
                </>
              )}
              {status === "error" && (
                <>
                  <span>⚠️</span>
                  <span>Could not reach the AI server.</span>
                  <button
                    onClick={handleRetry}
                    className="ml-2 px-3 py-1 rounded-md bg-white/10 hover:bg-white/20 transition-colors text-xs font-medium"
                  >
                    Retry
                  </button>
                </>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

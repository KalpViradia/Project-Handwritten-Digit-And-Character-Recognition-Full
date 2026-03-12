import { useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* ──────────────────────────── types ──────────────────────────── */
interface ForwardPassVisualizerProps {
  probabilities: number[];
  predictedIndex: number;
  labels: string[];
  mode: "digit" | "character";
  imageGrid?: number[][];
  onClose: () => void;
}

/* ──────────────────────────── palette ─────────────────────────── */
const COLORS = {
  bg: "#050508", // Very dark black
  indigo: "#6366f1",
  purple: "#8b5cf6",
  cyan: "#06b6d4",
  emerald: "#10b981",
  teal: "#14b8a6",
  white: "#f0f0f5",
  dimText: "#6b7280", // Gray-500 for better visibility
  highlight: "#22d3ee", // Cyan-400
};

/* ──────────────────────── utility helpers ─────────────────────── */
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function getPhase(t: number, start: number, end: number) {
  if (t < start) return 0;
  if (t > end) return 1;
  return (t - start) / (end - start);
}

function lerpColor(c1: string, c2: string, t: number) {
  if (t <= 0) return c1;
  if (t >= 1) return c2;
  return c2; // Simplified for this cinematic effect
}

// Removed generatePixelGrid fallback to enforce strict data requirements

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/* ──────────────────────── main component ─────────────────────── */
export default function ForwardPassVisualizer({
  probabilities,
  predictedIndex,
  labels,
  mode,
  imageGrid,
  onClose,
}: ForwardPassVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const startTimeRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  
  const accentColor = mode === "digit" ? COLORS.indigo : COLORS.emerald;
  const highlightColor = mode === "digit" ? COLORS.cyan : COLORS.teal;

  // Debug production missing image grid
  console.log("imageGrid length:", imageGrid?.length);

  // Validate imageGrid
  const isValidGrid = imageGrid && imageGrid.length === 28 && imageGrid[0]?.length === 28;
  const pixelGrid = imageGrid;

  if (!isValidGrid) {
    console.error("Visualizer Error: imageGrid is missing or invalid (expected 28x28 array).");
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const totalDuration = 10000; // 10s Cinematic Run

    const animate = (timestamp: number) => {
      if (!startTimeRef.current) startTimeRef.current = timestamp;
      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / totalDuration, 1);
      const time = timestamp / 1000;

      // Handle Resize & DPR
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (!rect) return;

      if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
      }

      const w = rect.width;
      const h = rect.height;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      // --- 7-PHASE TIMELINE ---
      // 1: 0.00 - 0.15 -> Input Image Only
      // 2: 0.15 - 0.30 -> Input layer nodes fade in
      // 3: 0.30 - 0.50 -> Connection growth In->Hid + Hidden nodes fade
      // 4: 0.50 - 0.65 -> Connection growth Hid->Out + Pulses
      // 5: 0.65 - 0.80 -> Dim weak (10-15%), highlight decision path
      // 6: 0.80 - 0.90 -> Output growth, glow, prob fade-in
      // 7: 0.90 - 1.00 -> Final composed layout + freeze

      const s1 = getPhase(progress, 0.00, 0.15);
      const s2 = getPhase(progress, 0.15, 0.30);
      const s3 = getPhase(progress, 0.30, 0.50);
      const s4 = getPhase(progress, 0.50, 0.65);
      const s5 = getPhase(progress, 0.65, 0.80);
      const s6 = getPhase(progress, 0.80, 0.90);
      const s7 = getPhase(progress, 0.90, 1.00);

      const centerY = h / 2;
      const layoutS2 = easeInOutCubic(s2);
      
      // X-Positions
      const inputX = lerp(w / 2, w * 0.15, layoutS2);
      const netX_in = lerp(inputX, w * 0.3, layoutS2);
      const netX_hid = w * 0.45;
      const netX_out = lerp(w * 0.5, w * 0.65, layoutS2);
      const imgScale = lerp(1.2, 0.7, layoutS2);

      // --- 1. Draw Input Image ---
      ctx.save();
      ctx.translate(inputX, centerY);
      ctx.scale(imgScale, imgScale);
      
      const cellSize = 5;
      const gridOffset = -(28 * cellSize) / 2;
      ctx.globalAlpha = Math.max(0.2, s1); // Ensure it's never fully invisible once animation starts
      
      if (isValidGrid && pixelGrid) {
        for (let r = 0; r < 28; r++) {
          for (let c = 0; c < 28; c++) {
            const val = pixelGrid[r][c]; 
            if (val > 0.1) {
              const intensity = Math.floor(lerp(40, 255, val)); // Boost low values for visibility
              ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
              ctx.fillRect(gridOffset + c * cellSize, gridOffset + r * cellSize, cellSize - 0.5, cellSize - 0.5);
            }
          }
        }
      } else {
        // Warning overlay if data is missing
        ctx.fillStyle = "rgba(255, 50, 50, 0.1)";
        ctx.fillRect(gridOffset, gridOffset, 28 * cellSize, 28 * cellSize);
        
        ctx.fillStyle = "#ff4444";
        ctx.font = "bold 10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("DATA", 0, -5);
        ctx.fillText("UNAVAILABLE", 0, 8);
      }
      
      // Image box
      ctx.strokeStyle = isValidGrid ? "rgba(255,255,255,0.1)" : "rgba(255, 50, 50, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(gridOffset - 4, gridOffset - 4, 28 * cellSize + 8, 28 * cellSize + 8);
      ctx.restore();

      // --- 2. THE NETWORK ---
      if (s2 > 0) {
        const nodeR = 6;
        const numIn = 12;
        const numHid = 10;
        const numOut = probabilities.length;

        const inSpace = (h - 160) / (numIn - 1);
        const hidSpace = (h - 200) / (numHid - 1);
        const outSpace = Math.min(40, (h - 160) / (numOut - 1));

        const inStart = centerY - ((numIn - 1) * inSpace) / 2;
        const hidStart = centerY - ((numHid - 1) * hidSpace) / 2;
        const outStart = centerY - ((numOut - 1) * outSpace) / 2;

        // Connections & Pulses
        // Phase 5 Highlight: 0.65 - 0.8
        const highlightPhase = s5;
        const dimOpacity = 0.12; // 10-15% rule

        // Input -> Hidden
        for (let i = 0; i < numIn; i++) {
          const iy = inStart + i * inSpace;
          for (let j = 0; j < numHid; j++) {
            const hy = hidStart + j * hidSpace;
            const growth = getPhase(progress, 0.3 + (i / numIn) * 0.1, 0.45 + (j / numHid) * 0.05);
            
            if (growth > 0) {
              const isCrit = (i % 3 === 0 && j % 2 === 0);
              const alpha = lerp(0.3, isCrit ? 1.0 : dimOpacity, highlightPhase);
              
              ctx.save();
              ctx.globalAlpha = s2 * alpha * (progress < 0.5 ? growth : 1);
              ctx.strokeStyle = isCrit && highlightPhase > 0 ? highlightColor : accentColor;
              ctx.lineWidth = 1 + (isCrit ? highlightPhase * 1.5 : 0);
              
              if (isCrit && highlightPhase > 0) {
                ctx.shadowBlur = 8 * highlightPhase;
                ctx.shadowColor = highlightColor;
              }
              
              ctx.beginPath();
              ctx.moveTo(netX_in, iy);
              ctx.lineTo(lerp(netX_in, netX_hid, growth), lerp(iy, hy, growth));
              ctx.stroke();
              ctx.restore();

              // Pulses (Phase 4: 0.5 - 0.65)
              if (progress >= 0.5) {
                const pulseT = (time * 1.5 + i * 0.1) % 1;
                ctx.fillStyle = "#fff";
                ctx.globalAlpha = s4 * (isCrit ? 1.0 : 0.2);
                ctx.beginPath();
                ctx.arc(lerp(netX_in, netX_hid, pulseT), lerp(iy, hy, pulseT), 1.5, 0, Math.PI * 2);
                ctx.fill();
              }
            }
          }
          // Input Nodes (Fade in Phase 2)
          ctx.beginPath();
          ctx.arc(netX_in, iy, nodeR * s2, 0, Math.PI * 2);
          ctx.fillStyle = accentColor;
          ctx.globalAlpha = s2 * 0.6;
          ctx.fill();
          
          ctx.strokeStyle = "rgba(255,255,255,0.15)";
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }

        // Hidden Layer
        for (let j = 0; j < numHid; j++) {
          const hy = hidStart + j * hidSpace;
          const nodeFade = getPhase(progress, 0.35, 0.45);
          if (nodeFade > 0) {
            ctx.beginPath();
            ctx.arc(netX_hid, hy, nodeR * nodeFade, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.purple;
            ctx.globalAlpha = s2 * 0.8;
            ctx.fill();
            
            ctx.strokeStyle = "rgba(255,255,255,0.15)";
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Hidden -> Output
            for (let k = 0; k < numOut; k++) {
              const oy = outStart + k * outSpace;
              const isTop = k === predictedIndex;
              const isCrit = (j % 2 === 0 && isTop);
              const growth = getPhase(progress, 0.5 + (j / numHid) * 0.1, 0.65 + (k / numOut) * 0.05);

              if (growth > 0) {
                const alpha = lerp(0.3, isCrit ? 1.0 : dimOpacity, highlightPhase);
                ctx.save();
                ctx.globalAlpha = s2 * alpha * (progress < 0.65 ? growth : 1);
                ctx.strokeStyle = isCrit && highlightPhase > 0 ? highlightColor : COLORS.cyan;
                ctx.lineWidth = 1 + (isCrit ? highlightPhase * 2 : 0);
                
                if (isCrit && highlightPhase > 0) {
                  ctx.shadowBlur = 12 * highlightPhase;
                  ctx.shadowColor = highlightColor;
                }

                ctx.beginPath();
                ctx.moveTo(netX_hid, hy);
                ctx.lineTo(lerp(netX_hid, netX_out, growth), lerp(hy, oy, growth));
                ctx.stroke();
                ctx.restore();

                if (progress >= 0.6) {
                  const pulseT = (time * 2 + j * 0.2) % 1;
                  ctx.fillStyle = "#fff";
                  ctx.globalAlpha = s4 * (isCrit ? 1.0 : 0.1);
                  ctx.beginPath();
                  ctx.arc(lerp(netX_hid, netX_out, pulseT), lerp(hy, oy, pulseT), 1.5, 0, Math.PI * 2);
                  ctx.fill();
                }
              }
            }
          }
        }

        // Output Layer (Phase 6 growth: 0.8 - 0.9)
        for (let k = 0; k < numOut; k++) {
          const oy = outStart + k * outSpace;
          const isTop = k === predictedIndex;
          const nodeFade = getPhase(progress, 0.55, 0.65);
          
          if (nodeFade > 0) {
            const growthS6 = s6;
            const currentR = (isTop ? nodeR + 3 * growthS6 : nodeR) * nodeFade;
            
            ctx.save();
            ctx.beginPath();
            ctx.arc(netX_out, oy, currentR, 0, Math.PI * 2);
            ctx.fillStyle = isTop ? highlightColor : COLORS.dimText;
            // Increased alpha for non-top nodes (0.5 -> 0.3 instead of 0.5 -> 0.1)
            ctx.globalAlpha = s2 * (isTop ? 1.0 : lerp(0.6, 0.4, highlightPhase));
            
            if (isTop && growthS6 > 0) {
              ctx.shadowBlur = 15 * growthS6;
              ctx.shadowColor = highlightColor;
            }
            
            ctx.fill();
            
            // Subtle border for all nodes to make them visible
            ctx.strokeStyle = isTop ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.2)";
            ctx.lineWidth = 0.8;
            ctx.stroke();
            
            ctx.restore();

            // --- ANCHORED LABEL (Phase 6) ---
            if (isTop && progress >= 0.8) {
              const reveal = s6;
              const prob = probabilities[k];
              const anchorX = netX_out + 15;
              const labelX = anchorX + 120; // 120px offset

              // Dotted Line
              ctx.save();
              ctx.beginPath();
              ctx.setLineDash([2, 3]);
              ctx.moveTo(anchorX, oy);
              ctx.lineTo(lerp(anchorX, labelX, reveal), oy);
              ctx.strokeStyle = "rgba(255,255,255,0.3)";
              ctx.lineWidth = 1;
              ctx.stroke();
              ctx.restore();

              // Text Hierarchy
              if (reveal > 0.5) {
                ctx.save();
                ctx.globalAlpha = getPhase(reveal, 0.5, 1);
                ctx.textAlign = "left";
                
                // Big Digit
                ctx.font = "bold 32px Inter, sans-serif";
                ctx.fillStyle = highlightColor;
                ctx.fillText(labels[k], labelX + 10, oy + 10);
                
                // Medium %
                const digitW = ctx.measureText(labels[k]).width;
                ctx.font = "500 18px Inter, sans-serif";
                ctx.fillStyle = "#fff";
                ctx.fillText(`${(prob * 100).toFixed(1)}%`, labelX + 10 + digitW + 15, oy + 6);
                
                // Small Confidence
                ctx.font = "uppercase 10px Inter, sans-serif";
                ctx.fillStyle = "rgba(255,255,255,0.4)";
                ctx.letterSpacing = "0.1em";
                ctx.fillText("Confidence", labelX + 10 + digitW + 15, oy + 22);
                
                ctx.restore();
              }
            } else if (progress >= 0.85) {
              // Secondary Prob Bars
              const reveal = getPhase(progress, 0.85, 0.95);
              const prob = probabilities[k];
              const barStartX = netX_out + 15;
              const barMaxW = 40;
              const barW = prob * barMaxW * reveal;
              
              ctx.fillStyle = COLORS.indigo;
              ctx.globalAlpha = 0.3 * reveal; // Increased from 0.15
              roundRect(ctx, barStartX, oy - 2, barW, 4, 2);
              ctx.fill();
            }
          }
        }
      }

      ctx.restore();

      // Stop Condition
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        cancelAnimationFrame(rafRef.current!);
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [probabilities, predictedIndex, labels, accentColor, highlightColor, pixelGrid]);

  const handleClose = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    onClose();
  }, [onClose]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-black/98 backdrop-blur-3xl p-4"
        onClick={handleClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="relative w-full max-w-7xl aspect-video bg-[#020205] rounded-[2.5rem] border border-white/5 overflow-hidden shadow-[0_0_100px_rgba(0,0,0,0.9)]"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header Layout: Flexbox justify-between */}
          <header className="absolute top-0 left-0 right-0 z-[130] flex items-center justify-between px-12 py-10 pointer-events-none">
            <div className="flex items-center gap-4">
              <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse shadow-[0_0_8px_#22d3ee]" />
              <span className="text-white/40 text-[10px] uppercase tracking-[0.4em] font-bold">
                Neural Path Inference View
              </span>
            </div>

            <button
              onClick={handleClose}
              className="p-3 text-white/30 hover:text-white hover:bg-white/10 rounded-full transition-all pointer-events-auto"
              aria-label="Exit Analysis"
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </header>

          <canvas ref={canvasRef} className="block w-full h-full" />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

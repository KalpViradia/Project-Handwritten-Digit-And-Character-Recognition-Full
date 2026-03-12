"use client";

import { useRef, useState, useEffect, forwardRef, useImperativeHandle } from "react";

interface DrawingCanvasProps {
    width?: number;
    height?: number;
}

export interface DrawingCanvasRef {
    getImageData: () => string;
    clearCanvas: () => void;
}

const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(
    ({ width = 280, height = 280 }, ref) => {
        const canvasRef = useRef<HTMLCanvasElement>(null);
        const [isDrawing, setIsDrawing] = useState(false);
        const [hasDrawn, setHasDrawn] = useState(false);

        // Initialize canvas with black background
        useEffect(() => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            // Set black background
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, width, height);

            // Set drawing style
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 18;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
        }, [width, height]);

        // Expose methods to parent
        useImperativeHandle(ref, () => ({
            getImageData: () => {
                const canvas = canvasRef.current;
                if (!canvas) return "";
                return canvas.toDataURL("image/png");
            },
            clearCanvas: () => {
                const canvas = canvasRef.current;
                if (!canvas) return;

                const ctx = canvas.getContext("2d");
                if (!ctx) return;

                // Clear with black
                ctx.fillStyle = "#000000";
                ctx.fillRect(0, 0, width, height);
                setHasDrawn(false);
            }
        }));

        const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
            const canvas = canvasRef.current;
            if (!canvas) return { x: 0, y: 0 };

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            if ("touches" in e) {
                return {
                    x: (e.touches[0].clientX - rect.left) * scaleX,
                    y: (e.touches[0].clientY - rect.top) * scaleY
                };
            }
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        };

        const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
            e.preventDefault();
            const canvas = canvasRef.current;
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const { x, y } = getCoordinates(e);
            ctx.beginPath();
            ctx.moveTo(x, y);
            setIsDrawing(true);
        };

        const draw = (e: React.MouseEvent | React.TouchEvent) => {
            e.preventDefault();
            if (!isDrawing) return;

            const canvas = canvasRef.current;
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const { x, y } = getCoordinates(e);
            ctx.lineTo(x, y);
            ctx.stroke();
            setHasDrawn(true);
        };

        const stopDrawing = () => {
            setIsDrawing(false);
        };

        return (
            <div className="canvas-container mx-auto" style={{ width: width, height: height }}>
                <canvas
                    ref={canvasRef}
                    width={width}
                    height={height}
                    className="drawing-canvas w-full h-full"
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                />

                {/* Hint overlay */}
                {!hasDrawn && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="text-gray-500 text-center">
                            <p className="text-4xl mb-2">✏️</p>
                            <p>Draw a digit here</p>
                        </div>
                    </div>
                )}
            </div>
        );
    }
);

DrawingCanvas.displayName = "DrawingCanvas";

export default DrawingCanvas;

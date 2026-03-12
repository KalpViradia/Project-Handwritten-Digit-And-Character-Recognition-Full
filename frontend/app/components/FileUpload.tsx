"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    isLoading: boolean;
}

export default function FileUpload({ onFileSelect, isLoading }: FileUploadProps) {
    const [isDragOver, setIsDragOver] = useState(false);
    const [preview, setPreview] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFile = useCallback((file: File) => {
        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file");
            return;
        }

        setFileName(file.name);

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);

        onFileSelect(file);
    }, [onFileSelect]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, [handleFile]);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = () => {
        setIsDragOver(false);
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    };

    const clearPreview = () => {
        setPreview(null);
        setFileName(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <div className="space-y-4">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleChange}
                accept="image/*"
                className="hidden"
            />

            {preview ? (
                <div className="text-center space-y-4">
                    <div className="inline-block p-4 bg-gray-900 rounded-lg">
                        <img
                            src={preview}
                            alt="Uploaded digit"
                            className="max-w-[200px] max-h-[200px] object-contain mx-auto"
                        />
                    </div>
                    <p className="text-gray-400 text-sm">{fileName}</p>

                    <div className="flex gap-4 justify-center">
                        <button
                            className="btn-primary"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <span className="flex items-center gap-2">
                                    <span className="spinner w-5 h-5"></span>
                                    Processing...
                                </span>
                            ) : (
                                "Upload Another"
                            )}
                        </button>
                        <button
                            className="btn-secondary"
                            onClick={clearPreview}
                        >
                            Clear
                        </button>
                    </div>
                </div>
            ) : (
                <div
                    className={`upload-area ${isDragOver ? "dragover" : ""}`}
                    onClick={handleClick}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    <div className="text-4xl mb-4">📤</div>
                    <p className="text-gray-300 font-medium mb-2">
                        Drag & drop an image here
                    </p>
                    <p className="text-gray-500 text-sm">
                        or click to browse files
                    </p>
                    <p className="text-gray-600 text-xs mt-4">
                        Supports PNG, JPG, JPEG
                    </p>
                </div>
            )}
        </div>
    );
}

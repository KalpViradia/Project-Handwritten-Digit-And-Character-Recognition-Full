import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Digit Recognizer | AI-Powered Handwriting Recognition",
  description: "Draw or upload handwritten digits and let our CNN model predict them with high accuracy. Built with TensorFlow and Next.js.",
  keywords: ["digit recognition", "CNN", "machine learning", "handwriting", "MNIST"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-animated min-h-screen antialiased">
        {/* Animated background particles */}
        <div className="particles" aria-hidden="true">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 20}s`,
                animationDuration: `${15 + Math.random() * 10}s`
              }}
            />
          ))}
        </div>

        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}

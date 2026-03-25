import "./globals.css";

export const metadata = {
  title: "Color Predictor — AI-Powered Color Taste",
  description:
    "Train a lightweight neural network in your browser to predict which colors you love. No data ever leaves your device.",
  keywords: ["color predictor", "machine learning", "tensorflow.js", "color preferences", "AI"],
  authors: [{ name: "Color Predictor" }],
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f0f2f8" },
    { media: "(prefers-color-scheme: dark)", color: "#080c18" },
  ],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}

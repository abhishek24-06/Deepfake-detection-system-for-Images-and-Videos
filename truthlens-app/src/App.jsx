import React, { useState } from 'react';
import { Upload, Shield, Zap, Eye, CheckCircle, AlertTriangle, Home, Info, Mail, Lock } from 'lucide-react';

const TruthLensApp = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (uploadedFile) => {
    setFile(uploadedFile);
    setResult(null);
  };

  const analyzeFile = () => {
    if (!file) return;
    
    setAnalyzing(true);
    setResult(null);
    
    // Simulate analysis
    setTimeout(() => {
      const isDeepfake = Math.random() > 0.5;
      const confidence = Math.floor(Math.random() * 20) + 80;
      
      setResult({
        isDeepfake,
        confidence,
        fileName: file.name
      });
      setAnalyzing(false);
    }, 3000);
  };

  const resetDetector = () => {
    setFile(null);
    setResult(null);
    setAnalyzing(false);
  };

  // Homepage Component
  const HomePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Navigation */}
      <nav className="bg-slate-900/50 backdrop-blur-lg border-b border-blue-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">TruthLens</span>
          </div>
          <button 
            onClick={() => setCurrentPage('detector')}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            Try Detector
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-6 pt-20 pb-32">
        <div className="text-center">
          <div className="inline-block mb-6">
            <div className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-blue-300">AI-Powered Detection</span>
            </div>
          </div>
          
          <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
            Detect Deepfakes<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              in Seconds
            </span>
          </h1>
          
          <p className="text-xl text-slate-300 mb-12 max-w-2xl mx-auto">
            AI-powered tool to verify authenticity of videos and images. 
            Protect yourself from digital misinformation.
          </p>
          
          <button 
            onClick={() => setCurrentPage('detector')}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-xl font-semibold text-lg shadow-lg shadow-blue-500/50 transition-all transform hover:scale-105"
          >
            Try Detector Now
          </button>
        </div>

        {/* Visual Element */}
        <div className="mt-20 relative">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 blur-3xl"></div>
          <div className="relative bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-12">
            <div className="grid grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-white font-semibold mb-2">Upload</h3>
                <p className="text-slate-400 text-sm">Drag and drop or select your media file</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-cyan-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-8 h-8 text-cyan-400" />
                </div>
                <h3 className="text-white font-semibold mb-2">Analyze</h3>
                <p className="text-slate-400 text-sm">AI scans for manipulation patterns</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-white font-semibold mb-2">Get Result</h3>
                <p className="text-slate-400 text-sm">Instant verdict with confidence score</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Why It Matters Section */}
      <div className="bg-slate-900/50 backdrop-blur-lg border-y border-blue-500/20 py-20">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-white text-center mb-12">Why It Matters</h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-slate-800/50 border border-blue-500/30 rounded-xl p-6">
              <Shield className="w-10 h-10 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Combat Misinformation</h3>
              <p className="text-slate-300">
                Deepfakes are increasingly used to spread false information. Our tool helps you verify authenticity and make informed decisions.
              </p>
            </div>
            <div className="bg-slate-800/50 border border-blue-500/30 rounded-xl p-6">
              <Lock className="w-10 h-10 text-cyan-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Digital Safety</h3>
              <p className="text-slate-300">
                Protect yourself and others from identity theft, fraud, and manipulation through advanced AI detection technology.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900/80 backdrop-blur-lg py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-2">
              <Shield className="w-6 h-6 text-blue-400" />
              <span className="text-xl font-bold text-white">TruthLens</span>
            </div>
            <div className="flex gap-8">
              <a href="#about" className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
                <Info className="w-4 h-4" />
                About
              </a>
              <a href="#contact" className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
                <Mail className="w-4 h-4" />
                Contact
              </a>
              <a href="#privacy" className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
                <Lock className="w-4 h-4" />
                Privacy
              </a>
            </div>
          </div>
          <div className="text-center mt-8 text-slate-500 text-sm">
            © 2025 TruthLens. Powered by advanced AI detection.
          </div>
        </div>
      </footer>
    </div>
  );

  // Detector Page Component
  const DetectorPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Navigation */}
      <nav className="bg-slate-900/50 backdrop-blur-lg border-b border-blue-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-8 h-8 text-blue-400" />
            <span className="text-2xl font-bold text-white">TruthLens</span>
          </div>
          <button 
            onClick={() => {
              setCurrentPage('home');
              resetDetector();
            }}
            className="flex items-center gap-2 px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors border border-blue-500/30"
          >
            <Home className="w-4 h-4" />
            Home
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">Deepfake Detector</h1>
          <p className="text-xl text-slate-300">Upload an image or video to analyze its authenticity</p>
        </div>

        {/* Upload Area */}
        {!result && (
          <div className="bg-slate-800/50 backdrop-blur-xl border-2 border-dashed border-blue-500/50 rounded-2xl p-12 mb-8">
            <div
              className={`transition-all ${dragActive ? 'scale-105' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="fileInput"
                className="hidden"
                accept="image/*,video/*"
                onChange={handleFileInput}
              />
              
              <label
                htmlFor="fileInput"
                className="flex flex-col items-center cursor-pointer"
              >
                <div className={`w-24 h-24 rounded-full flex items-center justify-center mb-6 transition-colors ${
                  dragActive ? 'bg-blue-500/30' : 'bg-blue-500/10'
                }`}>
                  <Upload className="w-12 h-12 text-blue-400" />
                </div>
                
                {file ? (
                  <div className="text-center">
                    <p className="text-white font-semibold mb-2">{file.name}</p>
                    <p className="text-slate-400 text-sm mb-4">File ready for analysis</p>
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        resetDetector();
                      }}
                      className="text-blue-400 hover:text-blue-300 text-sm"
                    >
                      Choose different file
                    </button>
                  </div>
                ) : (
                  <div className="text-center">
                    <p className="text-white font-semibold mb-2">Drop your file here</p>
                    <p className="text-slate-400 mb-4">or click to browse</p>
                    <p className="text-slate-500 text-sm">Supports images and videos</p>
                  </div>
                )}
              </label>
            </div>
          </div>
        )}

        {/* Analyze Button */}
        {file && !analyzing && !result && (
          <div className="text-center mb-8">
            <button
              onClick={analyzeFile}
              className="px-12 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-xl font-semibold text-lg shadow-lg shadow-blue-500/50 transition-all transform hover:scale-105"
            >
              Analyze Now
            </button>
          </div>
        )}

        {/* Loading State */}
        {analyzing && (
          <div className="bg-slate-800/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-12">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-6"></div>
              <h3 className="text-2xl font-semibold text-white mb-2">Analyzing...</h3>
              <p className="text-slate-400">Our AI is examining your media for signs of manipulation</p>
              
              {/* Progress Bar */}
              <div className="mt-8 max-w-md mx-auto">
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full animate-pulse" style={{width: '75%'}}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            <div className={`bg-slate-800/50 backdrop-blur-xl border-2 rounded-2xl p-8 ${
              result.isDeepfake ? 'border-red-500/50' : 'border-green-500/50'
            }`}>
              <div className="text-center mb-6">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4 ${
                  result.isDeepfake ? 'bg-red-500/20' : 'bg-green-500/20'
                }`}>
                  {result.isDeepfake ? (
                    <AlertTriangle className="w-10 h-10 text-red-400" />
                  ) : (
                    <CheckCircle className="w-10 h-10 text-green-400" />
                  )}
                </div>
                
                <h2 className={`text-3xl font-bold mb-2 ${
                  result.isDeepfake ? 'text-red-400' : 'text-green-400'
                }`}>
                  {result.isDeepfake ? 'Deepfake Detected' : 'Authentic'}
                </h2>
                
                <p className="text-slate-300 mb-6">
                  {result.isDeepfake 
                    ? 'This media shows signs of artificial manipulation'
                    : 'This media appears to be genuine'}
                </p>

                {/* Confidence Score */}
                <div className="max-w-xs mx-auto">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-400 text-sm">Confidence</span>
                    <span className="text-white font-bold text-lg">{result.confidence}%</span>
                  </div>
                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${
                        result.isDeepfake 
                          ? 'bg-gradient-to-r from-red-500 to-orange-500' 
                          : 'bg-gradient-to-r from-green-500 to-emerald-500'
                      }`}
                      style={{width: `${result.confidence}%`}}
                    ></div>
                  </div>
                </div>
              </div>

              {/* File Info */}
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                <p className="text-slate-400 text-sm mb-1">Analyzed File</p>
                <p className="text-white font-medium">{result.fileName}</p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 justify-center">
              <button
                onClick={resetDetector}
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
              >
                Analyze Another File
              </button>
              <button
                onClick={() => {
                  setCurrentPage('home');
                  resetDetector();
                }}
                className="px-8 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
              >
                Back to Home
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return currentPage === 'home' ? <HomePage /> : <DetectorPage />;
};

export default TruthLensApp;
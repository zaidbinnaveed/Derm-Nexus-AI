"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Region = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  label: string;
  risk: "low" | "medium" | "high";
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type Prediction = {
  class: string;
  displayName: string;
  probability: number;
  risk: "low" | "medium" | "high";
  info: string;
};

type ModelStats = {
  overallAccuracy: string;
  macroPrecision: string;
  macroRecall: string;
  macroF1: string;
  rocAuc: string;
  sensitivity: string;
  specificity: string;
  balancedAccuracy: string;
  mcc: string;
  top1: string;
  top3: string;
  calibrationError: string;
  inferenceLatency: string;
};

const modelStats: ModelStats = {
  overallAccuracy: "91.8%",
  macroPrecision: "90.4%",
  macroRecall: "89.7%",
  macroF1: "89.9%",
  rocAuc: "0.956",
  sensitivity: "88.9%",
  specificity: "93.5%",
  balancedAccuracy: "91.2%",
  mcc: "0.83",
  top1: "91.8%",
  top3: "98.1%",
  calibrationError: "0.041",
  inferenceLatency: "~420 ms/image",
};

const riskCopy: Record<Region["risk"], { label: string; advice: string }> = {
  low: {
    label: "Low concern",
    advice: "Usually safe to monitor. Re-check with a fresh photo in 4 weeks or sooner if it changes.",
  },
  medium: {
    label: "Needs review",
    advice: "Not an emergency, but book a dermatologist appointment in 1-2 weeks for professional assessment.",
  },
  high: {
    label: "High priority review",
    advice: "Please arrange a dermatologist visit as soon as possible. Go urgent if bleeding, painful, or rapidly changing.",
  },
};

const quickPrompts = [
  "What does this result mean in simple terms?",
  "How serious is this and how urgent?",
  "What should I do today and this week?",
  "Why did the AI flag this specific area?",
];

async function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function enhanceImage(dataUrl: string) {
  return new Promise<string>((resolve) => {
    const img = new window.Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(dataUrl); return; }
      ctx.filter = "contrast(1.12) saturate(1.06) brightness(1.03)";
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL("image/jpeg", 0.96));
    };
    img.src = dataUrl;
  });
}

export default function Home() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [enhancedImage, setEnhancedImage] = useState<string | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [activeRegion, setActiveRegion] = useState<Region | null>(null);
  const [allPredictions, setAllPredictions] = useState<Prediction[]>([]);
  const [topPrediction, setTopPrediction] = useState<Prediction | null>(null);
  const [loadingStage, setLoadingStage] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setChatLoading] = useState(false);
  const [chat, setChat] = useState<ChatMessage[]>([
    { role: "assistant", content: "I am here with you. Upload a skin image and I will explain findings in calm, clear language and guide you on practical next steps." },
  ]);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [backendError, setBackendError] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  const hasAnalysis = regions.length > 0;
  const activeRisk = activeRegion ? riskCopy[activeRegion.risk] : null;

  const neuralNodes = useMemo(
    () => Array.from({ length: 24 }, (_, idx) => ({
      id: idx,
      left: `${6 + ((idx * 37) % 90)}%`,
      top: `${4 + ((idx * 29) % 90)}%`,
      delay: `${(idx % 9) * 0.45}s`,
    })),
    []
  );

  useEffect(() => {
    const container = chatScrollRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
    setShowScrollDown(false);
  }, [chat, isChatLoading]);

  const handleUpload = async (file: File) => {
    setBackendError(false);
    setRegions([]);
    setAllPredictions([]);
    setTopPrediction(null);

    const data = await fileToDataUrl(file);
    setOriginalImage(data);
    setLoadingStage("Acquiring dermoscopic signals...");

    const enhanced = await enhanceImage(data);
    setEnhancedImage(enhanced);
    setLoadingStage("Enhancing texture and vascular contrast...");

    await new Promise((res) => setTimeout(res, 800));
    setLoadingStage("Running lesion intelligence inference...");

    try {
      // Send image to Python Flask backend
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("https://dermai-nexus-api-production.up.railway.app/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Backend error");

      const result = await response.json();

      setRegions(result.regions);
      setAllPredictions(result.allPredictions);
      setTopPrediction(result.topPrediction);
      setActiveRegion(result.regions[0]);

      setChat((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Analysis complete. The AI identified this as most likely ${result.topPrediction.displayName} with ${result.topPrediction.probability}% confidence. Risk level: ${result.topPrediction.risk}. ${result.topPrediction.info} Tap a highlighted region or ask me any question.`,
        },
      ]);
    } catch {
      setBackendError(true);
      setLoadingStage(null);
      setChat((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Could not connect to the analysis server. Make sure the Python backend is running on port 5000. Run: python app/app.py",
        },
      ]);
      return;
    }

    setLoadingStage(null);
  };

  const onDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) await handleUpload(file);
  };

  const submitChat = async (message: string, source: "manual" | "quick") => {
    const userMessage = message.trim();
    if (!userMessage || isChatLoading) return;
    if (source === "manual") setChatInput("");

    const updatedChat: ChatMessage[] = [...chat, { role: "user", content: userMessage }];
    setChat(updatedChat);
    setChatLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedChat,
          activeRegion,
          regions,
          topPrediction,
          allPredictions,
          hasImage: Boolean(originalImage),
        }),
      });
      const data = await response.json();
      const answer = data?.reply || "I could not generate a response right now.";
      setChat((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch {
      setChat((prev) => [...prev, {
        role: "assistant",
        content: "Connection issue. If this lesion is changing rapidly or bleeding, seek urgent in-person care.",
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChat = async (event: FormEvent) => {
    event.preventDefault();
    await submitChat(chatInput, "manual");
  };

  return (
    <div
      className="hero-bg medical-grid min-h-screen"
      onMouseMove={(e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 2;
        const y = (e.clientY / window.innerHeight - 0.5) * 2;
        setTilt({ x, y });
      }}
      style={{ "--tilt-x": `${tilt.x * 8}deg`, "--tilt-y": `${tilt.y * -8}deg`, "--shift-x": `${tilt.x * 18}px`, "--shift-y": `${tilt.y * 18}px` } as React.CSSProperties}
    >
      <div className="neural-layer" aria-hidden="true">
        {neuralNodes.map((node) => (
          <span key={node.id} className="neural-node" style={{ left: node.left, top: node.top, animationDelay: node.delay }} />
        ))}
      </div>

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-12 px-4 py-6 md:px-8 md:py-10">
        <section className="glass-panel parallax-panel relative overflow-hidden rounded-3xl p-8 md:p-12">
          <div className="pointer-events-none absolute -right-32 -top-32 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(200,16,46,0.35),transparent_65%)]" />
          <div className="hero-hologram pointer-events-none absolute inset-0" />
          <div className="mb-4 flex items-center gap-3">
            <div className="logo-orb" />
            <div>
              <p className="inline-flex rounded-full border border-red-900/60 bg-red-900/30 px-4 py-1 text-xs uppercase tracking-[0.25em] text-red-100">Derm AI Nexus</p>
              <p className="mt-1 text-xs uppercase tracking-[0.23em] text-cyan-200/70">Neural Dermatology Suite</p>
            </div>
          </div>
          <h1 className="font-serif text-3xl leading-tight md:text-6xl hero-title">
            Seeing Beyond the Surface — Early Detection. Lifesaving Precision.
          </h1>
          <p className="mt-5 max-w-3xl text-zinc-300">
            Cinematic-grade lesion intelligence with compassionate guidance for faster, safer dermatology pathways.
          </p>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="glass-panel parallax-panel rounded-3xl border border-red-900/40 p-5 md:p-7">
            <h2 className="font-serif text-2xl">Image Upload & Analysis Lab</h2>
            <p className="mt-2 text-sm text-zinc-300">Drop a lesion photo or click to upload. The AI model analyzes it and returns real predictions.</p>

            {backendError && (
              <div className="mt-4 rounded-xl border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-200">
                ⚠️ Python backend not running. Open a new terminal and run:<br />
                <code className="mt-1 block text-xs text-red-100">cd "D:\ANN Project\ANN Project" && python app/app.py</code>
              </div>
            )}

            <div
              className="mt-6 rounded-2xl border border-dashed border-red-700/60 bg-gradient-to-b from-[#1a0b0e] to-black/80 p-8 text-center"
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
            >
              <input type="file" id="uploader" accept="image/*" className="hidden"
                onChange={async (e) => { const file = e.target.files?.[0]; if (file) await handleUpload(file); }} />
              <label htmlFor="uploader" className="cursor-pointer">
                <span className="block text-lg text-red-100">Drag and drop image or click to upload</span>
                <span className="mt-2 block text-sm text-zinc-400">Supports JPG, PNG, HEIC up to 20MB</span>
              </label>
            </div>

            {loadingStage && (
              <div className="animated-dna mt-6 rounded-xl border border-cyan-400/25 bg-cyan-500/10 p-4">
                <p className="text-sm text-cyan-100">{loadingStage}</p>
              </div>
            )}

            <div className="mt-6 grid gap-4 lg:grid-cols-3">
              {[
                { title: "Original", image: originalImage, annotate: false },
                { title: "Enhanced", image: enhancedImage, annotate: false },
                { title: "Annotated", image: enhancedImage ?? originalImage, annotate: true },
              ].map((panel) => (
                <div key={panel.title} className="parallax-card overflow-hidden rounded-xl border border-red-950/70 bg-black/60">
                  <p className="border-b border-red-900/40 px-3 py-2 text-xs uppercase tracking-[0.18em] text-zinc-200">{panel.title}</p>
                  <div className="relative aspect-[4/3] bg-zinc-900/70">
                    {panel.image ? (
                      <>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={panel.image} alt={`${panel.title} lesion`} className="h-full w-full object-cover" />
                        {panel.annotate && regions.map((region) => (
                          <button key={region.id} type="button"
                            onClick={() => {
                              setActiveRegion(region);
                              setChat((prev) => [...prev, {
                                role: "assistant",
                                content: `Region ${region.id.toUpperCase()}: ${region.label} — ${region.confidence}% confidence. Risk: ${riskCopy[region.risk].label}. ${riskCopy[region.risk].advice}`,
                              }]);
                            }}
                            className={`pulse absolute rounded-xl border-2 border-[#ff304f] bg-red-600/20 ${activeRegion?.id === region.id ? "shadow-[0_0_36px_rgba(200,16,46,0.86)]" : ""}`}
                            style={{ left: `${region.x}%`, top: `${region.y}%`, width: `${region.width}%`, height: `${region.height}%` }}
                          />
                        ))}
                      </>
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-zinc-500">Awaiting image</div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Real predictions from model */}
            {allPredictions.length > 0 && (
              <div className="mt-6 rounded-xl border border-amber-300/25 bg-amber-950/20 p-4 text-sm text-amber-50">
                <p className="text-xs uppercase tracking-[0.18em] text-amber-200/80 mb-3">
                  AI Model Predictions (Not Final Diagnosis)
                </p>
                <div className="space-y-3">
                  {allPredictions.map((pred) => (
                    <div key={pred.class}>
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{pred.displayName}</span>
                        <span className="rounded-full border border-amber-300/40 px-2 py-0.5 text-xs">{pred.probability}%</span>
                      </div>
                      <div className="mt-1 h-1.5 w-full rounded-full bg-amber-950/40">
                        <div className="h-1.5 rounded-full bg-amber-400/70" style={{ width: `${pred.probability}%` }} />
                      </div>
                      <p className="mt-1 text-xs text-amber-100/80">{pred.info}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-xs text-amber-100/60">These are AI-supported possibilities. Always consult a dermatologist.</p>
              </div>
            )}
          </div>

          {/* Chat panel */}
          <aside className="glass-panel parallax-panel relative flex h-full min-h-[680px] flex-col overflow-hidden rounded-3xl border border-cyan-400/20 p-4 md:p-6">
            <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(95,209,230,0.25),transparent_62%)]" />
            <div className="mb-2 flex items-center gap-3">
              <div className="eve-shell eve-float flex h-14 w-14 items-center justify-center rounded-full">
                <div className="flex gap-2"><span className="eve-eye" /><span className="eve-eye" /></div>
              </div>
              <div>
                <h3 className="font-serif text-2xl">EVE Clinical AI</h3>
                <p className="text-xs uppercase tracking-[0.2em] text-cyan-200/70">Calm. Precise. Empathetic.</p>
              </div>
            </div>

            <div
              ref={chatScrollRef}
              className="scanline mt-4 h-[320px] space-y-3 overflow-y-auto rounded-2xl border border-cyan-400/20 bg-gradient-to-b from-[#0b1114] to-[#060809] p-4"
              onScroll={(e) => {
                const t = e.currentTarget;
                setShowScrollDown(t.scrollHeight - t.scrollTop - t.clientHeight > 40);
              }}
            >
              {chat.map((message, idx) => (
                <div key={idx} className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm ${
                  message.role === "assistant"
                    ? "border border-cyan-300/20 bg-cyan-950/30 text-cyan-50"
                    : "ml-auto border border-red-500/30 bg-gradient-to-r from-[#580009] to-[#8b0000] text-red-50"
                }`}>
                  {message.content}
                </div>
              ))}
              {isChatLoading && (
                <div className="w-fit rounded-2xl border border-cyan-300/20 bg-cyan-950/35 px-4 py-3 text-sm text-cyan-100">
                  EVE AI is composing guidance...
                </div>
              )}
            </div>

            {showScrollDown && (
              <button type="button" className="mt-2 self-end rounded-full border border-cyan-300/30 bg-cyan-950/35 px-3 py-1 text-xs text-cyan-100"
                onClick={() => chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: "smooth" })}>
                Scroll to latest
              </button>
            )}

            {hasAnalysis && activeRegion && activeRisk && (
              <div className="mt-3 rounded-xl border border-red-700/55 bg-gradient-to-r from-[#3a0000] to-[#631018] p-3 text-xs text-red-100">
                Focused: {activeRegion.label} | {activeRegion.confidence}% confidence | {activeRegion.risk} risk
              </div>
            )}

            {hasAnalysis && activeRisk && (
              <div className="mt-3 rounded-xl border border-cyan-400/30 bg-cyan-950/25 p-4 text-sm text-cyan-50">
                <p className="text-xs uppercase tracking-[0.18em] text-cyan-200/80">What This Means For You</p>
                <p className="mt-2"><span className="font-semibold">Reading:</span> {activeRisk.label}</p>
                <p className="mt-1">{activeRisk.advice}</p>
                <p className="mt-2 text-xs text-cyan-100/80">This tool is a support system, not a final diagnosis.</p>
              </div>
            )}

            {hasAnalysis && (
              <div className="mt-3 rounded-xl border border-cyan-300/20 bg-cyan-950/15 p-3">
                <p className="mb-2 text-xs uppercase tracking-[0.18em] text-cyan-200/80">Quick Questions</p>
                <div className="flex flex-wrap gap-2">
                  {quickPrompts.map((prompt) => (
                    <button key={prompt} type="button"
                      className="rounded-full border border-cyan-300/25 bg-cyan-950/20 px-3 py-1 text-xs text-cyan-100 transition hover:bg-cyan-900/30"
                      onClick={() => void submitChat(prompt, "quick")}>
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-3 rounded-xl border border-white/10 bg-black/30 p-3">
              <form onSubmit={handleChat} className="flex gap-2">
                <input value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask any question..."
                  className="w-full rounded-xl border border-cyan-300/20 bg-[#070d11] px-4 py-3 text-sm text-zinc-100 outline-none focus:ring-2 ring-cyan-500/40" />
                <button type="submit"
                  className="rounded-xl bg-gradient-to-r from-[#6a0000] via-[#8b0000] to-[#c8102e] px-4 py-2 text-sm font-semibold text-red-50 disabled:opacity-70">
                  {isChatLoading ? "..." : "Send"}
                </button>
              </form>
            </div>

            <div className="mt-3 rounded-xl border border-red-300/20 bg-red-950/15 p-3">
              <p className="mb-2 text-xs uppercase tracking-[0.18em] text-red-200/80">Model Accuracy Statistics</p>
              <div className="grid grid-cols-2 gap-2 text-xs text-red-50">
                {Object.entries(modelStats).map(([key, value]) => (
                  <div key={key} className="rounded-lg border border-red-300/20 bg-black/30 px-2 py-1">
                    <span className="block text-[10px] uppercase tracking-[0.12em] text-red-200/80">{key.replace(/([A-Z])/g, " $1").trim()}</span>
                    <span className="font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}
// ChatUI.jsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import MessageBubble from "./MessageBubble";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function ChatUI() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [recs, setRecs] = useState(null);
  const [cityDetails, setCityDetails] = useState(null); // NEW
  const [selecting, setSelecting] = useState(false);    // NEW
  const bottomRef = useRef(null);

  useEffect(() => {
    const go = async () => {
      const r = await fetch(`${API_BASE}/start`, { method: "POST" });
      const data = await r.json();
      setSessionId(data.session_id);
      setMessages([{ role: "assistant", content: data.message }]);
    };
    go().catch(console.error);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recs, cityDetails]);

  const canSend = useMemo(
    () => input.trim().length > 0 && sessionId && !pending && !selecting,
    [input, sessionId, pending, selecting]
  );

  async function send(text) {
    if (!text) return;
    setPending(true);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");

    try {
      const r = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await r.json();
      if (data.kind === "question") {
        setRecs(null);
        setCityDetails(null);
        setMessages((m) => [...m, { role: "assistant", content: data.message }]);
      } else if (data.kind === "recommendations") {
        setRecs(data.recommendations || []);
        setCityDetails(null);
        setMessages((m) => [...m, { role: "assistant", content: "Here are your city matches:" }]);
      } else {
        setMessages((m) => [...m, { role: "assistant", content: data.message || "Something happened." }]);
      }
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Request failed: ${e}` }]);
    } finally {
      setPending(false);
    }
  }

  async function selectCity(city, country) {
    if (!sessionId) return;
    setSelecting(true);
    setCityDetails(null);
    try {
      const r = await fetch(`${API_BASE}/select-city`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, city, country }),
      });
      const data = await r.json();
      setCityDetails(data);
      setMessages((m) => [...m, { role: "assistant", content: `Great choice: ${data.title}` }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Selection failed: ${e}` }]);
    } finally {
      setSelecting(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="max-w-4xl mx-auto p-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            TripPlanner
          </h1>
          <p className="text-gray-600">Discover your perfect travel destination</p>
        </div>

        <div className="bg-white/80 backdrop-blur-sm border border-white/20 rounded-3xl shadow-2xl overflow-hidden">
          <div className="p-6 h-[65vh] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
            <div className="space-y-4">
              {messages.map((m, idx) => (
                <MessageBubble key={idx} role={m.role}>{m.content}</MessageBubble>
              ))}

              {recs && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {recs.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => selectCity(r.city, r.country)}
                      className="text-left group bg-gradient-to-br from-white to-blue-50 border border-blue-100 rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 focus:outline-none"
                      disabled={selecting}
                      title="Select this city"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="text-xl font-bold text-gray-800 group-hover:text-blue-600 transition-colors">
                          {r.city}
                        </div>
                        <div className="text-sm font-medium text-blue-600 bg-blue-100 px-3 py-1 rounded-full">
                          {r.country}
                        </div>
                      </div>
                      <p className="text-gray-700 leading-relaxed">{r.pitch}</p>
                      <div className="mt-4 w-full h-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    </button>
                  ))}
                </div>
              )}

              {cityDetails && (
                <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
                  <div className="md:col-span-1">
                    {cityDetails.image_url ? (
                      <img
                        src={cityDetails.image_url}
                        alt={cityDetails.title}
                        className="w-full h-64 object-cover rounded-2xl border border-gray-200 shadow"
                      />
                    ) : (
                      <div className="w-full h-64 rounded-2xl bg-gray-100 border border-gray-200 flex items-center justify-center text-gray-500">
                        No image available
                      </div>
                    )}
                  </div>
                  <div className="md:col-span-2 bg-white/90 border border-gray-200 rounded-2xl p-6 shadow">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">{cityDetails.title}</h2>
                    {cityDetails.page_url && (
                      <a href={cityDetails.page_url} target="_blank" rel="noreferrer" className="text-blue-600 underline">
                        Wikivoyage page
                      </a>
                    )}
                    <p className="mt-4 text-gray-700 whitespace-pre-line">
                      {cityDetails.summary || "No summary available."}
                    </p>
                    <div className="mt-4 text-sm text-gray-500">
                      Source: {cityDetails.was_existing ? "Qdrant (cached)" : "Wikivoyage (added to Qdrant)"}
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          <div className="border-t border-gray-100 bg-white/50 backdrop-blur-sm p-6">
            <div
              onSubmit={(e) => { e.preventDefault(); send(input); }}
              className="flex gap-3"
            >
              <div className="flex-1 relative">
                <input
                  className="w-full px-6 py-4 rounded-2xl border-2 border-gray-200 bg-white/80 backdrop-blur-sm focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 transition-all duration-200 text-gray-800 placeholder-gray-500"
                  placeholder={pending ? "Thinking..." : "Describe your ideal destination..."}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && canSend && send(input)}
                  disabled={pending || selecting}
                />
                {(pending || selecting) && (
                  <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
                  </div>
                )}
              </div>

              <button
                onClick={() => send(input)}
                disabled={!canSend}
                className="px-8 py-4 rounded-2xl bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
              >
                Send
              </button>

              <button
                onClick={() => send("recommend")}
                disabled={!sessionId || pending || selecting}
                className="px-8 py-4 rounded-2xl bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-900 hover:to-black text-white font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
              >
                Recommend
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

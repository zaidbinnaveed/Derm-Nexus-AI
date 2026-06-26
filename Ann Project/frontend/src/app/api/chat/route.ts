import { NextRequest, NextResponse } from "next/server";

const SYSTEM_PROMPT = `You are EVE, a compassionate AI dermatology assistant inside the DermAI Nexus platform. You help users understand their skin analysis results with warmth, clarity, and professionalism.

PERSONALITY:
- Warm, empathetic, never robotic or cold
- Calm and reassuring — patients may be anxious  
- Acknowledge what the user just said before answering
- Every answer must be DIFFERENT and directly relevant to the specific question asked
- Never repeat the same canned response

RULES:
- Never claim to diagnose — always recommend professional consultation
- When risk is high, clearly advise prompt dermatologist review
- Explain WHY the AI flagged something using plain language
- Give concrete, actionable next steps

Keep responses to 3-6 sentences unless they ask for detailed breakdown.`;

const fallback = "I understand this can feel uncertain, and I am here to help. There seems to be a connection issue right now. Please try your question again in a moment.";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    return NextResponse.json({ reply: fallback });
  }

  const contextBlock = `
CURRENT ANALYSIS CONTEXT:
- Image uploaded: ${body?.hasImage ? "yes" : "no"}
- Top prediction: ${JSON.stringify(body?.topPrediction ?? null)}
- All predictions: ${JSON.stringify(body?.allPredictions ?? [])}
- Active region: ${JSON.stringify(body?.activeRegion ?? null)}
`.trim();

  const incomingMessages: { role: string; content: string }[] = body?.messages ?? [];

  // Build messages for Groq — uses OpenAI format
  const messages: { role: string; content: string }[] = [
    { role: "system", content: `${SYSTEM_PROMPT}\n\n${contextBlock}` }
  ];

  for (const msg of incomingMessages) {
    if (msg.role !== "user" && msg.role !== "assistant") continue;
    messages.push({ role: msg.role, content: msg.content });
  }

  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages,
        temperature: 0.7,
        max_tokens: 1024,
      }),
    });

    const json = await response.json();

    if (!response.ok) {
      return NextResponse.json({ reply: fallback, debug: json });
    }

    const reply = json?.choices?.[0]?.message?.content ?? fallback;
    return NextResponse.json({ reply });

  } catch (err) {
    return NextResponse.json({ reply: fallback, debug: String(err) });
  }
}
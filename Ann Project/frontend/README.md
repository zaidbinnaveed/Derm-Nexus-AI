# Derm AI Nexus Frontend

Cinematic, production-style medical AI frontend for skin lesion analysis with:

- Immersive dark clinical visual design
- Upload + enhancement + annotated lesion visualization workflow
- Region-aware empathetic AI dermatologist chatbot
- Responsive and accessible interface architecture

## Tech Stack

- Next.js (App Router) + TypeScript
- Tailwind CSS v4
- OpenAI Chat Completions API (server route)
- Custom CSS motion system (DNA/medical micro-animations)

## Local Setup

From workspace root (`ANN Project`) you can run everything with:

```bash
npm run dev
```

This starts Derm AI Nexus and prints a single simple run message with the local address.

1) Install dependencies:

```bash
npm run install:all
```

2) Create `.env.local` in project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

3) Start development server:

```bash
npm run dev
```

4) Open:

- [http://localhost:3000](http://localhost:3000)

## Production Build

```bash
npm run build
npm run start
```

## Core Flow

1. Upload a skin image via drag-drop or file picker.
2. Client-side enhancement improves contrast and texture clarity.
3. Annotated panel overlays suspicious regions with confidence cues.
4. Click any region to trigger contextual explanation in chat.
5. Use chatbot for risk interpretation, prevention advice, and urgency guidance.

## Notes

- AI responses are advisory and intentionally avoid definitive diagnosis.
- If `OPENAI_API_KEY` is absent, the chatbot returns a safe clinical fallback response.

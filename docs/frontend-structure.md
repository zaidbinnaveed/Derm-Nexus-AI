# Derm AI Nexus - Project Structure

```text
frontend/
|-- src/
|   |-- app/
|   |   |-- api/
|   |   |   `-- chat/
|   |   |       `-- route.ts            # OpenAI-powered medical chatbot endpoint
|   |   |-- globals.css                 # Dark cinematic theme + animation system
|   |   |-- layout.tsx                  # Global metadata + typography
|   |   `-- page.tsx                    # Hero, upload pipeline, annotated views, chatbot
|-- public/                             # Cleaned (no default boilerplate assets)
|-- DermAI_Nexus_Report.tex             # Full LaTeX project report
|-- README.md                           # Setup and run instructions
|-- PROJECT_STRUCTURE.md                # This file
|-- package.json
|-- tsconfig.json
|-- next.config.ts
|-- postcss.config.mjs
`-- eslint.config.mjs
```

## Component Responsibility Map

- `page.tsx`
  - Hero cinematic landing experience
  - Upload and drag-drop workflow
  - Client-side image enhancement
  - Annotated lesion visualization with interactive regions
  - Region-aware empathetic chat interface
- `api/chat/route.ts`
  - Secure server-side OpenAI call
  - Context injection (regions, selected lesion, image state)
  - Safe fallback behavior if API key is missing/unavailable

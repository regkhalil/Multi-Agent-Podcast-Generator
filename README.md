---
title: Multi-Agent Podcast Generator
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
---

# 🎙️ Multi-Agent Podcast Generator

A multi-agent AI system that generates full podcast episodes from a single topic. Powered by [CrewAI](https://www.crewai.com/), it orchestrates specialized AI agents to research, write, and produce a complete podcast conversation with realistic text-to-speech audio.

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/regkhalil/multi-agent-podcast-generator)**

## Architecture

The pipeline is divided into four phases, each handled by specialized agents:

```
                        ┌─────────────────────┐
                        │  User Input: Topic  │
                        └─────────┬───────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                  ┌─────│    Frontend UI      │◄───────── ─────────┐
                  │     └─────────────────────┘                    │
                  ▼                                                │
        ┌───────────────────────┐                                  │
        │  Prompt Expander      │                                  │
        │  Topic → Rich Brief   │                                  │
        └──────────┬────────────┘                                  │
                   │ Master Prompt                                 │
                   ▼                                               │
    ┌─────────────────────────────────────┐                        │
    │    Research Phase (Parallel)        │                        │
    │  ┌───────────┐                      │                        │
    │  │ Orchestrator                     │                        │
    │  └─┬───────┬───────┬───────────────┘│                        │ 
    │    │       │       │                │                        │
    │    ▼       ▼       ▼                │                        │
    │ ┌──────┐┌──────┐┌──────────────┐    │                        │
    │ │Hist. ││Tech. ││Pop Culture / │    │                        │
    │ │Agent ││Agent ││Future Agent  │    │     Final Audio        │
    │ └──┬───┘└──┬───┘└──────┬───────┘    │                        │
    └────┼───────┼───────────┼────────────┘                        │
         │       │           │                                     │
         ▼       ▼           ▼                                     │
    ┌──────────────────────────────────┐                           │
    │       Synthesis Phase            │                           │
    │  ┌────────────────────────────┐  │                           │
    │  │ Scriptwriter Agent         │  │                           │
    │  └────────────┬───────────────┘  │                           │
    └───────────────┼──────────────────┘                           │
                    │ JSON Script                                  │
                    ▼                                              │
    ┌──────────────────────────────────┐                           │
    │      Audio Pipeline Phase        │                           │
    │  ┌───────────────────────────┐   │                           │
    │  │ Audio Processing Script   │   │                           │
    │  └─────┬──────────┬──────────┘   │                           │
    │        ▼          ▼              │                           │
    │   ┌─────────┐ ┌─────────┐        │                           │
    │   │ Voice A │ │ Voice B │        │                           │
    │   └────┬────┘ └────┬────┘        │                           │
    │        └─────┬─────┘             │                           │
    │              ▼                   │                           │
    │     ┌──────────────┐             │                           │
    │     │Audio Stitching├──────────  ┼───────────────────────────┘
    │     └──────────────┘             │
    └──────────────────────────────────┘
```

### Phase 1: Prompt Expansion

A **Prompt Expander** agent takes the user's basic topic (e.g., "Quantum Computing") and expands it into a detailed, multi-faceted research directive covering historical, technical, and cultural angles.

### Phase 2: Parallel Research

Three research agents work **in parallel**, each focusing on a different dimension:

| Agent | Focus |
|-------|-------|
| **Historian** | Origins, key milestones, founding figures |
| **Technologist** | Technical mechanics, engineering specs, scientific principles |
| **Futurist** | Future trends, upcoming developments, pop-culture impact |

### Phase 3: Script Synthesis

A **Scriptwriter** agent synthesizes all research notes into a natural, engaging podcast conversation between two characters:

- **Ali** — The charismatic host who drives the conversation
- **Amir** — The knowledgeable guest expert

The output is a structured JSON script with speaker, text, and emotion fields.

### Phase 4: Audio Pipeline

The script is converted to audio using **Edge TTS** (Microsoft's text-to-speech), with distinct voices for each speaker. Segments are generated concurrently, then stitched together with natural pauses into a final MP3 file.

## Features

- **Multi-agent orchestration** with CrewAI — 5 specialized agents collaborate sequentially and in parallel
- **Dual LLM support** — use Google Gemini (API) or local Ollama models
- **Realistic TTS** — Microsoft Edge neural voices with per-speaker voice mapping
- **Streamlit web UI** — real-time pipeline progress, log viewer, audio player, and download
- **CLI support** — run the full pipeline from the terminal via `make`
- **Timestamped outputs** — all scripts and audio files are timestamped to preserve history
- **Dockerized deployment** — deployable to HuggingFace Spaces or any Docker host
- **CI/CD** — GitHub Actions auto-deploys to HuggingFace on push to `main`

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker (only for local Ollama models)

### Installation

```bash
git clone https://github.com/regkhalil/Multi-Agent-Podcast-Generator.git
cd Multi-Agent-Podcast-Generator
uv sync
```

### Using Gemini (recommended)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)

2. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Add your key to `.env`:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

4. Set the provider in `config.toml`:
   ```toml
   [llm]
   provider = "gemini"
   ```

5. Run:
   ```bash
   make all        # Generate script + audio (CLI)
   make app        # Launch the Streamlit web UI
   ```

### Using Ollama (local, offline)

1. Set the provider in `config.toml`:
   ```toml
   [llm]
   provider = "ollama"
   ```

2. Run:
   ```bash
   make all        # Sets up Docker, pulls the model, generates script + audio
   ```

## Usage

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Start Ollama Docker container and pull the configured model |
| `make run` | Generate a podcast script (auto-detects provider) |
| `make audio` | Convert the latest script to audio |
| `make all` | Run the full pipeline: script + audio |
| `make app` | Launch the Streamlit web UI |
| `make run-gemini` | Generate a script using Gemini (skips Docker setup) |

### CLI

```bash
# Generate just the script
uv run python orchestrator.py

# Convert a specific script to audio
uv run python audio_pipeline.py output/script_20260331_120000.json
```

## Configuration

All settings are in `config.toml`:

```toml
[llm]
provider = "gemini"         # "ollama" or "gemini"

[ollama]
model = "smollm2"
base_url = "http://localhost:11434"
temperature = 0.7

[gemini]
model = "gemini-2.5-flash"
temperature = 0.7

[tts]
voice_ali = "en-GB-RyanNeural"
voice_amir = "en-US-AndrewNeural"
```

## Project Structure

```
├── app.py                 # Streamlit web interface
├── orchestrator.py        # CrewAI agent definitions and task pipeline
├── audio_pipeline.py      # TTS generation and audio stitching
├── config.toml            # LLM and TTS configuration
├── Dockerfile             # Docker build for HuggingFace Spaces
├── docker-compose.yaml    # Ollama local server setup
├── Makefile               # Build and run commands
├── requirements.txt       # Python dependencies (for Docker)
├── pyproject.toml         # Project metadata and dependencies (for uv)
├── .env.example           # API key template
├── .github/workflows/
│   └── deploy-hf.yml      # CI/CD: auto-deploy to HuggingFace Spaces
├── img/
│   └── architecture.png   # Architecture diagram
└── output/                # Generated scripts and audio files
```

## Deployment

The app is deployed to [HuggingFace Spaces](https://huggingface.co/spaces/regkhalil/multi-agent-podcast-generator) using Docker. A GitHub Actions workflow (`.github/workflows/deploy-hf.yml`) automatically pushes to HuggingFace on every commit to `main`.

To deploy your own instance:

1. Create a new HuggingFace Space with **Docker** SDK
2. Add `GEMINI_API_KEY` as a secret in the Space settings
3. Add `HF_TOKEN` as a secret in your GitHub repo settings
4. Push to `main` — the CI/CD pipeline handles the rest

## License

See [LICENSE](LICENSE) for details.

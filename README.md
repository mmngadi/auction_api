

```markdown
# Car Auction Intelligence API

LLM-powered natural-language interface to a vehicle auction database.
Ask plain-English questions about fair bid prices, market trends, and
vehicle conditions — the agent translates them into SQL, queries
PostgreSQL (with pgvector), and returns data-backed recommendations.

The API exposes an **OpenAI-compatible** `/v1/chat/completions` endpoint
(text only) with full **SSE streaming** support, making it a drop-in
backend for any OpenAI SDK or frontend framework like Vercel AI SDK.

**Model ID:** `auction-intelligence-v1`

---

## Prerequisites

| Dependency | Version |
|---|---|
| [Docker & Docker Compose](https://docs.docker.com/get-docker/) | V2+ |
| [Python](https://www.python.org/downloads/) | 3.12+ |
| [Ollama](https://ollama.com/) | Latest |

**Pull the required Ollama models before starting:**

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
```

**Then serve the Ollama local endpoint (on separate terminal):**

```bash
ollama serve
```

---

# Basic Usage

Everything below gets you running with the **included car auction dataset** — no configuration needed.

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd auction_api

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Copy the environment template (defaults work out of the box)
cp .env.example .env

# 4. Start the database, import data, and seed embeddings
make setup

# 5. Verify the environment
make row-count
make embedding-status

# 6. Start the API
make api
```

## Chat Interface (Open WebUI)

The recommended way to interact with the API is through
[Open WebUI](https://github.com/open-webui/open-webui) — a
self-hosted ChatGPT-style interface that connects to any
OpenAI-compatible backend.

**Start Open WebUI** (in a new terminal):

```bash
docker run -d -p 3000:8080 --name open-webui \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

**Connect it to the API:**

1. Open `http://localhost:3000` and create an admin account
2. Go to **Admin Panel** → **Settings** → **Connections**
3. Under **OpenAI API**, click **＋** and enter:

   | Field | Value |
   |---|---|
   | **URL** | `http://host.docker.internal:8000/v1` |
   | **API Key** | `no-key` |

4. Click the **verify** button (🔄) — it should find `auction-intelligence-v1`
5. **Save**, go back to chat, select **auction-intelligence-v1** from the model dropdown, and start asking questions

> **Note:** Responses take **30–60 seconds** on average. The agent is
> running tool calls behind the scenes (planning a query, executing SQL,
> summarising results). Open WebUI shows a loading indicator while this
> happens. Streaming kicks in once the final answer is ready — you'll
> see it appear token by token.

**Example conversation:**

```
You:  What is the average bid for a Renault Kwid 2019?
Bot:  Based on auction data… R45,000 …

You:  What about 2020?
Bot:  ← understands you still mean Renault Kwid, just a different year
```

> On **Linux**, replace `host.docker.internal` with your machine's LAN
> IP (e.g. `http://192.168.1.x:8000/v1`).

## Test via CLI

If you prefer the terminal, these Make targets are available:

```bash
# Health check
make health

# Non-streaming — full JSON response
make test-chat

# Streaming — rendered to stdout like ChatGPT
make test-stdout-stream
```

<details>
<summary><strong>curl examples</strong></summary>

**Non-streaming:**

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auction-intelligence-v1",
    "messages": [
      {"role": "user", "content": "What is the average bid for Hyundai cars?"}
    ]
  }' | python -m json.tool
```

**Streaming:**

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auction-intelligence-v1",
    "messages": [
      {"role": "user", "content": "Fair bid for a 2019 Renault Kwid with 60000 KM?"}
    ],
    "stream": true
  }'
```

</details>

## API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Healthcheck |

### SDK Integration

Since the API is OpenAI-compatible, any OpenAI SDK works as a client.

<details>
<summary><strong>TypeScript — Vercel AI SDK</strong></summary>

```typescript
import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

const auction = createOpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "sk-your-key", // omit if auth is disabled
});

const { textStream } = streamText({
  model: auction("auction-intelligence-v1"),
  prompt: "What is a fair bid for a 2020 VW Polo with 45000 KM?",
});

for await (const chunk of textStream) {
  process.stdout.write(chunk);
}
```

</details>

<details>
<summary><strong>Python — OpenAI SDK</strong></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-key",  # or "no-key" if auth is off
)

stream = client.chat.completions.create(
    model="auction-intelligence-v1",
    messages=[{"role": "user", "content": "Average bid for BMW?"}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

</details>

## Make Targets

```
make help               Show all available targets
make setup              Full env: start DB → seed embeddings
make up                 Start DB container
make down               Stop containers
make restart            Restart DB container
make reset              Wipe volume, re-import CSV, re-seed embeddings
make seed-embeddings    Generate condition embeddings
make embedding-status   Show embedding progress
make row-count          Print total rows in auction_lots
make psql               Open interactive psql session
make api                Start the FastAPI server
make health             Hit the /health endpoint
make test-chat          Smoke-test (non-streaming, JSON output)
make test-stream        Smoke-test (streaming, raw SSE output)
make test-stdout-stream Smoke-test (streaming, rendered to stdout)
make clean              Remove all containers and volumes
```

---

# Advanced Usage

This section covers configuration, architecture, authentication, and
how to adapt the project to your own dataset and domain.

## Configuration

All runtime settings are controlled through a single **`.env`** file.
Copy the template and edit as needed:

```bash
cp .env.example .env
```

### `.env` reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/auction` | Postgres connection string (local or cloud) |
| `POSTGRES_USER` | `postgres` | Docker Compose container user |
| `POSTGRES_PASSWORD` | `postgres` | Docker Compose container password |
| `POSTGRES_DB` | `auction` | Docker Compose database name |
| `POSTGRES_PORT` | `5432` | Host port mapped to the container |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_CHAT_MODEL` | `qwen3.5:9b` | Any tool-calling capable model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Any Ollama embedding model |
| `EMBED_DIMENSIONS` | `768` | Must match your embed model's output size |
| `OLLAMA_TIMEOUT` | `120` | Seconds before Ollama requests time out |
| `OLLAMA_NUM_PREDICT` | `4096` | Max tokens for LLM responses |
| `API_KEYS` | *(empty)* | Comma-separated keys; empty = auth disabled |
| `API_PORT` | `8000` | Port for the FastAPI server |
| `MODEL_ID` | `auction-intelligence-v1` | Model ID in `/v1/models` and responses |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_AGENT_ITERATIONS` | `6` | Max tool-calling loop iterations |
| `MAX_ROWS` | `500` | Hard ceiling on returned rows |
| `DEFAULT_LIMIT` | `20` | Default SQL LIMIT when LLM omits one |
| `MIN_LIMIT` | `10` | Floor for SQL LIMIT |
| `MILEAGE_TOLERANCE_KM` | `25000` | ± range for mileage similarity |
| `YEAR_TOLERANCE` | `5` | ± range for year similarity |

### Swapping models

Any **tool-calling capable** Ollama model works as the chat model, and
any Ollama embedding model works for vector search:

```env
# Example: switch to Llama 3.1 + mxbai embeddings
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=mxbai-embed-large
EMBED_DIMENSIONS=1024
```

> **⚠️ Important:** If you change `EMBED_DIMENSIONS`, also update the
> `vector(768)` type in `init.sql` to match (e.g. `vector(1024)`),
> then run `make reset` to rebuild the database and re-seed embeddings.

### Using a cloud database

Point `DATABASE_URL` at any Postgres instance with the
[pgvector](https://github.com/pgvector/pgvector) extension:

```env
# Supabase, Neon, RDS, etc.
DATABASE_URL=postgresql://user:pass@db.cloud-provider.com:5432/mydb
```

When using a cloud database the Docker container is not needed — skip
`make up` and run `make api` directly. You are responsible for running
`init.sql` against your cloud database and importing your CSV.

## Authentication

Disabled by default. Enable by adding keys to `.env`:

```env
API_KEYS=sk-key-one,sk-key-two
```

When enabled, all `/v1/*` requests require an `Authorization: Bearer <key>` header:

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-key-one" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Average bid for Toyota?"}]}'
```

### How it works

1. On startup, `app/config.py` reads `API_KEYS` from `.env` → splits by comma → stores as a tuple
2. Every `/v1/*` request passes through the `_verify_api_key` dependency in `app/main.py`
3. If the tuple is empty, auth is skipped entirely
4. If non-empty, the `Bearer` token must match one of the keys — otherwise a `401` is returned in OpenAI error format

### Custom auth strategies

The entire auth logic is one function in **`app/main.py`**:

```python
async def _verify_api_key(request: Request):
    if not settings.api_keys:
        return                      # no keys configured → open access
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ")
    if token not in settings.api_keys:
        raise HTTPException(401, ...)
```

Replace it with any strategy you need:

| Strategy | Example |
|---|---|
| Database lookup | Query a `users` table for the key |
| JWT validation | Decode and verify a signed token |
| OAuth2 / OIDC | Validate against an identity provider |
| Rate limiting | Track usage per key with Redis |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Client (Open WebUI / SDK / curl)                    │
│  POST /v1/chat/completions                           │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  main.py — OpenAI-compatible endpoint                │
│  Auth → extract prompt + history → stream or block   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  agent.py — Tool-calling loop (with conversation     │
│  history for multi-turn context)                     │
│                                                      │
│  SYSTEM_PROMPT + history + user message → LLM        │
│                                                      │
│  ┌─ LLM says call tool ──────────────────────────┐   │
│  │  tools.py — parse QueryPlan                   │   │
│  │  compiler.py — QueryPlan → parameterised SQL  │   │
│  │  database.py — execute against Postgres       │   │
│  │  embeddings.py — vector search (if needed)    │   │
│  └───────────────────────── result back to LLM ──┘   │
│                                                      │
│  LLM summarises results → stream / return            │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  PostgreSQL + pgvector                               │
│  auction_lots table + condition_embedding column     │
└──────────────────────────────────────────────────────┘
```

### Request lifecycle

1. **Client** sends a chat completion request with a natural-language question
2. **`main.py`** validates auth, extracts the prompt and conversation history
3. **`agent.py`** wraps the history with a system prompt containing the database schema and sends it to the LLM
4. The **LLM** returns a tool call with a structured `QueryPlan` dict
5. **`tools.py`** validates the plan with Pydantic and passes it to the compiler
6. **`compiler.py`** translates the QueryPlan into a parameterised SQL query
7. **`embeddings.py`** generates a vector if the plan includes `condition_text`
8. **`database.py`** executes the query and returns rows
9. The result is fed back to the **LLM** for summarisation
10. The final answer is streamed back as SSE chunks or returned as a single JSON response

> **Why do responses take 30–60 seconds?** The agent makes multiple LLM
> calls behind the scenes: one to plan the query, one to process tool
> results, and one to write the final answer. Each call runs through
> Ollama on local hardware. Faster GPUs and smaller models reduce this.

## Adapting to Your Own Data

This project is designed as a **plug-and-play template**. The table
below shows what to change and what to leave alone.

### ✏️ Customise for your project

| File | What to change | Why |
|---|---|---|
| `.env` | DB string, model names, API keys, ports | All runtime configuration |
| `init.sql` | `CREATE TABLE`, `COPY` column list, vector dimensions | Your table schema |
| `dataset/dataset.csv` | Replace entirely | Your source data |
| `app/agent.py` | `DATABASE_SCHEMA`, `SYSTEM_PROMPT`, `_build_summarisation_prompt` | Teach the LLM your schema and domain |
| `app/models.py` | `QueryPlan`, `Filter`, `Aggregation` fields | Match your schema's columns and types |
| `app/compiler.py` | Allowed columns, table name | SQL generation must reference your tables |
| `seed_embeddings.py` | Column name in the `SELECT` / `UPDATE` | If you embed a different column |

### ✅ Leave as-is (framework plumbing)

| File | Why it's universal |
|---|---|
| `app/main.py` | OpenAI-compatible endpoint — works for any backend |
| `app/config.py` | Reads `.env` → typed settings, schema-agnostic |
| `app/database.py` | Connection pooling, schema-agnostic |
| `app/embeddings.py` | Generic Ollama embedding helper |
| `app/tools.py` | Bridges QueryPlan → compiler → DB, schema-agnostic |
| `app/utils.py` | JSON serialisation helpers |
| `run.py` | Uvicorn launcher |
| `Makefile` | Orchestration targets |
| `docker-compose.yml` | Postgres + pgvector container |

### Step-by-step

1. **Replace the data** — drop your CSV into `dataset/` and update the filename in `docker-compose.yml` volumes and `init.sql` `COPY` command.

2. **Update the schema** — edit `init.sql` with your `CREATE TABLE`. Add a `vector(N)` column if you want semantic search on a text field.

3. **Update the agent** — edit `DATABASE_SCHEMA` and `SYSTEM_PROMPT` in `app/agent.py` so the LLM knows your columns, types, and domain rules.

4. **Update the query model** — adjust fields in `app/models.py` to match your column names and filter types.

5. **Update the compiler** — edit the allowed column list in `app/compiler.py` to match your schema.

6. **Update the embedder** — edit the `SELECT` / `UPDATE` query in `seed_embeddings.py` to target whichever text column you want embedded.

7. **Rebuild** — run `make reset` to wipe the database, reimport your data, and regenerate embeddings.

## Project Structure

```
auction_api/
├── app/
│   ├── config.py            # Centralised settings (reads .env)
│   ├── models.py            # Pydantic models and enums
│   ├── database.py          # Thread-safe DB pool
│   ├── compiler.py          # QueryPlan → parameterised SQL
│   ├── embeddings.py        # Ollama vector embeddings
│   ├── tools.py             # LangChain tool definitions
│   ├── agent.py             # LLM agent orchestration + prompts
│   ├── main.py              # FastAPI OpenAI-compatible endpoint
│   └── utils.py             # Serialisation helpers
├── dataset/
│   └── dataset.csv          # Auction lot source data
├── .env.example             # Environment template (copy to .env)
├── .env                     # Your local config (git-ignored)
├── docker-compose.yml
├── init.sql                 # DB schema + CSV import
├── seed_embeddings.py       # One-time embedding generation
├── run.py                   # API launcher
├── Makefile
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
```
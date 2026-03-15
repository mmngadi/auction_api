# Car Auction Intelligence API

LLM-powered natural-language interface to a vehicle auction database.
Ask plain-English questions about fair bid prices, market trends, and
vehicle conditions — the agent translates them into SQL, queries
PostgreSQL (with pgvector), and returns data-backed recommendations.

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

**Then serve the Ollama local endpoint:**

```bash
ollama serve
```

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd auction_api

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Start the database, import data, and seed embeddings
make setup

# 4. Verify the environment
make row-count
make embedding-status
make health

# 5. Start the API
make api
```

## Usage

```bash
# Health check
curl -sS http://localhost:8000/health | python -m json.tool

# Ask a question
curl -sS -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the average winning bid for Hyundai cars?"}' \
  | python -m json.tool

# More examples
curl -sS -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I see a 2019 Renault Kwid with 60000 KM, spray work and alignment issues. What is a fair bid?"}' \
  | python -m json.tool
```

## Project Structure

```
auction_api/
├── app/
│   ├── config.py            # Centralised settings
│   ├── models.py            # Pydantic models and enums
│   ├── database.py          # Thread-safe DB pool
│   ├── compiler.py          # QueryPlan → parameterised SQL
│   ├── embeddings.py        # Ollama vector embeddings
│   ├── tools.py             # LangChain tool definitions
│   ├── agent.py             # LLM agent orchestration
│   ├── main.py              # FastAPI entry-point
│   └── utils.py             # Serialisation helpers
├── dataset/
│   └── dataset.csv          # Auction lot source data
├── docker-compose.yml
├── init.sql                 # DB schema + CSV import
├── seed_embeddings.py       # One-time embedding generation
├── run.py                   # API launcher
├── Makefile
├── requirements.txt
└── README.md
```

## Make Targets

```
make help              Show all available targets
make setup             Full env: start DB → seed embeddings
make up                Start DB container
make down              Stop containers
make reset             Wipe volume, re-import CSV, re-seed embeddings
make seed-embeddings   Generate condition embeddings
make embedding-status  Show embedding progress
make row-count         Print total rows in auction_lots
make psql              Open interactive psql session
make api               Start the FastAPI server
make health            Hit the /health endpoint
make test-chat         Send a smoke-test prompt
make clean             Remove all containers and volumes
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

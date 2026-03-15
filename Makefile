# ──────────────────────────────────────────────────────────────
#  Auction Intelligence API — environment orchestration
# ──────────────────────────────────────────────────────────────

-include .env

COMPOSE   := docker compose
SERVICE   := auction_db
DB_USER   ?= postgres
DB_NAME   ?= auction
API_PORT  ?= 8000
MAKE = make

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Database lifecycle ────────────────────────────────────────

.PHONY: up down restart reset

up: ## Start DB container (waits until healthy)
	$(COMPOSE) up -d --wait
	@echo "Database ready on localhost:5432"

down: ## Stop and remove containers
	$(COMPOSE) down

restart: down up ## Restart the DB container

reset: ## Destroy volume and re-import CSV from scratch
	$(COMPOSE) down -v
	$(COMPOSE) up -d --wait
	$(MAKE) seed-embeddings
	@echo "Database reset — fresh import from dataset.csv"

# ── Database inspection ───────────────────────────────────────

.PHONY: psql logs status row-count sample

psql: ## Open an interactive psql session
	$(COMPOSE) exec $(SERVICE) psql -U $(DB_USER) -d $(DB_NAME)

logs: ## Tail DB container logs
	$(COMPOSE) logs -f $(SERVICE)

status: ## Show container status and health
	$(COMPOSE) ps

row-count: ## Print total rows in auction_lots
	@$(COMPOSE) exec $(SERVICE) psql -U $(DB_USER) -d $(DB_NAME) -c \
		"SELECT COUNT(*) AS total_rows FROM auction_lots;"

sample: ## Show 5 sample rows
	@$(COMPOSE) exec $(SERVICE) psql -U $(DB_USER) -d $(DB_NAME) -c \
		"SELECT id, make, year, mileage, winning_bid_amount FROM auction_lots LIMIT 5;"

makes: ## List distinct makes and their counts
	@$(COMPOSE) exec $(SERVICE) psql -U $(DB_USER) -d $(DB_NAME) -c \
		"SELECT make, COUNT(*) AS cnt FROM auction_lots GROUP BY make ORDER BY cnt DESC;"

# ── API ───────────────────────────────────────────────────────

.PHONY: api health test-chat test-stream test-stdout-stream

api: ## Start the FastAPI server
	python run.py

health: ## Hit the /health endpoint
	@curl -sS http://localhost:$(API_PORT)/health | python -m json.tool

test-chat: ## Send a smoke-test prompt (non-streaming)
	@curl -sS -X POST http://localhost:$(API_PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"What car makes are available in the auction database?"}]}' \
		| python -m json.tool

test-stream: ## Send a smoke-test prompt (streaming SSE)
	@curl -sN -X POST http://localhost:$(API_PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"What is the average bid for Toyota?"}],"stream":true}'

test-stdout-stream: ## Stream a chat response directly to stdout
	@curl -sN -X POST http://localhost:$(API_PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"What is the average bid for Toyota?"}],"stream":true}' \
		| python -c "import sys,json;[print(json.loads(l[6:])['choices'][0]['delta'].get('content',''),end='',flush=True) for l in sys.stdin if l.strip().startswith('data: {')];print()"

# ── Cleanup ───────────────────────────────────────────────────

.PHONY: clean

clean: ## Remove all containers, volumes, and networks
	$(COMPOSE) down -v --remove-orphans
	@echo "All project containers and volumes removed"

# ── Embeddings ────────────────────────────────────────────────

.PHONY: seed-embeddings embedding-status

seed-embeddings: ## Generate condition embeddings (requires Ollama)
	python seed_embeddings.py

embedding-status: ## Show how many rows still need embeddings
	@$(COMPOSE) exec $(SERVICE) psql -U $(DB_USER) -d $(DB_NAME) -c \
		"SELECT \
		   COUNT(*) FILTER (WHERE condition_embedding IS NOT NULL) AS embedded, \
		   COUNT(*) FILTER (WHERE condition_embedding IS NULL)     AS pending, \
		   COUNT(*)                                                AS total \
		 FROM auction_lots;"

# ── Full setup ────────────────────────────────────────────────

.PHONY: setup

setup: up seed-embeddings ## Full env: start DB → seed embeddings
	@echo "Environment ready — run 'make api' to start the server"

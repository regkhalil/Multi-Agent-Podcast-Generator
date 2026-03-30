.PHONY: setup run

MODEL := $(shell python3 -c "import tomllib; print(tomllib.load(open('config.toml','rb'))['ollama']['model'])" 2>/dev/null || echo llama3)

setup:
	docker compose up -d
	@echo "Waiting for Ollama to be ready..."
	@until docker exec local_llm_server ollama list >/dev/null 2>&1; do sleep 1; done
	docker exec local_llm_server ollama pull $(MODEL)
	@echo "Ready — $(MODEL) is available."

run: setup
	uv run python orchestrator.py

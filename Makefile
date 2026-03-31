.PHONY: setup run audio all app run-gemini

PROVIDER := $(shell python3 -c "import tomllib; print(tomllib.load(open('config.toml','rb')).get('llm',{}).get('provider','ollama'))" 2>/dev/null || echo ollama)
MODEL := $(shell python3 -c "import tomllib; print(tomllib.load(open('config.toml','rb'))['ollama']['model'])" 2>/dev/null || echo llama3)

setup:
	docker compose up -d
	@echo "Waiting for Ollama to be ready..."
	@until docker exec local_llm_server ollama list >/dev/null 2>&1; do sleep 1; done
	docker exec local_llm_server ollama pull $(MODEL)
	@echo "Ready — $(MODEL) is available."

run:
ifeq ($(PROVIDER),gemini)
	uv run python orchestrator.py
else
	$(MAKE) setup
	uv run python orchestrator.py
endif

run-gemini:
	uv run python orchestrator.py

audio:
	@LATEST_SCRIPT=$$(ls -t output/script_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST_SCRIPT" ]; then \
		echo "No script found in output/. Run 'make run' first."; \
		exit 1; \
	fi; \
	uv run python audio_pipeline.py "$$LATEST_SCRIPT"

app: setup
	uv run streamlit run app.py

all: run audio

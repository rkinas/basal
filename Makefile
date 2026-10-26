.DEFAULT_GOAL := help
PYTHON ?= .venv/bin/python
UV ?= uv
EXTRAS ?= test
MODEL ?=
MODE ?=
HOST ?= 127.0.0.1
PORT ?= 8000
URL ?= http://127.0.0.1:$(PORT)/v1/systemone
INPUT ?=
OUTPUT ?=
GGUF ?=
OLLAMA_MODEL ?=
ARGS ?=
export PYTHON UV EXTRAS MODEL MODE HOST PORT URL INPUT OUTPUT GGUF OLLAMA_MODEL

.PHONY: help setup check test serve bench run loadtest check-backend check-input
help:
	@printf '%s\n' \
	  'make setup    Install editable app and EXTRAS=test into prepared PYTHON=.venv/bin/python' \
	  'make check    Run pinned, read-only pre-commit checks on all files (no hook installation)' \
	  'make test     Run pytest in PYTHON; requires app/test dependencies and backend hardware' \
	  'make serve    Start server on HOST (127.0.0.1); MODE empty selects backend automatically' \
	  'make bench    Offline benchmark; MODE empty uses CLI defaults, otherwise one backend' \
	  'make run      Batch requests to URL; requires INPUT and OUTPUT paths' \
	  'make loadtest Load-test URL; optional INPUT contains simple JSONL questions' \
	  'Variables: PYTHON UV EXTRAS MODEL MODE HOST PORT URL INPUT OUTPUT GGUF OLLAMA_MODEL ARGS' \
	  'MODEL empty uses the default model of basal-serve / basal-bench; HOST=0.0.0.0 listens on every interface' \
	  'MODE: eager fast fast-nocompile fast-exit fp8 nvfp4 vllm sglang mlx mlx-q8 mps gguf strix tpu ollama llamacpp' \
	  'GGUF mode requires GGUF=<file>; Ollama mode requires OLLAMA_MODEL=<import name>' \
	  'ARGS appends CLI options (shell quoting applies); see README for setup and examples.'

setup:
	"$$UV" pip install --python "$$PYTHON" -e ".[$$EXTRAS]"

check:
	"$$UV" tool run pre-commit==4.7.0 run --all-files

test:
	"$$PYTHON" -m pytest $(ARGS)

check-backend:
	@if [ "$$MODE" = gguf ] && [ -z "$$GGUF" ]; then echo 'MODE=gguf requires GGUF=<file>' >&2; exit 1; fi
	@if [ -n "$$GGUF" ] && { [ ! -f "$$GGUF" ] || [ ! -r "$$GGUF" ]; }; then echo 'GGUF must be a readable file' >&2; exit 1; fi
	@if [ "$$MODE" = ollama ] && [ -z "$$OLLAMA_MODEL" ]; then echo 'MODE=ollama requires OLLAMA_MODEL=<import name>' >&2; exit 1; fi

check-input:
	@if [ -n "$$INPUT" ] && { [ ! -f "$$INPUT" ] || [ ! -r "$$INPUT" ]; }; then echo 'INPUT must be a readable file' >&2; exit 1; fi
	@if [ -n "$$INPUT" ] && [ -n "$$OUTPUT" ] && { [ "$$INPUT" = "$$OUTPUT" ] || [ "$$INPUT" -ef "$$OUTPUT" ]; }; then echo 'OUTPUT must not be the INPUT file' >&2; exit 1; fi

serve: check-backend
	"$$PYTHON" -m basal.server $(if $(MODEL),--model "$$MODEL") --host "$$HOST" --port "$$PORT" \
	  $(if $(MODE),--mode "$$MODE") $(if $(GGUF),--gguf "$$GGUF") \
	  $(if $(OLLAMA_MODEL),--ollama-model "$$OLLAMA_MODEL") $(ARGS)

bench: check-backend check-input
	"$$PYTHON" -m basal.bench $(if $(MODEL),--model "$$MODEL") \
	  $(if $(MODE),--modes "$$MODE") $(if $(GGUF),--gguf "$$GGUF") \
	  $(if $(OLLAMA_MODEL),--ollama-model "$$OLLAMA_MODEL") \
	  $(if $(INPUT),--questions "$$INPUT") $(if $(OUTPUT),--out "$$OUTPUT") $(ARGS)

run: check-input
	@test -n "$$INPUT" && test -n "$$OUTPUT" || { echo 'run requires INPUT=<file> and OUTPUT=<file>' >&2; exit 1; }
	"$$PYTHON" -m basal.run --url "$$URL" --input "$$INPUT" --output "$$OUTPUT" $(ARGS)

loadtest: check-input
	"$$PYTHON" -m basal.loadtest --url "$$URL" \
	  $(if $(INPUT),--questions "$$INPUT") $(if $(OUTPUT),--out "$$OUTPUT") $(ARGS)

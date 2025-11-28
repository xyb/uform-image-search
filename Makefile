.PHONY: install start start-dev stop

PYTHON ?= python3
APP = server:app
IMAGE_DIR ?= data

install:
	$(PYTHON) -m pip install -r requirements.txt

start:
	IMAGE_DIR=$(IMAGE_DIR) $(PYTHON) -m uvicorn $(APP) --host 0.0.0.0 --port 8000

start-dev:
	IMAGE_DIR=$(IMAGE_DIR) $(PYTHON) -m uvicorn $(APP) --host 0.0.0.0 --port 8000 --reload

stop:
	@echo "Stopping FastAPI server..."
	@pkill -f "$(APP)" && echo "Server stopped." || echo "Server not running."

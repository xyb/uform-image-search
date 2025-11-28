import os
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from image_search_engine import ImageSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEFAULT_IMAGE_DIR = str(BASE_DIR / "data")
raw_image_dirs = os.environ.get("IMAGE_DIR")

# If IMAGE_DIR is unset or blank, fall back to ./data in the repo
if raw_image_dirs:
    IMAGE_DIRS = [p.strip() for p in raw_image_dirs.split(",") if p.strip()]
else:
    IMAGE_DIRS = [DEFAULT_IMAGE_DIR]

if not IMAGE_DIRS:
    IMAGE_DIRS = [DEFAULT_IMAGE_DIR]

# Ensure directories exist
for _dir in IMAGE_DIRS:
    os.makedirs(_dir, exist_ok=True)

TEXT_SIMILARITY_THRESHOLD = float(os.environ.get("TEXT_SIMILARITY_THRESHOLD", 0.15))
IMAGE_SIMILARITY_THRESHOLD = float(os.environ.get("IMAGE_SIMILARITY_THRESHOLD", 0.85))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IMAGE_DIRS:
        raise RuntimeError("IMAGE_DIR must point to at least one directory with images.")
    logger.info("Indexing images from: %s", IMAGE_DIRS)
    app.state.search_engine = ImageSearchEngine(image_dirs=IMAGE_DIRS)
    yield
    app.state.search_engine = None


app = FastAPI(lifespan=lifespan)


@app.get("/image/{key:int}", response_class=FileResponse)
async def get_image(request: Request, key: int):
    path = request.app.state.search_engine.get_image_path(key)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/search/image/{key:int}", response_class=HTMLResponse)
async def search_image(request: Request, key: int):
    path = request.app.state.search_engine.get_image_path(key)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    results = request.app.state.search_engine.search_by_image(Image.open(path), score_threshold=IMAGE_SIMILARITY_THRESHOLD)
    return templates.TemplateResponse("search_results.html", {"request": request, "results": results})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = request.app.state.search_engine.search(query, score_threshold=TEXT_SIMILARITY_THRESHOLD)
    return templates.TemplateResponse("search_results.html", {"request": request, "results": results})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_dirs": ", ".join(request.app.state.search_engine.list_image_directories()),
        },
    )

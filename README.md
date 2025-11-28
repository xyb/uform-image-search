# Simple Image Search

Minimal service to demo [UForm](https://github.com/unum-cloud/UForm)-powered image search using the Hugging Face model [unum-cloud/uform3-image-text-multilingual-base](https://huggingface.co/unum-cloud/uform3-image-text-multilingual-base). Benefits:
- Works out of the box with your own image folder.
- Compact and fast to run.
- Multilingual (90+ languages) zero-shot retrieval.

![Image search UI](docs/search.png)

## Quickstart

1) Install deps (use any venv you like):
```bash
pip install -r requirements.txt
```

2) Default: drop images into `./data` and run:
```bash
make start
```

Different folder? Set `IMAGE_DIR` and use Makefile:
```bash
export IMAGE_DIR=/absolute/path/to/images
make start
```

Or run uvicorn directly:
```bash
IMAGE_DIR=/absolute/path/to/images uvicorn server:app --host 0.0.0.0 --port 8000
```

3) Open `http://localhost:8000` and start typing; results stream in automatically. Click "Search similar" on any result to run an image-to-image search.

> On first run the UForm model (~2.2GB) is downloaded automatically—expect a short wait.
### Make targets (optional)

```
make install      # pip install -r requirements.txt
make start        # run server on 0.0.0.0:8000 (honors IMAGE_DIR)
make start-dev    # same with uvicorn reload
```

## Notes

- Index builds on startup from the provided `IMAGE_DIR` (recurses through subdirectories, skips empty files).
- Supported extensions: jpg, jpeg, png, webp.
- No datasets or generated index files are included; bring your own images.

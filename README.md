# Assignment API Service: Corpus, BQML, Promotion, PEFT, and Quantization

This project implements all five assignment endpoints:

```text
POST /build-corpus
POST /bqml
POST /promote
POST /adapt
POST /quantize
```

## Files

- `main.py` — the complete FastAPI service
- `requirements.txt` — pinned Python packages
- `README.md` — local testing and deployment instructions
- `test_request.py` — tests `POST /build-corpus`
- `test_bqml.py` — tests both BQML phases
- `test_promote.py` — tests promotion and replay retention
- `test_adapt.py` — tests intervention choice and PEFT repair
- `test_quantize.py` — tests quantization freeze, replay, integrity, and selection

## Run locally

1. Open a terminal in this folder.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it on Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the service:

   ```bash
   uvicorn main:app --reload
   ```

6. In a second terminal, run the test:

   ```bash
   python test_request.py http://127.0.0.1:8000
   python test_bqml.py http://127.0.0.1:8000
   python test_promote.py http://127.0.0.1:8000
   python test_adapt.py http://127.0.0.1:8000
   python test_quantize.py http://127.0.0.1:8000
   ```

## Deploy on Render

1. Upload all eight files listed above to the root of a GitHub repository.
2. In Render, create a **Web Service** from that repository.
3. Set the build command to:

   ```text
   pip install -r requirements.txt
   ```

4. Set the start command to:

   ```text
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

5. Deploy and copy the URL ending in `.onrender.com`.
6. Test it, replacing the example URL with yours:

   ```bash
   python test_request.py https://your-service-name.onrender.com
   python test_bqml.py https://your-service-name.onrender.com
   python test_promote.py https://your-service-name.onrender.com
   python test_adapt.py https://your-service-name.onrender.com
   python test_quantize.py https://your-service-name.onrender.com
   ```

7. Submit only the base URL, without an endpoint path.

The graders themselves add `/build-corpus`, `/bqml`, `/promote`, `/adapt`, or `/quantize` and send POST requests.

# Canopy Image Processing — Web App

Streamlit application based on `Okra_Shoot_HSI_Depth_Annotation.ipynb`.

## Features
- Upload a ZIP containing canopy images (JPG/JPEG/PNG/BMP/TIF/TIFF/WEBP).
- Optionally upload CVAT image-task XML, or a ZIP containing XML.
- Automatically generates depth maps, HSI, HSIE (H, S, normalized predicted depth), optional H/S/I channels, and matched annotation overlays.
- Preview outputs and download all results as one ZIP.
- Names and output folders are generated from the uploaded dataset; no day-specific paths need editing.

## Deploy online with Streamlit Community Cloud
1. Create a GitHub repository and upload `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, and this `README.md`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file to `streamlit_app.py`.
4. Deploy. The first run downloads the Hugging Face Depth Anything V2 model, so allow time for the first startup.
5. Open the deployed URL and upload your image ZIP. CVAT XML is optional.

## Local test
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important
- This app uses the notebook's model ID: `depth-anything/Depth-Anything-V2-Small-hf`.
- Depth Anything output is relative depth, not calibrated physical distance.
- HSIE follows the notebook's convention: H (hue), S (saturation), normalized predicted depth.
- CVAT polygon overlays are drawn on HSIE outputs using the XML polygon coordinates. XML image names should match the uploaded image paths or basenames.
- Public cloud hosts have CPU, memory, upload, and execution limits. Large datasets may need a paid instance or a GPU-backed host; test with a small ZIP first.
- Do not upload confidential crop imagery to a public app unless access controls and data handling are appropriate.

import os
import io
import re
import glob
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

st.set_page_config(page_title="Canopy Image Processing", page_icon="🌱", layout="wide")
st.title("🌱 Canopy Image Processing")
st.caption("Upload canopy images, generate depth / HSI / HSIE outputs, inspect intermediate images, and download results.")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

@st.cache_resource(show_spinner="Loading Depth Anything V2 model (first run may take a while)...")
def load_depth_model(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device

def safe_name(name):
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "dataset"

def extract_zip(uploaded_file, destination):
    # Reject path traversal entries in uploaded archives.
    with zipfile.ZipFile(uploaded_file) as zf:
        root = Path(destination).resolve()
        for info in zf.infolist():
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("The ZIP contains an unsafe file path.")
        zf.extractall(destination)

def rgb_to_hsi(image_bgr):
    img = image_bgr.astype(np.float32) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    intensity = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    saturation = np.zeros_like(intensity)
    nz = intensity > 1e-6
    saturation[nz] = 1.0 - min_rgb[nz] / intensity[nz]
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt(np.maximum((R - G) ** 2 + (R - B) * (G - B), 0.0)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    hue = np.zeros_like(intensity)
    hue[B <= G] = theta[B <= G]
    hue[B > G] = 2.0 * np.pi - theta[B > G]
    hue /= (2.0 * np.pi)
    H = (hue * 255).astype(np.uint8)
    S = (saturation * 255).astype(np.uint8)
    I = (intensity * 255).astype(np.uint8)
    return cv2.merge([H, S, I]), H, S, I

def parse_cvat_xml(xml_path):
    """Read CVAT image-task XML; retain both relative paths and basenames for matching."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    for image_tag in root.findall('.//image'):
        name = (image_tag.get('name') or '').replace('\\\\', '/').lstrip('./')
        polygons = image_tag.findall('polygon')
        if name:
            annotations[name] = polygons
            # Keep a basename fallback for common CVAT exports.
            annotations.setdefault(os.path.basename(name), polygons)
    return annotations

def make_overlay(image_bgr, polygons):
    # BGR colors, one stable color per label.
    color_map = {
        'leaf': (0, 255, 0), 'bud': (255, 0, 0), 'flower': (0, 255, 255),
        'pre-mature pod': (255, 165, 0), 'mature pod': (0, 128, 255),
        'post-mature pod': (0, 0, 255)
    }
    base = image_bgr.copy()
    fill = image_bgr.copy()
    count = 0
    for poly in polygons:
        label = (poly.get('label') or 'object').lower()
        pts_str = poly.get('points', '')
        try:
            pts = np.array([[float(v) for v in pair.split(',')] for pair in pts_str.split(';') if pair], dtype=np.int32)
            if len(pts) < 3:
                continue
            pts = pts.reshape((-1, 1, 2))
        except (ValueError, TypeError):
            continue
        color = color_map.get(label, (0, 255, 0))
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(base, [pts], True, (255, 255, 255), 2)
        x_min, y_min = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        cv2.putText(base, label, (int(x_min), max(20, int(y_min) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        count += 1
    return cv2.addWeighted(fill, 0.35, base, 0.65, 0), count

def save_png(path, array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise IOError(f"Could not write image: {path}")

with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("Depth model", value=MODEL_ID, help="Hugging Face model ID for monocular relative depth estimation.")
    preview_count = st.slider("Images to preview after processing", min_value=1, max_value=20, value=6)
    save_hsi_channels = st.checkbox("Save separate H, S and I channel images", value=True)
    st.caption("Depth estimates are relative depth, not calibrated physical distances.")

images_zip = st.file_uploader("1. Upload canopy image ZIP", type=["zip"], key="images_zip")
annotations_upload = st.file_uploader("2. Optional: upload CVAT annotation XML or ZIP", type=["xml", "zip"], key="annotations")
run = st.button("🚀 Process uploaded dataset", type="primary", disabled=images_zip is None, use_container_width=True)

if run:
    dataset_label = safe_name(images_zip.name)
    work_root = tempfile.mkdtemp(prefix="canopy_pipeline_")
    input_dir = os.path.join(work_root, "input")
    output_root = os.path.join(work_root, "processed", dataset_label)
    os.makedirs(input_dir, exist_ok=True)
    raw_dir = os.path.join(output_root, "original")
    depth_dir = os.path.join(output_root, "depth_maps")
    hsi_dir = os.path.join(output_root, "hsi_images")
    hsie_dir = os.path.join(output_root, "hsie_enhanced")
    channel_dir = os.path.join(output_root, "hsi_channels")
    overlay_dir = os.path.join(output_root, "annotation_overlays")
    for folder in [raw_dir, depth_dir, hsi_dir, hsie_dir, channel_dir, overlay_dir]:
        os.makedirs(folder, exist_ok=True)

    try:
        with st.status("Preparing uploaded files...", expanded=True) as status:
            images_zip.seek(0)
extract_zip(images_zip, input_dir)
            image_paths = sorted(p for p in Path(input_dir).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if not image_paths:
                raise ValueError("No supported images were found in the uploaded ZIP.")
            st.write(f"Found **{len(image_paths)}** images.")
            annotations = {}
            if annotations_upload is not None:
                ann_dir = os.path.join(work_root, "annotations")
                os.makedirs(ann_dir, exist_ok=True)
                if annotations_upload.name.lower().endswith('.xml'):
                    xml_path = os.path.join(ann_dir, os.path.basename(annotations_upload.name))
                    with open(xml_path, 'wb') as f:
                        f.write(annotations_upload.getvalue())
                else:
                    annotations_upload.seek(0)
extract_zip(annotations_upload, ann_dir)
                    xmls = list(Path(ann_dir).rglob('*.xml'))
                    if not xmls:
                        raise ValueError("The annotation ZIP did not contain an XML file.")
                    xml_path = str(xmls[0])
                annotations = parse_cvat_xml(xml_path)
                st.write(f"Loaded CVAT annotations for **{len(annotations)}** image entries.")

            processor, model, device = load_depth_model(model_id)
            st.write(f"Depth model loaded on **{device}**.")
            progress = st.progress(0, text="Processing images...")
            records = []
            used_names = set()
            for idx, path in enumerate(image_paths, start=1):
                raw_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if raw_bgr is None:
                    st.warning(f"Skipped unreadable image: {path.name}")
                    continue
                # Preserve relative folder names and avoid collisions for duplicate basenames.
                rel = path.relative_to(input_dir)
                key = str(rel.with_suffix('')).replace(os.sep, "__")
                key = safe_name(key)
                candidate = key
                n = 2
                while candidate in used_names:
                    candidate = f"{key}_{n}"
                    n += 1
                used_names.add(candidate)
                h, w = raw_bgr.shape[:2]
                raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(raw_rgb)
                inputs = processor(images=pil_img, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                depth_map = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False
                ).squeeze().detach().cpu().numpy()
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hsi_img, H, S, I = rgb_to_hsi(raw_bgr)
                # Retains the notebook's HSIE fusion convention: H, S, normalized depth.
                hsie_img = cv2.merge([H, S, depth_norm])
                save_png(Path(raw_dir) / f"{candidate}_original.png", raw_bgr)
                save_png(Path(depth_dir) / f"{candidate}_depth.png", depth_norm)
                save_png(Path(hsi_dir) / f"{candidate}_hsi.png", hsi_img)
                save_png(Path(hsie_dir) / f"{candidate}_hsie.png", hsie_img)
                if save_hsi_channels:
                    save_png(Path(channel_dir) / f"{candidate}_H_hue.png", H)
                    save_png(Path(channel_dir) / f"{candidate}_S_saturation.png", S)
                    save_png(Path(channel_dir) / f"{candidate}_I_intensity.png", I)
                relative_annotation_name = str(path.relative_to(input_dir)).replace(os.sep, '/')
                matched_polys = annotations.get(relative_annotation_name, annotations.get(path.name, []))
                if matched_polys:
                    overlay, polygon_count = make_overlay(cv2.imread(str(Path(hsie_dir) / f"{candidate}_hsie.png")), matched_polys)
                    save_png(Path(overlay_dir) / f"{candidate}_annotated.png", overlay)
                else:
                    polygon_count = 0
                records.append({"name": path.name, "key": candidate, "original": str(Path(raw_dir) / f"{candidate}_original.png"),
                                "depth": str(Path(depth_dir) / f"{candidate}_depth.png"), "hsi": str(Path(hsi_dir) / f"{candidate}_hsi.png"),
                                "hsie": str(Path(hsie_dir) / f"{candidate}_hsie.png"),
                                "overlay": str(Path(overlay_dir) / f"{candidate}_annotated.png") if polygon_count else None,
                                "polygons": polygon_count})
                progress.progress(idx / len(image_paths), text=f"Processed {idx}/{len(image_paths)}: {path.name}")
            if not records:
                raise ValueError("No readable images were processed.")
            # Human-readable run metadata.
            with open(Path(output_root) / "processing_summary.txt", 'w', encoding='utf-8') as f:
                f.write(f"Dataset: {dataset_label}\nInput image count: {len(image_paths)}\nProcessed image count: {len(records)}\nDepth model: {model_id}\nDevice: {device}\n")
                f.write("HSIE channel convention: H (hue), S (saturation), normalized predicted depth.\n")
                f.write("Depth values are relative estimates and are not metric measurements.\n")
            status.update(label=f"Completed: {len(records)} images processed", state="complete", expanded=False)

        st.session_state["records"] = records
        st.session_state["output_root"] = output_root
        st.session_state["work_root"] = work_root
        st.session_state["dataset_label"] = dataset_label
        st.success(f"Successfully processed {len(records)} images.")
    except Exception as exc:
        shutil.rmtree(work_root, ignore_errors=True)
        st.error(f"Processing failed: {exc}")
        st.info("Check that the ZIP contains image files and that the machine has internet access for the first model download.")

if st.session_state.get("records"):
    records = st.session_state["records"]
    output_root = st.session_state["output_root"]
    dataset_label = st.session_state.get("dataset_label", "dataset")
    st.divider()
    st.header("📊 Processing results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Images processed", len(records))
    m2.metric("Images with matched annotations", sum(1 for r in records if r['overlay']))
    m3.metric("Processing outputs", "HSI + HSIE + Depth")
    options = {f"{i+1}. {r['name']}": i for i, r in enumerate(records)}
    selected_label = st.selectbox("Choose an image to inspect", list(options.keys()))
    selected = records[options[selected_label]]
    tabs = st.tabs(["Original", "Depth map", "HSI", "HSIE", "Annotation overlay"])
    display_items = [(tabs[0], selected['original'], "Original input image"),
                     (tabs[1], selected['depth'], "Normalized relative depth estimate"),
                     (tabs[2], selected['hsi'], "HSI representation (OpenCV display uses the saved channel order)"),
                     (tabs[3], selected['hsie'], "HSIE fusion: hue, saturation, normalized depth"),
                     (tabs[4], selected['overlay'], "CVAT polygon overlay on HSIE image")]
    for tab, path, caption in display_items:
        with tab:
            if path and os.path.exists(path):
                image_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image_array is None:
                    st.warning("This output could not be read for preview.")
                elif image_array.ndim == 3:
                    st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
                else:
                    st.image(image_array, caption=caption, use_container_width=True)
                st.caption(os.path.basename(path))
            else:
                st.info("No annotation overlay for this image. Upload matching CVAT XML to generate overlays.")

    st.subheader("Quick gallery")
    for start in range(0, min(len(records), preview_count), 3):
        cols = st.columns(3)
        for col, record in zip(cols, records[start:start+3]):
            with col:
                st.image(record['original'], caption=f"Original: {record['name']}", use_container_width=True)
                st.image(record['depth'], caption="Depth map", use_container_width=True)
                st.image(record['hsie'], caption="HSIE", use_container_width=True)
                if record['overlay']:
                    st.image(record['overlay'], caption=f"Annotations: {record['polygons']} polygons", use_container_width=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in Path(output_root).rglob('*'):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(Path(output_root).parent)))
    st.download_button("⬇️ Download all processed results (ZIP)", data=zip_buffer.getvalue(),
                       file_name=f"{dataset_label}_processed.zip", mime="application/zip", type="primary", use_container_width=True)
    st.caption("The ZIP includes original copies, depth maps, HSI, HSIE, optional H/S/I channels, annotation overlays, and a processing summary.")

"""CrowdSense - crowd counting from a single photo.

Run with:  streamlit run app.py
Needs in the same folder: best_model.pth
Optional (for test images / benchmark): data/part_B/test_data/...
"""
import io
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "best_model.pth"
FULL_DIR = BASE / "data" / "part_B" / "test_data"      # full dataset (your laptop)
SAMPLE_DIR = BASE / "sample_data"                      # 15 sample photos (live server)
TEST_DIR = FULL_DIR if (FULL_DIR / "images").exists() else SAMPLE_DIR
USING_SAMPLE = TEST_DIR == SAMPLE_DIR
BASELINE_GUESS = 121.3          # average training count (the "always guess" baseline)
REPORTED_MAE, REPORTED_MSE, REPORTED_BASELINE = 9.03, 14.34, 71.53

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INK, MUTED, PLATFORM, PANEL, LINE = "#14202B", "#5B6773", "#EEF0F2", "#FFFFFF", "#D3D9DF"
SIGN, OK, WARN, BAD = "#0B4F8A", "#1E8E5A", "#F2B100", "#D6322E"

COLORMAPS = {
    "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
}

st.set_page_config(page_title="CrowdSense", page_icon=None, layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Theme file (light theme so every widget matches the design)
# ----------------------------------------------------------------------------
THEME_FILE = BASE / ".streamlit" / "config.toml"
THEME_TEXT = f"""[theme]
base = "light"
primaryColor = "{SIGN}"
backgroundColor = "{PLATFORM}"
secondaryBackgroundColor = "{PANEL}"
textColor = "{INK}"
font = "sans serif"

[browser]
gatherUsageStats = false
"""
if not THEME_FILE.exists():
    THEME_FILE.parent.mkdir(exist_ok=True)
    THEME_FILE.write_text(THEME_TEXT)
    st.info("Theme file created. Stop the app (Ctrl + C in the Anaconda Prompt) and run "
            "`streamlit run app.py` again to apply the new look.")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Barlow', 'Segoe UI', sans-serif; }
.stApp { background: %PLATFORM%; color: %INK%; }
[data-testid="stHeader"] { background: transparent; }
.stDeployButton, #MainMenu, footer { display: none !important; }
.block-container { padding-top: 1.1rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: %PANEL%; border-right: 1px solid %LINE%; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'Segoe UI', sans-serif; letter-spacing: 0; color: %INK%; }

.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  background: %SIGN%; padding: 14px 22px; border-radius: 6px; margin-bottom: 14px; flex-wrap: wrap; }
.topbar, .topbar * { color: #FFFFFF !important; }
.brand { font-family: 'Barlow Condensed', sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1; }
.tag { font-size: 1.02rem; opacity: .9; margin-top: 3px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,.16); padding: 6px 12px; border-radius: 4px; font-size: .92rem; }

.panel { background: %PANEL%; border: 1px solid %LINE%; border-radius: 6px; padding: 16px 18px; }
.empty { text-align: left; padding: 26px 24px; }
.empty b { font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem; }
.empty p { color: %MUTED%; margin: 6px 0 0 0; max-width: 62ch; line-height: 1.5; }

.board { background: %PANEL%; border: 1px solid %LINE%; border-radius: 6px; padding: 16px 18px 12px 18px; }
.board-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; }
.count { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 5.6rem; line-height: .9; color: %INK%; }
.count-label { color: %MUTED%; font-size: 1rem; margin-top: 4px; }
.plate { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.55rem; padding: 8px 16px;
  border-radius: 4px; white-space: nowrap; }
.advice { color: %INK%; margin: 12px 0 14px 0; font-size: 1rem; line-height: 1.4; }
.gauge-track { position: relative; height: 16px; background: #E3E7EB; border-radius: 3px; overflow: visible; }
.gauge-fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 3px; }
.tick { position: absolute; top: -4px; width: 2px; height: 24px; background: %INK%; opacity: .45; }
.tick.strong { opacity: 1; }
.gauge-labels { position: relative; height: 22px; margin-top: 8px; font-size: .85rem; color: %MUTED%; }
.gauge-labels span { position: absolute; white-space: nowrap; }

.facts { width: 100%; border-collapse: collapse; margin-top: 4px; }
.facts td { padding: 7px 0; border-bottom: 1px solid %LINE%; font-size: .98rem; }
.facts td:last-child { text-align: right; font-weight: 600; }
.facts tr:last-child td { border-bottom: none; }

.zones { border-collapse: separate; border-spacing: 4px; width: 100%; table-layout: fixed; }
.zones th { font-weight: 500; font-size: .85rem; color: %MUTED%; text-align: center; padding: 0; }
.zones td { text-align: center; padding: 12px 0; border-radius: 3px; font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.35rem; font-weight: 600; }
.zones td.hot { outline: 3px solid %INK%; outline-offset: 1px; }
.small { color: %MUTED%; font-size: .9rem; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid %LINE%; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 600; color: %MUTED%; }
.stTabs [aria-selected="true"] { color: %SIGN%; }
[data-testid="stMetric"] { background: %PANEL%; border: 1px solid %LINE%; border-radius: 6px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
.stButton > button, .stDownloadButton > button { border-radius: 4px; font-weight: 600; }
:focus-visible { outline: 3px solid %SIGN%; outline-offset: 2px; }
</style>
"""
for _k, _v in {"%INK%": INK, "%MUTED%": MUTED, "%PLATFORM%": PLATFORM, "%PANEL%": PANEL,
               "%LINE%": LINE, "%SIGN%": SIGN}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)          # real weights come from best_model.pth
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        layers, in_ch = [], 512
        for v in [512, 512, 512, 256, 128, 64]:
            layers += [nn.Conv2d(in_ch, v, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
            in_ch = v
        self.backend = nn.Sequential(*layers)
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))


@st.cache_resource(show_spinner="Loading the crowd-counting model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval(), device


def prepare(img_rgb, max_side=1024):
    """Resize (longest side <= 1024, sides divisible by 8) and normalise like training."""
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nh, nw = max(64, int(round(h * s / 8)) * 8), max(64, int(round(w * s / 8)) * 8)
    img = cv2.resize(img_rgb, (nw, nh))
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return img, torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


@st.cache_data(show_spinner="Counting people...", max_entries=64)
def analyse(data: bytes):
    """Photo bytes -> (resized photo, density map, milliseconds)."""
    rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    img, x = prepare(rgb)
    model, device = load_model()
    t0 = time.perf_counter()
    with torch.no_grad():
        d = model(x.to(device))[0, 0].cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return img, np.clip(d, 0, None), (time.perf_counter() - t0) * 1000


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def heat_rgb(d, size_hw, cmap, vmax=None):
    """Density map -> (coloured heatmap RGB, 0-1 intensity), resized to the photo."""
    h, w = size_hw
    big = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(big / max(vmax if vmax else big.max(), 1e-6), 0, 1)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), COLORMAPS[cmap])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), norm


def blend(img, d, cmap, opacity):
    heat, norm = heat_rgb(d, img.shape[:2], cmap)
    alpha = (norm * opacity)[..., None]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def zone_matrix(d, rows, cols):
    return np.array([[part.sum() for part in np.array_split(rp, cols, axis=1)]
                     for rp in np.array_split(d, rows, axis=0)])


def zone_id(r, c):
    return f"{chr(65 + r)}{c + 1}"


def draw_zones(img, zones):
    out = img.copy()
    h, w = out.shape[:2]
    rows, cols = zones.shape
    busiest = np.unravel_index(np.argmax(zones), zones.shape)
    scale = max(w / 1000, 0.5)
    for r in range(rows + 1):
        y = int(r * h / rows)
        cv2.line(out, (0, min(y, h - 1)), (w, min(y, h - 1)), (255, 255, 255), 2)
    for c in range(cols + 1):
        x = int(c * w / cols)
        cv2.line(out, (min(x, w - 1), 0), (min(x, w - 1), h), (255, 255, 255), 2)
    for r in range(rows):
        for c in range(cols):
            x0, x1, y0, y1 = int(c * w / cols), int((c + 1) * w / cols), int(r * h / rows), int((r + 1) * h / rows)
            if (r, c) == tuple(busiest):
                cv2.rectangle(out, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (242, 177, 0), 5)
            text = f"{zone_id(r, c)}: {zones[r, c]:.0f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, 2)
            org = (x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 + th) // 2)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def show(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:                                   # older Streamlit versions
        st.image(img, caption=caption, use_container_width=True)


# ----------------------------------------------------------------------------
# Status board
# ----------------------------------------------------------------------------
def status_for(count, capacity, warn_pct):
    ratio = count / capacity
    if ratio >= 1:
        over = count - capacity
        return dict(name="Overcrowded", bg=BAD, fg="#FFFFFF",
                    advice=f"About {over:.0f} people over capacity. Pause entry and keep exits clear.")
    if ratio >= warn_pct / 100:
        left = capacity - count
        return dict(name="Getting crowded", bg=WARN, fg=INK,
                    advice=f"Room for about {left:.0f} more people. Watch the busiest zone.")
    return dict(name="Normal", bg=OK, fg="#FFFFFF",
                advice=f"Well within capacity, with room for about {capacity - count:.0f} more people.")


def board_html(count, capacity, warn_pct, info):
    scale = max(capacity * 1.3, count * 1.1)
    fill = min(count / scale * 100, 100)
    warn_x = warn_pct / 100 * capacity / scale * 100
    cap_x = capacity / scale * 100
    return (
        '<div class="board"><div class="board-row"><div>'
        f'<div class="count">{count:.0f}</div><div class="count-label">people in view</div></div>'
        f'<div class="plate" style="background:{info["bg"]};color:{info["fg"]}">{info["name"]}</div></div>'
        f'<div class="advice">{info["advice"]}</div>'
        f'<div class="gauge-track"><div class="gauge-fill" style="width:{fill:.1f}%;background:{info["bg"]}"></div>'
        f'<div class="tick" style="left:{warn_x:.1f}%"></div><div class="tick strong" style="left:{cap_x:.1f}%"></div></div>'
        f'<div class="gauge-labels"><span style="left:{warn_x:.1f}%;transform:translateX(-100%)">Warning {warn_pct * capacity / 100:.0f}&nbsp;</span>'
        f'<span style="left:{cap_x:.1f}%">&nbsp;Capacity {capacity}</span></div></div>'
    )


def zones_html(zones):
    rows, cols = zones.shape
    peak = max(zones.max(), 1e-6)
    busiest = tuple(np.unravel_index(np.argmax(zones), zones.shape))
    html = '<table class="zones"><tr><th></th>' + "".join(f"<th>{c + 1}</th>" for c in range(cols)) + "</tr>"
    for r in range(rows):
        html += f"<tr><th>{chr(65 + r)}</th>"
        for c in range(cols):
            a = 0.07 + 0.88 * zones[r, c] / peak
            fg = "#FFFFFF" if a > 0.5 else INK
            hot = ' class="hot"' if (r, c) == busiest else ""
            html += f'<td{hot} style="background:rgba(11,79,138,{a:.2f});color:{fg}">{zones[r, c]:.0f}</td>'
        html += "</tr>"
    return html + "</table>"


def facts_html(pairs):
    return '<table class="facts">' + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in pairs) + "</table>"


# ----------------------------------------------------------------------------
# Test-image helpers
# ----------------------------------------------------------------------------
def test_images():
    folder = TEST_DIR / "images"
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []


def true_density(name):
    f = TEST_DIR / "density8" / name.replace(".jpg", ".npy")
    return np.load(f) if f.exists() else None


def log_result(key, source, count, status):
    log = st.session_state.setdefault("log", [])
    if log and log[-1]["key"] == key:
        return
    log.append(dict(key=key, time=datetime.now().strftime("%H:%M:%S"), source=source,
                    people=int(round(count)), status=status))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Venue settings")
    capacity = st.number_input("Capacity (people in view)", 10, 5000, 150, step=10,
                               help="The number of people that is acceptable in this camera view.")
    warn_pct = st.slider("Warn at (% of capacity)", 30, 95, 60, step=5)
    st.markdown("### Heatmap")
    cmap_name = st.selectbox("Colour style", list(COLORMAPS), index=0)
    opacity = st.slider("Heatmap strength", 0.2, 1.0, 0.7, step=0.05)
    st.markdown("### Zones")
    z_rows = st.select_slider("Rows", options=[1, 2, 3, 4, 5], value=3)
    z_cols = st.select_slider("Columns", options=[1, 2, 3, 4, 5], value=3)
    st.markdown(
        f'<p class="small">Model: CSRNet-style CNN.<br>Test error: {REPORTED_MAE} people on average '
        f'(MAE) on 316 unseen photos.</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
try:
    model, device = load_model()
except Exception as e:                                   # missing or broken model file
    st.error(f"The model could not be loaded: {e}\n\nCheck that `best_model.pth` is in the same "
             f"folder as app.py ({BASE}).")
    st.stop()

gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
st.markdown(
    '<div class="topbar"><div><div class="brand">CrowdSense</div>'
    '<div class="tag">Count a crowd from a single photo and see where it is building up.</div></div>'
    f'<div class="chips"><span class="chip">Model ready</span><span class="chip">{gpu}</span>'
    f'<span class="chip">Test MAE {REPORTED_MAE}</span></div></div>', unsafe_allow_html=True)

tab_an, tab_cmp, tab_batch, tab_test, tab_how = st.tabs(
    ["Count a crowd", "Compare with truth", "Several photos", "Test the model", "How it works"])

# ============================================================================
# Tab 1 - Count a crowd
# ============================================================================
with tab_an:
    source = st.radio("Photo source", ["Upload a photo", "Use the camera", "Pick a test photo"],
                      horizontal=True, label_visibility="collapsed")
    data, name, truth = None, None, None
    if source == "Upload a photo":
        up = st.file_uploader("Crowd photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up:
            data, name = up.getvalue(), up.name
    elif source == "Use the camera":
        cam = st.camera_input("Take a photo of a crowd", label_visibility="collapsed")
        if cam:
            data, name = cam.getvalue(), "camera photo"
    else:
        files = test_images()
        if files:
            pick = st.selectbox("Test photo", files, label_visibility="collapsed")
            data, name = (TEST_DIR / "images" / pick).read_bytes(), pick
            td = true_density(pick)
            truth = float(td.sum()) if td is not None else None
        else:
            st.warning("No test photos found. Put the ShanghaiTech `data` folder next to app.py.")

    if data is None:
        st.markdown(
            '<div class="panel empty"><b>Add a photo to count the crowd</b>'
            "<p>Upload a picture, take one with the camera, or pick a test photo. It works best on "
            "street, station or square scenes photographed from a distance or from above. "
            "Close-up shots and very large stadium crowds are less accurate.</p></div>",
            unsafe_allow_html=True)
    else:
        img, dens, ms = analyse(data)
        count = float(dens.sum())
        info = status_for(count, capacity, warn_pct)
        zones = zone_matrix(dens, z_rows, z_cols)
        busiest = tuple(np.unravel_index(np.argmax(zones), zones.shape))
        share = zones[busiest] / max(zones.sum(), 1e-6) * 100
        log_result(hash(data), name, count, info["name"])

        left, right = st.columns([1.55, 1], gap="large")
        with left:
            view = st.radio("View", ["Photo and heatmap", "Heatmap on photo", "Zone grid", "Original photo"],
                            horizontal=True, label_visibility="collapsed")
            overlay = blend(img, dens, cmap_name, opacity)
            zone_img = draw_zones(overlay, zones)
            if view == "Photo and heatmap":
                a, b = st.columns(2)
                with a:
                    show(img, "Original photo")
                with b:
                    show(overlay, "Where the crowd is")
                current = overlay
            elif view == "Heatmap on photo":
                show(overlay)
                current = overlay
            elif view == "Zone grid":
                show(zone_img, f"Busiest zone: {zone_id(*busiest)}")
                current = zone_img
            else:
                show(img)
                current = img
            d1, d2, _ = st.columns([1, 1, 1.2])
            d1.download_button("Download this view", png_bytes(current), file_name="crowdsense_view.png",
                               mime="image/png")
            zdf = pd.DataFrame(zones.round(1),
                               index=[chr(65 + r) for r in range(z_rows)],
                               columns=[str(c + 1) for c in range(z_cols)])
            d2.download_button("Download zone counts", zdf.to_csv().encode(), file_name="crowdsense_zones.csv",
                               mime="text/csv")
        with right:
            st.markdown(board_html(count, capacity, warn_pct, info), unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            pairs = [("Busiest zone", f"{zone_id(*busiest)} ({share:.0f}% of the crowd)"),
                     ("Counting time", f"{ms:.0f} ms")]
            if truth is not None:
                err = count - truth
                pairs = [("True count", f"{truth:.0f}"), ("Difference", f"{err:+.0f} people")] + pairs
            st.markdown('<div class="panel">' + facts_html(pairs) + "</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="panel"><b>People per zone</b>' + zones_html(zones) + "</div>",
                        unsafe_allow_html=True)

    log = st.session_state.get("log", [])
    if log:
        with st.expander(f"Session log ({len(log)} checks)"):
            ldf = pd.DataFrame(log).drop(columns="key")
            ldf.columns = ["Time", "Photo", "People", "Status"]
            st.dataframe(ldf, hide_index=True)
            c1, c2, _ = st.columns([1, 1, 3])
            c1.download_button("Download log", ldf.to_csv(index=False).encode(), file_name="crowdsense_log.csv",
                               mime="text/csv")
            if c2.button("Clear log"):
                st.session_state["log"] = []
                st.rerun()

# ============================================================================
# Tab 2 - Compare with truth
# ============================================================================
with tab_cmp:
    files = test_images()
    if not files:
        st.warning("This tab needs the ShanghaiTech test photos in `data/part_B/test_data`.")
    else:
        if st.session_state.get("cmp_pick") not in files:
            st.session_state["cmp_pick"] = files[0]

        def step(delta):
            i = files.index(st.session_state["cmp_pick"])
            st.session_state["cmp_pick"] = files[(i + delta) % len(files)]

        def pick_random():
            st.session_state["cmp_pick"] = random.choice(files)

        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.selectbox("Test photo", files, key="cmp_pick", label_visibility="collapsed")
        c2.button("Previous", on_click=step, args=(-1,), key="cmp_prev")
        c3.button("Next", on_click=step, args=(1,), key="cmp_next")
        c4.button("Random", on_click=pick_random, key="cmp_rand")

        name = st.session_state["cmp_pick"]
        img, dens, ms = analyse((TEST_DIR / "images" / name).read_bytes())
        gt = true_density(name)
        if gt is None:
            st.warning("Ground-truth density maps (density8 folder) were not found for this photo.")
        else:
            if gt.shape != dens.shape:                       # keep the total count when resizing
                total = gt.sum()
                gt = cv2.resize(gt, (dens.shape[1], dens.shape[0]))
                gt = gt * total / max(gt.sum(), 1e-6)
            vmax = max(gt.max(), dens.max())
            gt_img, _ = heat_rgb(gt, img.shape[:2], cmap_name, vmax)
            pr_img, _ = heat_rgb(dens, img.shape[:2], cmap_name, vmax)
            a, b, c = st.columns(3)
            with a:
                show(img, f"{name}")
            with b:
                show(gt_img, f"Ground truth: {gt.sum():.0f} people")
            with c:
                show(pr_img, f"Predicted: {dens.sum():.0f} people")
            err = dens.sum() - gt.sum()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True count", f"{gt.sum():.0f}")
            m2.metric("Predicted", f"{dens.sum():.0f}")
            m3.metric("Difference", f"{err:+.0f}")
            m4.metric("Error", f"{abs(err) / max(gt.sum(), 1) * 100:.1f}%")
            st.markdown('<p class="small">Both heatmaps use the same colour scale, so equal colours mean '
                        "equal crowd density.</p>", unsafe_allow_html=True)

# ============================================================================
# Tab 3 - Several photos
# ============================================================================
with tab_batch:
    ups = st.file_uploader("Upload several crowd photos", type=["jpg", "jpeg", "png"],
                           accept_multiple_files=True)
    if not ups:
        st.markdown('<div class="panel empty"><b>Compare several places or times</b>'
                    "<p>Upload two or more photos to count each one and see which is the most crowded.</p></div>",
                    unsafe_allow_html=True)
    else:
        rows_out, bar = [], st.progress(0.0)
        for i, f in enumerate(ups):
            _, d, t = analyse(f.getvalue())
            cnt = float(d.sum())
            zm = zone_matrix(d, z_rows, z_cols)
            rows_out.append({"Photo": f.name, "People": int(round(cnt)),
                             "Status": status_for(cnt, capacity, warn_pct)["name"],
                             "Busiest zone": zone_id(*np.unravel_index(np.argmax(zm), zm.shape)),
                             "Time (ms)": int(round(t))})
            bar.progress((i + 1) / len(ups))
        bar.empty()
        df = pd.DataFrame(rows_out)
        m1, m2, m3 = st.columns(3)
        m1.metric("Photos", len(df))
        m2.metric("Total people", int(df["People"].sum()))
        m3.metric("Most crowded", df.loc[df["People"].idxmax(), "Photo"])
        st.bar_chart(df.set_index("Photo")["People"], color=SIGN)
        st.dataframe(df, hide_index=True)
        st.download_button("Download results", df.to_csv(index=False).encode(),
                           file_name="crowdsense_batch.csv", mime="text/csv")

# ============================================================================
# Tab 4 - Test the model
# ============================================================================
with tab_test:
    files = test_images()
    if not files:
        st.warning("This tab needs the ShanghaiTech test photos in `data/part_B/test_data`.")
    else:
        st.markdown("Run the model on test photos it has never seen and compare each count with the "
                    "true count.")
        c1, c2 = st.columns([3, 1])
        n = c1.slider("Number of test photos", 10, len(files), min(50, len(files)), step=5)
        run = c2.button("Run test", type="primary")
        if run:
            rng = random.Random(7)
            chosen = rng.sample(files, n)
            names, trues, preds = [], [], []
            bar = st.progress(0.0, text="Testing...")
            for i, nm in enumerate(chosen):
                _, d, _ = analyse((TEST_DIR / "images" / nm).read_bytes())
                t = true_density(nm)
                if t is None:
                    continue
                names.append(nm)
                trues.append(float(t.sum()))
                preds.append(float(d.sum()))
                bar.progress((i + 1) / n, text=f"Testing... {i + 1} of {n}")
            bar.empty()
            st.session_state["bench"] = (names, np.array(trues), np.array(preds))

        if "bench" in st.session_state:
            names, trues, preds = st.session_state["bench"]
            err = preds - trues
            mae, rmse = np.abs(err).mean(), np.sqrt((err ** 2).mean())
            base = np.abs(trues - BASELINE_GUESS).mean()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average error (MAE)", f"{mae:.2f}", help="Average number of people the count is off by.")
            m2.metric("RMSE (MSE in papers)", f"{rmse:.2f}")
            m3.metric("Baseline error", f"{base:.2f}", help="Error if you always guess the average crowd size.")
            m4.metric("Better than baseline", f"{base / max(mae, 1e-6):.1f}x")

            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            ax.scatter(trues, preds, s=26, color=SIGN, alpha=0.75, edgecolor="white", linewidth=0.5)
            top = max(trues.max(), preds.max()) * 1.05
            ax.plot([0, top], [0, top], color=INK, linewidth=1, linestyle="--")
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_xlabel("True count")
            ax.set_ylabel("Predicted count")
            ax.set_title("Each dot is one photo (dashed line = perfect)", fontsize=11, color=INK)
            ax.grid(color=LINE, linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            g1, g2 = st.columns([1.2, 1], gap="large")
            with g1:
                st.pyplot(fig)
            plt.close(fig)
            with g2:
                st.markdown("**Largest mistakes**")
                worst = np.argsort(-np.abs(err))[:6]
                wdf = pd.DataFrame({"Photo": [names[i] for i in worst],
                                    "True": trues[worst].round(0).astype(int),
                                    "Predicted": preds[worst].round(0).astype(int),
                                    "Difference": err[worst].round(0).astype(int)})
                st.dataframe(wdf, hide_index=True)
                st.markdown('<p class="small">The biggest mistakes are almost always very dense crowds, '
                            "where the model counts fewer people than are really there.</p>",
                            unsafe_allow_html=True)

# ============================================================================
# Tab 5 - How it works
# ============================================================================
with tab_how:
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown("### From photo to head-count")
        st.markdown(
            "1. **Photo in.** The picture is resized and normalised the same way as during training.\n"
            "2. **Density map out.** A convolutional network (CSRNet-style: a pretrained VGG-16 front-end "
            "plus dilated convolutions) paints a map that is bright where people are packed together.\n"
            "3. **Add it up.** The sum of all values in the map is the number of people. Nobody has to be "
            "detected one by one, which is why it still works when people overlap.\n"
            "4. **Check the venue.** The count is compared with your capacity, and the map is split "
            "into zones to show where the crowd is building up.")
        st.markdown("### Good to know")
        st.markdown(
            "- Trained on ShanghaiTech Part B: 360 photos for training, 40 for validation, "
            "316 kept aside for testing.\n"
            "- It under-counts very dense crowds (roughly 250 people and more in one view).\n"
            "- It is a decision aid, not a safety system. Always keep a person in the loop.")
    with b:
        st.markdown("### Results on unseen photos")
        st.markdown(facts_html([("Test photos", "316"), ("Average error (MAE)", f"{REPORTED_MAE}"),
                                ("RMSE (MSE in papers)", f"{REPORTED_MSE}"),
                                ("Always-guess baseline", f"{REPORTED_BASELINE}"),
                                ("Training time", "about 2 h 20 min on an RTX 3050")]),
                    unsafe_allow_html=True)
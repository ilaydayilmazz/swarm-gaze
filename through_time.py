"""
Through Time
------------
A machine looks at real artworks from the first human drawings to the
20th century, reads how the depiction of people changes over time, and
makes its own piece from them.

  1. collect   find and download openly licensed artworks, oldest to newest:
               - Wikimedia Commons: cave paintings (only files whose licence allows reuse)
               - The Met Open Access (CC0): Egypt, Greece, medieval, Renaissance, ...
               - Art Institute of Chicago (CC0): Vincent van Gogh and others
               Every file's source, author and licence go into credits.csv.
  2. look      a vision model (CLIP) turns each work into a vector the
               machine can compare
  3. read      how strongly each work matches a few descriptions of how a
               person can be shown, plotted against time
  4. atlas     all works on one map (UMAP): near means "looks alike to the machine"
  5. journey   the machine's own piece: its gaze travels through the works in
               time order, and light flows through each work's own lines

Run it in Google Colab with `through_time_colab.ipynb`. Steps 2-4 need
`open_clip_torch` and `umap-learn`; step 5 needs only numpy and Pillow.
"""

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

UA = {"User-Agent": "ThroughTime/0.1 (student art project; contact via GitHub ilaydayilmazz)"}

# Periods to collect from The Met, oldest first: (label, start year, end year, search words)
MET_PERIODS = [
    ("Ancient Egypt", -3000, -500, "figure"),
    ("Ancient Greece and Rome", -700, 300, "figure"),
    ("Medieval", 500, 1400, "saint"),
    ("Renaissance", 1400, 1600, "portrait"),
    ("Baroque", 1600, 1750, "portrait"),
    ("18th-19th century", 1750, 1880, "portrait"),
]
REUSABLE = ("public domain", "pd", "cc0", "cc by", "cc-by", "cc by-sa", "cc-by-sa")


def _save(url, path, session):
    r = session.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    Path(path).write_bytes(r.content)


def collect(out="data/time", per_period=15, cave=15, artists=("Vincent van Gogh",), per_artist=15, pause=0.15):
    """Download artworks with their metadata. Returns the number saved."""
    import requests

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    rows = []

    def add(file, title, year, period, author, licence, source):
        rows.append({"file": file, "title": title, "year": year, "period": period,
                     "author": author, "licence": licence, "source": source})
        print(f"  {period:24s} {year:>6}  {str(title)[:60]}")

    # 1. Cave paintings from Wikimedia Commons, reusable licences only
    print("Cave paintings (Wikimedia Commons)")
    api = "https://commons.wikimedia.org/w/api.php"
    q = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
         "gsrsearch": "cave painting prehistoric filetype:bitmap", "gsrlimit": 50,
         "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 900}
    try:
        pages = s.get(api, params=q, headers=UA, timeout=60).json().get("query", {}).get("pages", {})
    except Exception as e:
        print("  Commons search failed:", e)
        pages = {}
    n = 0
    for p in pages.values():
        if n >= cave:
            break
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = (meta.get("LicenseShortName", {}).get("value") or "").lower()
        if not any(k in lic for k in REUSABLE):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        name = f"cave_{p['pageid']}.jpg"
        try:
            _save(url, out / name, s)
        except Exception:
            continue
        author = Image_strip(meta.get("Artist", {}).get("value", ""))
        add(name, p.get("title", "").replace("File:", ""), -15000, "Prehistoric (cave)", author,
            meta.get("LicenseShortName", {}).get("value", ""), info.get("descriptionurl", ""))
        n += 1
        time.sleep(pause)

    # 2. The Met, period by period (CC0)
    met = "https://collectionapi.metmuseum.org/public/collection/v1"
    for label, a, b, words in MET_PERIODS:
        print(label, "(The Met)")
        try:
            ids = s.get(f"{met}/search", params={"q": words, "hasImages": "true", "dateBegin": a, "dateEnd": b},
                        headers=UA, timeout=60).json().get("objectIDs") or []
        except Exception as e:
            print("  search failed:", e)
            continue
        n = 0
        for oid in ids:
            if n >= per_period:
                break
            try:
                o = s.get(f"{met}/objects/{oid}", headers=UA, timeout=60).json()
            except Exception:
                continue
            time.sleep(pause)
            url = o.get("primaryImageSmall")
            if not o.get("isPublicDomain") or not url:
                continue
            name = f"met_{oid}.jpg"
            try:
                _save(url, out / name, s)
            except Exception:
                continue
            add(name, o.get("title"), o.get("objectBeginDate"), label, o.get("artistDisplayName") or o.get("culture") or "",
                "CC0 (The Met Open Access)", o.get("objectURL"))
            n += 1

    # 3. Artists from the Art Institute of Chicago (CC0 images)
    aic = "https://api.artic.edu/api/v1/artworks/search"
    for artist in artists:
        print(artist, "(Art Institute of Chicago)")
        try:
            data = s.get(aic, params={"q": artist, "limit": 60,
                                      "fields": "id,title,date_start,artist_title,image_id,is_public_domain"},
                         headers=UA, timeout=60).json().get("data", [])
        except Exception as e:
            print("  search failed:", e)
            data = []
        n = 0
        for d in data:
            if n >= per_artist:
                break
            if not d.get("is_public_domain") or not d.get("image_id") or artist.split()[-1].lower() not in (d.get("artist_title") or "").lower():
                continue
            name = f"aic_{d['id']}.jpg"
            try:
                _save(f"https://www.artic.edu/iiif/2/{d['image_id']}/full/843,/0/default.jpg", out / name, s)
            except Exception:
                continue
            add(name, d.get("title"), d.get("date_start"), artist, d.get("artist_title"),
                "CC0 (Art Institute of Chicago)", f"https://www.artic.edu/artworks/{d['id']}")
            n += 1
            time.sleep(pause)

    rows.sort(key=lambda r: (r["year"] if isinstance(r["year"], (int, float)) else 0))
    with open(out / "credits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "year", "period", "author", "licence", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} works and credits.csv in {out}")
    return len(rows)


def Image_strip(html):
    """Remove HTML tags from a Commons metadata field."""
    import re
    return re.sub(r"<[^>]+>", "", html or "").strip()


def load_collection(folder="data/time"):
    rows = list(csv.DictReader(open(Path(folder) / "credits.csv", encoding="utf-8")))
    for r in rows:
        r["year"] = float(r["year"]) if r["year"] not in ("", None) else 0.0
        r["path"] = str(Path(folder) / r["file"])
    return sorted(rows, key=lambda r: r["year"])


# ---------------------------------------------------------------------------
# 2-4. Look, read and map (needs open_clip_torch and umap-learn)
# ---------------------------------------------------------------------------

READINGS = {
    "symbolic": "a flat, symbolic figure of a person",
    "lifelike": "a lifelike, realistic portrait of a person",
    "emotional": "an emotional, expressive face",
    "fragmented": "an abstract, fragmented human figure",
    "light": "a person dissolving into light and colour",
}


def look(rows, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    """CLIP image vectors for every work, plus how each one matches the readings."""
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        txt = model.encode_text(tok(list(READINGS.values())).to(device))
        txt = txt / txt.norm(dim=-1, keepdim=True)
        vecs = []
        for r in rows:
            x = prep(Image.open(r["path"]).convert("RGB")).unsqueeze(0).to(device)
            v = model.encode_image(x)
            vecs.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0])
        vecs = np.array(vecs)
        scores = vecs @ txt.cpu().numpy().T
    return vecs, scores


def plot_readings(rows, scores, path="out_time/readings.png"):
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    years = np.array([r["year"] for r in rows])
    z = (scores - scores.mean(0)) / (scores.std(0) + 1e-6)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = np.argsort(years)
    x = np.arange(len(rows))  # even spacing: prehistory and the 1880s on one readable axis
    for k, name in enumerate(READINGS):
        smooth = np.convolve(z[order, k], np.ones(5) / 5, mode="same")
        ax.plot(x, smooth, label=name, linewidth=2)
    ticks = np.linspace(0, len(rows) - 1, 8).astype(int)
    ax.set_xticks(ticks, [str(int(years[order][t])) for t in ticks])
    ax.set_ylabel("how strongly the machine reads it (z-score)")
    ax.set_title("How the machine reads the human figure over time")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def atlas(rows, vecs, path="out_time/atlas.png", thumb=48):
    """All works on one map: near means 'looks alike to the machine'."""
    import umap

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    xy = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=0).fit_transform(vecs)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)
    size = 1400
    canvas = Image.new("RGB", (size + thumb, size + thumb), (10, 10, 14))
    for (x, y), r in zip(xy, rows):
        im = Image.open(r["path"]).convert("RGB")
        im.thumbnail((thumb, thumb))
        canvas.paste(im, (int(x * size), int(y * size)))
    canvas.save(path)
    return path


# ---------------------------------------------------------------------------
# 5. Journey: the machine's own piece (numpy + Pillow only)
# ---------------------------------------------------------------------------

def journey(rows, out="out_time", seconds_per_work=1.6, fps=30, size=(960, 640), flies=80, seed=1):
    """
    The machine's gaze travels through the works in time order. On each work a
    DFO swarm looks for what stands out; light carrying that work's colours
    flows along its own lines. The canvas is never cleared, so each age flows
    into the next.
    """
    import imageio.v3 as iio
    import swarm_gaze as sg

    sg.rng = np.random.default_rng(seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    W, H = size
    frames, canvas = [], None
    per = max(2, int(seconds_per_work * fps))
    for k, r in enumerate(rows):
        im = Image.open(r["path"]).convert("RGB")
        im.thumbnail((W, H))
        board = Image.new("RGB", (W, H), (0, 0, 0))
        board.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        rgb = np.asarray(board)
        swarm = sg.Swarm(sg.salience_map(rgb), flies, 0.03)
        pig = sg.Pigment(rgb, flow=1.4, decay=0.955)
        if canvas is not None:
            pig.canvas = canvas
        for f in range(per):
            swarm.step()
            pig.emit(swarm)
            pig.move()
            frames.append(pig.frame())
        canvas = pig.canvas
        print(f"  {k + 1:3d}/{len(rows)}  {int(r['year']):>6}  {str(r['title'])[:50]}")
    iio.imwrite(Path(out) / "journey.mp4", np.stack(frames), fps=fps, codec="libx264", macro_block_size=1)
    Image.fromarray(frames[-1]).save(Path(out) / "journey_last.png")
    print(f"Saved {out}/journey.mp4 ({len(frames) / fps:.0f} s)")

"""
Bee Gaze
--------
A machine looks at ~100 works by one artist, learns that artist's visual
vocabulary, and then a colony of artificial bees (Artificial Bee Colony, ABC)
forages over a new image for the places that look most like the artist.
Every time a bee feeds, it paints one of the learned marks. The result is a
new image made only from what the machine learned.

Steps
  1. collect   download public-domain (CC0) works by one artist from The Met
  2. learn     cut every work into small patches and cluster them into a
               vocabulary of marks (k-means)
  3. look      score each region of a target image by how much it resembles
               the artist's signature marks (the machine's learned attention)
  4. forage    ABC bees search that attention map; where they feed, they paint
  5. humans    (later) compare the bees' attention with people's responses

Usage
  python bee_gaze.py collect --artist "Katsushika Hokusai" --n 100
  python bee_gaze.py run --target assets/drawing.jpg
"""

import argparse
import csv
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

rng = np.random.default_rng()

PATCH = 16          # size of one "mark" in pixels
VOCAB = 64          # number of marks in the learned vocabulary


# ---------------------------------------------------------------------------
# 1. Collect: public-domain works from The Met Open Access API
# ---------------------------------------------------------------------------

MET = "https://collectionapi.metmuseum.org/public/collection/v1"


def collect(artist, n=100, out="data/collection", pause=0.1):
    """Download up to n public-domain images by `artist` and save their metadata."""
    import requests

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = requests.get(f"{MET}/search", params={"q": artist, "hasImages": "true"}, timeout=30).json().get("objectIDs") or []
    print(f"{len(ids)} search results for {artist!r}")

    rows = []
    surname = artist.split()[-1].lower()
    for oid in ids:
        if len(rows) >= n:
            break
        try:
            obj = requests.get(f"{MET}/objects/{oid}", timeout=30).json()
        except Exception:
            continue
        time.sleep(pause)  # be polite to the API
        if not obj.get("isPublicDomain") or surname not in (obj.get("artistDisplayName") or "").lower():
            continue
        url = obj.get("primaryImageSmall") or obj.get("primaryImage")
        if not url:
            continue
        try:
            img = requests.get(url, timeout=60)
            img.raise_for_status()
        except Exception:
            continue
        path = out / f"{oid}.jpg"
        path.write_bytes(img.content)
        rows.append({"file": path.name, "title": obj.get("title"), "date": obj.get("objectDate"),
                     "artist": obj.get("artistDisplayName"), "url": obj.get("objectURL"), "licence": "CC0 (The Met Open Access)"})
        print(f"  {len(rows):3d}  {obj.get('title')}")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "date", "artist", "url", "licence"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} images to {out}")


def load_rgb(path, max_side):
    img = Image.open(path).convert("RGB")
    s = max_side / max(img.size)
    return np.asarray(img.resize((max(PATCH, round(img.width * s)), max(PATCH, round(img.height * s))), Image.LANCZOS))


# ---------------------------------------------------------------------------
# 2. Learn: a vocabulary of marks from the collection
# ---------------------------------------------------------------------------

def random_patches(rgb, k):
    h, w = rgb.shape[:2]
    ys = rng.integers(0, h - PATCH + 1, k)
    xs = rng.integers(0, w - PATCH + 1, k)
    return np.stack([rgb[y:y + PATCH, x:x + PATCH] for y, x in zip(ys, xs)])


def features(patches):
    """Describe each patch by its shape (normalised grey values) and its colour."""
    p = patches.astype(np.float32) / 255.0
    grey = p.mean(axis=3).reshape(len(p), -1)
    shape = grey - grey.mean(1, keepdims=True)
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-6
    colour = p.reshape(len(p), -1, 3).mean(1)
    return np.hstack([shape, colour * 2.0])


class Vocabulary:
    def __init__(self, folder, per_image=400, side=320):
        files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
        if not files:
            raise SystemExit(f"No images found in {folder}. Run the collect step first.")
        patches = np.concatenate([random_patches(load_rgb(f, side), per_image) for f in files])
        print(f"Learning from {len(files)} works, {len(patches)} patches")

        feats = features(patches)
        self.pca = PCA(n_components=32, random_state=0).fit(feats)
        z = self.pca.transform(feats)
        self.kmeans = MiniBatchKMeans(n_clusters=VOCAB, random_state=0, n_init=3, batch_size=4096).fit(z)
        labels = self.kmeans.labels_

        # how common each mark is; rarer marks are more "signature"
        counts = np.bincount(labels, minlength=VOCAB)
        self.signature = np.log(len(labels) / (counts + 1))
        self.signature /= self.signature.max()

        # keep a few real examples of each mark, closest to the cluster centre
        dist = np.linalg.norm(z - self.kmeans.cluster_centers_[labels], axis=1)
        self.typical_distance = np.median(dist)
        self.examples = []
        for c in range(VOCAB):
            idx = np.flatnonzero(labels == c)
            idx = idx[np.argsort(dist[idx])[:8]] if len(idx) else rng.integers(0, len(patches), 1)
            self.examples.append(patches[idx])
        self.paper = np.percentile(patches.reshape(-1, 3), 90, axis=0)  # the collection's background tone
        self.n_works = len(files)

    def describe(self, patches):
        z = self.pca.transform(features(patches))
        d = np.linalg.norm(z[:, None] - self.kmeans.cluster_centers_[None], axis=2)
        return d.argmin(1), d.min(1)

    def sheet(self, scale=3):
        """A contact sheet of the learned marks."""
        cols = 8
        tile = PATCH * scale
        sheet = Image.new("RGB", (cols * (tile + 4) + 4, (VOCAB // cols) * (tile + 4) + 4), (14, 15, 20))
        order = np.argsort(-self.signature)
        for i, c in enumerate(order):
            mark = Image.fromarray(self.examples[c][0]).resize((tile, tile), Image.NEAREST)
            sheet.paste(mark, (4 + (i % cols) * (tile + 4), 4 + (i // cols) * (tile + 4)))
        return sheet


# ---------------------------------------------------------------------------
# 3. Look: the machine's learned attention on a target image
# ---------------------------------------------------------------------------

def learned_attention(vocab, target, stride=4):
    """For each region: how much it looks like the artist's signature marks."""
    h, w = target.shape[:2]
    ys = np.arange(0, h - PATCH + 1, stride)
    xs = np.arange(0, w - PATCH + 1, stride)
    grid = np.stack([target[y:y + PATCH, x:x + PATCH] for y in ys for x in xs])
    label, dist = vocab.describe(grid)
    likeness = np.exp(-dist / vocab.typical_distance)
    # empty paper has no nectar: only regions with some mark in them count
    contrast = grid.astype(np.float32).mean(axis=3).reshape(len(grid), -1).std(1)
    marked = np.clip(contrast / (np.percentile(contrast, 90) + 1e-6), 0, 1)
    score = (likeness * (0.2 + 0.8 * vocab.signature[label]) * marked).reshape(len(ys), len(xs))
    label = label.reshape(len(ys), len(xs))

    att = Image.fromarray((score / score.max() * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    att = np.asarray(att.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0
    return att ** 1.5, label, stride


# ---------------------------------------------------------------------------
# 4. Forage: Artificial Bee Colony (Karaboga, 2005) on the attention map
# ---------------------------------------------------------------------------

class BeeColony:
    """
    Food sources are places in the image; their nectar is the learned attention.
    Employed bees refine their own source, onlookers pick sources in proportion
    to nectar, and a source abandoned after `limit` failed tries is replaced by a
    scout's random find. Fed places lose some nectar, so the colony moves on.
    """

    def __init__(self, attention, n_sources=40, limit=15):
        self.att = attention.copy()
        self.full = attention.copy()       # nectar never regrows beyond what the flower had
        self.use, self.regrow = 0.97, 0.01
        self.h, self.w = attention.shape
        self.n, self.limit = n_sources, limit
        self.src = rng.random((self.n, 2)) * [self.w - 1, self.h - 1]
        self.trials = np.zeros(self.n, int)
        self.visits = np.zeros_like(attention)
        self.fed = []  # places where a bee fed in this cycle

    def nectar(self, p):
        return float(self.att[int(p[1]), int(p[0])])

    def clip(self, p):
        return np.clip(p, [0, 0], [self.w - 1, self.h - 1])

    def try_neighbour(self, i):
        k = rng.choice([j for j in range(self.n) if j != i])
        d = rng.integers(2)
        v = self.src[i].copy()
        v[d] += rng.uniform(-1, 1) * (self.src[i][d] - self.src[k][d])
        v = self.clip(v + rng.normal(0, 2, 2))
        if self.nectar(v) > self.nectar(self.src[i]):
            self.src[i], self.trials[i] = v, 0
        else:
            self.trials[i] += 1
        self.feed(self.src[i])

    def feed(self, p):
        x, y = int(p[0]), int(p[1])
        self.visits[y, x] += 1
        self.fed.append((x, y))
        y0, y1, x0, x1 = max(0, y - 6), y + 7, max(0, x - 6), x + 7
        self.att[y0:y1, x0:x1] *= self.use  # nectar is used up here for a while

    def cycle(self):
        self.fed = []
        for i in range(self.n):                     # employed bees
            self.try_neighbour(i)
        nectar = np.array([self.nectar(s) for s in self.src]) + 1e-6
        for i in rng.choice(self.n, self.n, p=nectar / nectar.sum()):  # onlooker bees
            self.try_neighbour(i)
        for i in np.flatnonzero(self.trials > self.limit):            # scout bees
            self.src[i] = rng.random(2) * [self.w - 1, self.h - 1]
            self.trials[i] = 0
        self.att += (self.full - self.att) * self.regrow  # nectar slowly regrows


# ---------------------------------------------------------------------------
# 5. Paint: every feeding leaves one learned mark
# ---------------------------------------------------------------------------

class Canvas:
    def __init__(self, vocab, target, labels, stride):
        self.vocab, self.target, self.labels, self.stride = vocab, target, labels, stride
        h, w = target.shape[:2]
        self.img = np.ones((h, w, 3), np.float32) * vocab.paper
        r = np.linspace(-1, 1, PATCH)
        self.mask = np.clip(1 - np.hypot(*np.meshgrid(r, r)), 0, 1)[..., None] * 0.75  # soft round brush

    def paint(self, x, y):
        h, w = self.img.shape[:2]
        x0 = min(max(0, x - PATCH // 2), w - PATCH)
        y0 = min(max(0, y - PATCH // 2), h - PATCH)
        gy = min(y0 // self.stride, self.labels.shape[0] - 1)
        gx = min(x0 // self.stride, self.labels.shape[1] - 1)
        examples = self.vocab.examples[self.labels[gy, gx]]
        mark = examples[rng.integers(len(examples))].astype(np.float32)
        region = self.img[y0:y0 + PATCH, x0:x0 + PATCH]
        region[:] = region * (1 - self.mask) + mark * self.mask

    def frame(self):
        return self.img.clip(0, 255).astype(np.uint8)


def heat_overlay(target, values, colour=(255, 196, 64)):
    v = np.asarray(Image.fromarray((values / (values.max() + 1e-6) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
    v = np.sqrt(v / (v.max() + 1e-6))[..., None]
    base = target.astype(np.float32) * 0.3
    return Image.fromarray((base * (1 - v) + np.array(colour) * v).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 6. Humans (to do): compare the colony's attention with people's responses
# ---------------------------------------------------------------------------

def compare_with_humans(bee_visits, human_csv):
    """
    Placeholder. Planned input: a CSV of real human responses for the same image,
    e.g. clicks on 'where did you look first?' (x, y) or emotion labels from ArtEmis.
    Planned output: correlation between the bees' visit map and the human map.
    No human data is generated or simulated here.
    """
    raise NotImplementedError("Add real human response data first.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(collection, target_path, out="out", cycles=500, sources=40, seed=None, fps=30):
    global rng
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(collection)
    vocab.sheet().save(out / "1_vocabulary.png")

    target = load_rgb(target_path, 640)
    h, w = target.shape[0] - target.shape[0] % 2, target.shape[1] - target.shape[1] % 2
    target = np.ascontiguousarray(target[:h, :w])
    attention, labels, stride = learned_attention(vocab, target)
    heat_overlay(target, attention).save(out / "2_learned_attention.png")

    colony = BeeColony(attention, n_sources=sources)
    canvas = Canvas(vocab, target, labels, stride)
    frames = []
    for c in range(cycles):
        colony.cycle()
        for x, y in colony.fed:
            canvas.paint(x, y)
        if c % 2 == 0:
            frames.append(canvas.frame())
        if c % 100 == 0:
            print(f"cycle {c:4d}/{cycles}")

    Image.fromarray(canvas.frame()).save(out / "4_bee_painting.png")
    heat_overlay(target, colony.visits).save(out / "3_bee_visits.png")
    np.save(out / "bee_visits.npy", colony.visits)
    iio.imwrite(out / "5_bees_painting.mp4", np.stack(frames), fps=fps, codec="libx264", macro_block_size=1)
    json.dump({"collection": str(collection), "works_learned_from": vocab.n_works, "target": str(target_path),
               "cycles": cycles, "food_sources": sources, "seed": seed}, open(out / "run.json", "w"), indent=2)
    print(f"Saved results in {out}/")


def main():
    ap = argparse.ArgumentParser(description="Bee Gaze: a bee colony paints with an artist's learned marks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="download public-domain works from The Met")
    c.add_argument("--artist", default="Katsushika Hokusai")
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--out", default="data/collection")
    r = sub.add_parser("run", help="learn, look, forage and paint")
    r.add_argument("--collection", default="data/collection")
    r.add_argument("--target", default=str(Path(__file__).parent / "assets" / "drawing.jpg"))
    r.add_argument("--out", default="out")
    r.add_argument("--cycles", type=int, default=500)
    r.add_argument("--sources", type=int, default=40)
    r.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.artist, a.n, a.out)
    else:
        run(a.collection, a.target, a.out, a.cycles, a.sources, a.seed)


if __name__ == "__main__":
    main()

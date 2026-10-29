"""
Machine Feels
-------------
A very simple prototype of one question:
If a machine learns what people feel in front of art, what would it draw?

  1. look   the machine reads a few visual qualities of an artwork
            (warmth, brightness, colour, line energy, contrast) and its palette
  2. feel   it guesses which feelings the work evokes, using the 9 emotion
            categories of the ArtEmis dataset
              - with a trained model (see `learn`), the guess comes from real people
              - without one, it uses a small hand-made STARTER RULE (not human data)
  3. draw   a flock of birds (boids) draws an ORIGINAL picture: the feeling decides
            how they fly and which colours they carry; the artwork only decides
            where they start and which palette they borrow
  4. learn  (needs real data) fit the feel step on ArtEmis: images + people's emotions

The machine does not feel anything. Whatever it "feels" is borrowed from us.

Usage
  python machine_feels.py look-and-draw assets/drawing.jpg
  python machine_feels.py learn --artemis artemis.csv --images wikiart/
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans

rng = np.random.default_rng()

# The 9 emotion categories used in ArtEmis (Achlioptas et al., 2021),
# placed on a valence (unpleasant -> pleasant) / arousal (calm -> energetic) plane.
EMOTIONS = {
    "amusement":     (0.7, 0.5),
    "awe":           (0.6, 0.7),
    "contentment":   (0.8, -0.5),
    "excitement":    (0.7, 0.9),
    "anger":         (-0.7, 0.8),
    "disgust":       (-0.8, 0.3),
    "fear":          (-0.6, 0.7),
    "sadness":       (-0.7, -0.6),
    "something else": (0.0, 0.0),
}
FEATURES = ["warmth", "brightness", "saturation", "line_energy", "contrast"]
MODEL = Path(__file__).parent / "feel_model.json"


# ---------------------------------------------------------------------------
# 1. Look
# ---------------------------------------------------------------------------

def load(path, side=480):
    img = Image.open(path).convert("RGB")
    s = side / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def look(img):
    """A few simple visual qualities, each between 0 and 1, plus the work's palette."""
    a = np.asarray(img, np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), np.float32) / 255.0

    seen = {
        "warmth": float(np.clip(0.5 + (r - b).mean(), 0, 1)),
        "brightness": float(lum.mean()),
        "saturation": float(np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0).mean()),
        "line_energy": float(np.clip(edges.mean() * 2.5, 0, 1)),
        "contrast": float(np.clip(lum.std() * 3, 0, 1)),
    }
    pixels = a.reshape(-1, 3)[rng.choice(a.shape[0] * a.shape[1], 4000, replace=False)]
    palette = KMeans(n_clusters=6, n_init=3, random_state=0).fit(pixels).cluster_centers_
    salient = np.argwhere(edges >= np.percentile(edges, 97))  # where the lines are strongest
    if len(salient) == 0:
        salient = np.argwhere(np.ones_like(edges, bool))
    return seen, palette, salient, edges.shape


# ---------------------------------------------------------------------------
# 2. Feel
# ---------------------------------------------------------------------------

def starter_rule(seen):
    """
    Hand-made placeholder, NOT learned from people:
    warm, bright works lean pleasant; busy, high-contrast works lean energetic.
    It only exists so the prototype runs before real data is added.
    """
    valence = (seen["warmth"] - 0.5) * 1.2 + (seen["brightness"] - 0.5) * 1.0 + (seen["saturation"] - 0.3) * 0.6
    arousal = (seen["line_energy"] - 0.3) * 1.6 + (seen["contrast"] - 0.4) * 1.0 + (seen["saturation"] - 0.3) * 0.6
    v, a = np.tanh(valence), np.tanh(arousal)
    names = list(EMOTIONS)
    d = np.array([np.hypot(v - ev, a - ea) for ev, ea in EMOTIONS.values()])
    p = np.exp(-d * 3)
    return dict(zip(names, p / p.sum())), (float(v), float(a))


def learned_rule(seen, model):
    """Feelings predicted by a model trained on real people's responses (see `learn`)."""
    x = np.array([seen[f] for f in FEATURES])
    z = (x - model["mean"]) / model["std"]
    logits = np.array(model["coef"]) @ z + np.array(model["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    feelings = dict(zip(model["classes"], p))
    v = sum(p_ * EMOTIONS[e][0] for e, p_ in feelings.items())
    a = sum(p_ * EMOTIONS[e][1] for e, p_ in feelings.items())
    return feelings, (float(v), float(a))


def feel(seen):
    if MODEL.exists():
        return (*learned_rule(seen, json.loads(MODEL.read_text())), "learned from people")
    return (*starter_rule(seen), "starter rule (not human data)")


# ---------------------------------------------------------------------------
# 3. Draw: a flock of birds paints the feeling
# ---------------------------------------------------------------------------

def mood_palette(palette, valence, arousal):
    """Borrow the work's colours, then let the feeling shift them."""
    hsv = np.array(Image.fromarray((palette[None] * 255).astype(np.uint8)).convert("HSV"), np.float32)[0] / 255
    hsv[:, 1] = np.clip(hsv[:, 1] * (1 + 0.6 * arousal) + 0.15, 0, 1)                  # energy -> vivid
    hsv[:, 2] = np.clip(hsv[:, 2] * 0.6 + 0.35 + 0.2 * valence, 0.25, 1)              # pleasant -> lighter
    hsv[:, 0] = (hsv[:, 0] + (0.03 if valence > 0 else -0.05)) % 1                    # small warm/cool shift
    rgb = Image.fromarray((hsv[None] * 255).astype(np.uint8), "HSV").convert("RGB")
    return np.asarray(rgb, np.float32)[0]


def draw(palette, salient, shape, valence, arousal, size=640, steps=420, birds=140):
    h, w = shape
    s = size / max(h, w)
    W, H = round(w * s) // 2 * 2, round(h * s) // 2 * 2  # even sizes for video

    # how the flock behaves, decided by the feeling
    speed = 1.5 + 2.5 * (arousal + 1) / 2          # calm -> slow, energetic -> fast
    cohesion = 0.002 + 0.01 * (valence + 1) / 2    # pleasant -> gather together
    jitter = 0.05 + 0.6 * max(arousal, 0)          # energetic -> restless
    alpha = 0.06 + 0.06 * (1 - (arousal + 1) / 2)  # calm -> softer, layered strokes

    colours = mood_palette(palette, valence, arousal)
    paper = (238, 233, 224) if valence >= 0 else (24, 24, 32)

    # birds start where the artwork's lines were strongest: what the machine "saw"
    start = salient[rng.choice(len(salient), birds)] * s
    pos = start[:, ::-1].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, birds)
    vel = np.stack([np.cos(ang), np.sin(ang)], 1) * speed
    col = colours[rng.integers(len(colours), size=birds)]

    canvas = Image.new("RGB", (W, H), paper)
    layer = ImageDraw.Draw(canvas, "RGBA")
    frames = []
    for t in range(steps):
        centre = pos.mean(0)
        diff = pos[:, None] - pos[None]
        dist = np.linalg.norm(diff, axis=2) + 1e-6
        near = dist < 40
        align = (near[..., None] * vel[None]).sum(1) / near.sum(1, keepdims=True)  # fly like your neighbours
        apart = ((dist < 12)[..., None] * diff / dist[..., None] ** 2).sum(1)        # don't crash
        vel += (centre - pos) * cohesion + (align - vel) * 0.05 + apart * 1.5 + rng.normal(0, jitter, vel.shape)
        vel *= speed / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6)
        new = pos + vel
        # soft walls
        vel[(new[:, 0] < 10) | (new[:, 0] > W - 10), 0] *= -1
        vel[(new[:, 1] < 10) | (new[:, 1] > H - 10), 1] *= -1
        for (x0, y0), (x1, y1), c in zip(pos, new, col):
            layer.line([(x0, y0), (x1, y1)], fill=(*c.astype(int), int(255 * alpha)), width=2)
        pos = np.clip(new, 0, [W - 1, H - 1])
        if t % 3 == 0:
            frames.append(np.asarray(canvas))
    return canvas, frames


# ---------------------------------------------------------------------------
# 4. Learn (needs real data)
# ---------------------------------------------------------------------------

def learn(artemis_csv, image_dir, max_images=5000):
    """
    Fit the feel step on real people's responses.
    artemis_csv: the ArtEmis file with columns `painting` and `emotion`
    image_dir:   folder of the matching WikiArt images (painting name + .jpg)
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(artemis_csv)
    majority = df.groupby("painting")["emotion"].agg(lambda e: e.value_counts().index[0])
    X, y = [], []
    for name, emotion in list(majority.items())[:max_images]:
        hits = list(Path(image_dir).rglob(f"{name}.jpg"))
        if not hits:
            continue
        seen, *_ = look(load(hits[0], 256))
        X.append([seen[f] for f in FEATURES])
        y.append(emotion)
    X = np.array(X)
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000).fit((X - mean) / std, y)
    MODEL.write_text(json.dumps({"classes": list(clf.classes_), "coef": clf.coef_.tolist(),
                                 "intercept": clf.intercept_.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                                 "trained_on": len(y)}, indent=1))
    print(f"Learned from {len(y)} artworks and the people who responded to them. Saved {MODEL.name}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def feelings_card(feelings, source, path):
    top = sorted(feelings.items(), key=lambda kv: -kv[1])
    card = Image.new("RGB", (520, 60 + 34 * len(top)), (18, 19, 26))
    d = ImageDraw.Draw(card)
    d.text((16, 14), f"What the machine 'feels' ({source})", fill=(235, 232, 240))
    for i, (name, p) in enumerate(top):
        y = 50 + i * 34
        d.text((16, y + 6), name, fill=(170, 168, 185))
        d.rectangle([150, y + 4, 150 + int(330 * p / top[0][1]), y + 22], fill=(255, 178, 74))
        d.text((150 + int(330 * p / top[0][1]) + 6, y + 6), f"{p:.0%}", fill=(170, 168, 185))
    card.save(path)


def look_and_draw(image_path, out="out_feels", seed=None):
    global rng
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    seen, palette, salient, shape = look(load(image_path))
    feelings, (v, a), source = feel(seen)
    print("Seen:", {k: round(x, 2) for k, x in seen.items()})
    print(f"Feels ({source}):", ", ".join(f"{k} {p:.0%}" for k, p in sorted(feelings.items(), key=lambda kv: -kv[1])[:3]))

    canvas, frames = draw(palette, salient, shape, v, a)
    canvas.save(out / "machine_drawing.png")
    feelings_card(feelings, source, out / "machine_feelings.png")
    iio.imwrite(out / "machine_drawing.mp4", np.stack(frames), fps=30, codec="libx264", macro_block_size=1)
    json.dump({"image": str(image_path), "seen": seen, "feelings": feelings, "valence": v, "arousal": a,
               "source": source, "seed": seed}, open(out / "run.json", "w"), indent=2)
    print(f"Saved results in {out}/")


def main():
    ap = argparse.ArgumentParser(description="Machine Feels: look, feel and draw")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("look-and-draw")
    d.add_argument("image", nargs="?", default=str(Path(__file__).parent / "assets" / "drawing.jpg"))
    d.add_argument("--out", default="out_feels")
    d.add_argument("--seed", type=int)
    l = sub.add_parser("learn")
    l.add_argument("--artemis", required=True)
    l.add_argument("--images", required=True)
    a = ap.parse_args()
    if a.cmd == "learn":
        learn(a.artemis, a.images)
    else:
        look_and_draw(a.image, a.out, a.seed)


if __name__ == "__main__":
    main()

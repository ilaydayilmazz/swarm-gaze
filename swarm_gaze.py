"""
Swarm Gaze
----------
What does a machine notice when it looks at a work of art?

A swarm based on Dispersive Flies Optimisation (DFO) explores an image,
lingers where something stands out, and releases particles carrying the
colour it is looking at. The particles drift on a flow field and fade,
turning the swarm's attention into "data pigment".

Outputs:
  - swarm_gaze.mp4   the moving piece
  - pigment.png      the last frame
  - attention.png    where the swarm lingered (heat map over the image)
  - attention.npy    the raw attention grid, for analysis

Usage:
  python swarm_gaze.py                      # uses my drawing in assets/drawing.jpg
  python swarm_gaze.py --image my_art.jpg --frames 600 --flies 60
"""

import argparse

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

DEFAULT_IMAGE = Path(__file__).parent / "assets" / "drawing.jpg"

rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# 1. The image
# ---------------------------------------------------------------------------

def load_image(path, max_side=800):
    img = Image.open(path).convert("RGB")
    scale = min(max_side / max(img.size), 2.0)  # shrink big images, enlarge small ones (up to 2x)
    if scale != 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# 2. Noticing: a simple salience map (edges + saturated colour)
# ---------------------------------------------------------------------------

def salience_map(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn), 0.0)  # saturation weighted by brightness

    # Sobel edges
    p = np.pad(lum, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    edges = np.hypot(gx, gy)

    sal = 0.6 * edges + 0.4 * sat
    sal = sal / (sal.max() + 1e-6)
    return sal ** 1.2


# ---------------------------------------------------------------------------
# 3. Lingering: a DFO swarm with "boredom"
# ---------------------------------------------------------------------------

class Swarm:
    CELL = 8           # size of one attention cell in pixels
    THRESHOLD = 0.25   # salience needed for a fly to linger

    def __init__(self, sal, n_flies=60, disturbance=0.03):
        self.sal = sal
        self.h, self.w = sal.shape
        self.n = n_flies
        self.dt = disturbance
        gh, gw = -(-self.h // self.CELL), -(-self.w // self.CELL)
        self.interest = np.ones((gh, gw), np.float32)   # drops where flies have been (boredom)
        self.attention = np.zeros((gh, gw), np.float32)  # how long flies lingered in each cell
        self.pos = rng.random((self.n, 2)) * [self.w, self.h]
        self.fit = np.array([self.fitness(x, y) for x, y in self.pos])

    def cell(self, x, y):
        return int(y) // self.CELL, int(x) // self.CELL

    def fitness(self, x, y):
        if not (1 <= x < self.w - 1 and 1 <= y < self.h - 1):
            return 0.0
        return float(self.sal[int(y), int(x)] * self.interest[self.cell(x, y)])

    def clamp(self, x, y):
        return min(self.w - 2, max(1, x)), min(self.h - 2, max(1, y))

    def step(self):
        radius = max(3, self.w / 120)
        for i in range(self.n):
            x, y = self.pos[i]

            # Lingering: look around nearby when standing on something interesting
            if self.fit[i] > self.THRESHOLD and rng.random() < 0.8:
                cx, cy = self.clamp(x + rng.normal() * radius, y + rng.normal() * radius)
                cf = self.fitness(cx, cy)
                if cf >= self.fit[i] * 0.85:
                    self.pos[i] = cx, cy
                c = self.cell(*self.pos[i])
                self.attention[c] += 1
                self.interest[c] *= 0.88
                self.fit[i] = self.fitness(*self.pos[i])
                continue

            # Exploring: DFO update
            # best neighbour on a ring, pulled towards the best of a small random sample
            left, right = (i - 1) % self.n, (i + 1) % self.n
            nb = left if self.fit[left] > self.fit[right] else right
            sample = rng.integers(0, self.n, 4)
            best = max([nb, *sample], key=lambda j: self.fit[j])

            new = self.pos[nb] + rng.random(2) * (self.pos[best] - self.pos[i])
            # disturbance: each dimension may be reset at random
            if rng.random() < self.dt:
                new[0] = rng.random() * self.w
            if rng.random() < self.dt:
                new[1] = rng.random() * self.h
            new += rng.normal(size=2) * 2
            self.pos[i] = self.clamp(*new)
            self.fit[i] = self.fitness(*self.pos[i])

        # interest slowly comes back
        self.interest += (1 - self.interest) * 0.0015


# ---------------------------------------------------------------------------
# 4. Pigment: particles that carry colour on a flow field
# ---------------------------------------------------------------------------

class Pigment:
    def __init__(self, rgb, capacity=40000, flow=1.4, decay=0.965):
        self.rgb = rgb.astype(np.float32) / 255.0
        self.h, self.w = rgb.shape[:2]
        self.flow = flow
        self.decay = decay
        self.canvas = np.zeros((self.h, self.w, 3), np.float32)
        self.cap = capacity
        self.p = np.zeros((capacity, 2), np.float32)
        self.v = np.zeros((capacity, 2), np.float32)
        self.col = np.zeros((capacity, 3), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.cursor = 0
        self.t = 0.0

    def emit(self, swarm):
        for (x, y), f in zip(swarm.pos, swarm.fit):
            k = 4 if f > swarm.THRESHOLD else 1
            # take the most vivid colour in a small window around the fly
            x0, y0 = max(0, int(x) - 3), max(0, int(y) - 3)
            patch = self.rgb[y0:int(y) + 4, x0:int(x) + 4].reshape(-1, 3)
            colour = patch[np.argmax(patch.max(1) - patch.min(1))]
            colour = np.minimum(1, colour * 0.9 / max(colour.max(), 0.08))  # brighten
            idx = (self.cursor + np.arange(k)) % self.cap
            self.cursor = (self.cursor + k) % self.cap
            self.p[idx] = [x, y] + (rng.random((k, 2)) - 0.5) * 4
            self.v[idx] = rng.random((k, 2)) - 0.5
            self.col[idx] = colour
            self.life[idx] = self.max_life[idx] = 120 + rng.random(k) * 160

    def move(self):
        alive = self.life > 0
        p, v = self.p[alive], self.v[alive]
        x, y = p[:, 0], p[:, 1]
        angle = (np.sin(x * 0.006 + self.t * 0.4) + np.cos(y * 0.005 - self.t * 0.3)
                 + np.sin((x + y) * 0.0032 + self.t * 0.2)) * np.pi * 0.75
        v = v * 0.9 + np.stack([np.cos(angle), np.sin(angle)], 1) * self.flow * 0.1
        p = p + v
        self.p[alive], self.v[alive] = p, v
        self.life[alive] -= 1

        inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < self.w) & (p[:, 1] < self.h)
        idx = np.flatnonzero(alive)
        self.life[idx[~inside]] = 0
        idx, p = idx[inside], p[inside]

        # deposit colour where each particle is
        strength = 0.035 * np.sqrt(np.clip(self.life[idx] / self.max_life[idx], 0, 1))
        np.add.at(self.canvas, (p[:, 1].astype(int), p[:, 0].astype(int)), self.col[idx] * strength[:, None])
        self.canvas *= self.decay
        self.t += 0.01
        return len(idx)

    def frame(self, background=(8, 9, 14)):
        """Soft tone mapping so bright areas glow instead of clipping."""
        x = self.canvas * 2.2
        bg = np.array(background, np.float32)
        return (bg + (255 - bg) * x / (1 + x)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Outputs
# ---------------------------------------------------------------------------

def attention_image(img, swarm):
    """The machine's view: the image dimmed, with a warm glow where the swarm lingered."""
    base = np.asarray(img, np.float32) * 0.32
    heat = np.sqrt(swarm.attention / (swarm.attention.max() + 1e-6))
    heat = np.kron(heat, np.ones((swarm.CELL, swarm.CELL)))[: swarm.h, : swarm.w, None]
    warm = np.array([255, 178, 74], np.float32)
    out = base * (1 - heat * 0.85) + warm * heat * 0.85
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description="Swarm Gaze: a swarm's attention as data pigment")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="path to an image (default: assets/drawing.jpg)")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--flies", type=int, default=60)
    ap.add_argument("--disturbance", type=float, default=0.03)
    ap.add_argument("--flow", type=float, default=1.4)
    ap.add_argument("--trail", type=float, default=0.965)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    img = load_image(args.image)
    w, h = img.width - img.width % 2, img.height - img.height % 2  # video needs even sizes
    img = img.crop((0, 0, w, h))
    rgb = np.asarray(img)

    swarm = Swarm(salience_map(rgb), args.flies, args.disturbance)
    pigment = Pigment(rgb, flow=args.flow, decay=args.trail)

    frames = []
    for f in range(args.frames):
        for _ in range(2):  # two swarm steps per video frame
            swarm.step()
            pigment.emit(swarm)
            alive = pigment.move()
        frames.append(pigment.frame())
        if f % 100 == 0:
            print(f"frame {f:4d}/{args.frames}  particles: {alive}")

    iio.imwrite("swarm_gaze.mp4", np.stack(frames), fps=args.fps, codec="libx264")
    Image.fromarray(frames[-1]).save("pigment.png")
    attention_image(img, swarm).save("attention.png")
    np.save("attention.npy", swarm.attention)
    print("Saved swarm_gaze.mp4, pigment.png, attention.png, attention.npy")


if __name__ == "__main__":
    main()

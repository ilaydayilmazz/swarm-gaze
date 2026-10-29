# Swarm Gaze

> What does a machine notice when it looks at a work of art?

**Swarm Gaze** is an early study in computational art. A swarm of virtual flies explores an image, lingers where something stands out, and turns its attention into flowing colour, like data pigment.

| The swarm's attention on my drawing | Its gaze as data pigment |
| --- | --- |
| ![Where the swarm lingered on my drawing](docs/attention.png) | ![The swarm's gaze as flowing pigment](docs/pigment.png) |

The project has two versions:

- **Browser (`index.html`)**: an interactive piece. Open it (or the GitHub Pages link) and upload any image to watch the swarm look at it.
- **Python (`swarm_gaze.py`)**: the research version. It renders a video and saves the swarm's attention map for analysis.

### Run the Python version

```bash
pip install -r requirements.txt
python swarm_gaze.py                          # uses my drawing in assets/drawing.jpg
python swarm_gaze.py --image my_art.jpg --frames 600 --flies 60 --seed 7
```

Outputs: `swarm_gaze.mp4` (the moving piece), `pigment.png` (last frame), `attention.png` (heat map of where the swarm lingered) and `attention.npy` (the raw attention grid).

## How it works

1. **Noticing.** The image is turned into a salience map: strong edges and saturated colour. This is a crude kind of attention, with no feeling behind it.
2. **Lingering.** A swarm moves over the map using a simplified version of Dispersive Flies Optimisation (DFO). Flies stay longer where the image stands out, then lose interest (an inhibition map) and move on. Where they linger is what the machine has "noticed". A small panel shows this as a heat map.
3. **Pigment.** Each fly releases particles that carry the colour it is looking at. Near a line, particles flow along the direction of that line; in empty space, a gentle wind carries them. They fade over time, so each artwork leaves its own moving trace of the swarm's gaze. (In the Python version; the browser version still uses only the wind.)

## Controls

| Control | Effect |
| --- | --- |
| Flies | Size of the swarm |
| Restlessness | Disturbance threshold: chance that a fly jumps to a random place |
| Flow speed | How fast the pigment drifts |
| Trail length | How slowly the pigment fades |

## Next steps

- Replace the hand-made salience map with the attention of a trained vision model.
- Run the swarm over a collection of artworks and compare where it looks with where people report an emotional response.
- Rebuild the pipeline in Python for analysis, and render longer video pieces.

## References

- al-Rifaie, M. M. (2014). *Dispersive Flies Optimisation*. Proceedings of FedCSIS 2014, pp. 529–538.
- al-Rifaie, M. M. & Bishop, J. M. (2013). *Swarmic Paintings and Colour Attention*. EvoMUSART 2013.
- Visual language inspired by the data-pigment works of Refik Anadol.

## About

Prototype by İlayda Yılmaz, MSc Data Science, University of Greenwich, October 2026.
Built with the help of an AI coding assistant.

The default image (`assets/drawing.jpg`) is my own ink drawing. Images you upload in the browser stay in your browser and are never sent anywhere.

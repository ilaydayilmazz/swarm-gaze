# Through Time

> What does a machine notice when it looks at how people have painted themselves, from the first cave drawings to the 20th century?

A machine looks at real artworks across history, reads how the human figure changes, and makes its own piece from them.

## How it works

1. **Collect**: downloads openly licensed artworks from Wikimedia Commons, age by age: prehistoric, Egypt, Greece, Roman Egypt, Byzantine, Gothic, Renaissance, Baroque, Impressionism, Van Gogh, Modigliani. Sources and licences go into `credits.csv`.
2. **Look**: a vision model (CLIP) turns each work into a vector.
3. **Read**: how strongly each work matches five ways of showing a person (symbolic, lifelike, emotional, fragmented, dissolving into light), plotted over time.
4. **Atlas**: all works on one map (UMAP); near means "looks alike to the machine".
5. **Journey**: the machine's own piece. A swarm based on Dispersive Flies Optimisation looks at each work in time order, lingers where something stands out, and releases light that flows along the work's own lines. Each age fades into the next.

## Run it

Open in Google Colab and choose **Runtime → Run all**:
https://colab.research.google.com/github/ilaydayilmazz/swarm-gaze/blob/main/through_time_colab.ipynb

## Files

| File | What it is |
| --- | --- |
| `through_time_colab.ipynb` | The notebook to run |
| `through_time.py` | Collect, look, read, atlas, journey |
| `swarm_gaze.py` | The swarm and the light (used by the journey; can also run on one image) |
| `assets/drawing.jpg` | My own ink drawing, the default image for `swarm_gaze.py` |
| `experiments/` | Earlier studies, kept for the record |

## References

- al-Rifaie, M. M. (2014). *Dispersive Flies Optimisation*. FedCSIS 2014.
- al-Rifaie, M. M. & Bishop, J. M. (2013). *Swarmic Paintings and Colour Attention*. EvoMUSART 2013.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Visual language inspired by the data-pigment works of Refik Anadol.

## About

By İlayda Yılmaz, MSc Data Science, University of Greenwich, 2026. Built with the help of an AI coding assistant.

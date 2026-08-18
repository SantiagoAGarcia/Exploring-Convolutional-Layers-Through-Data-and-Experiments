# Exploring Convolutional Layers Through Data and Experiments

Clothing image classification using Fashion-MNIST, comparing a dense baseline against a
convolutional network. The goal is not to maximize accuracy but to understand what a
convolutional layer actually contributes — and whether it wins because it has more capacity
or because its structure fits image data better.

## Repository structure

```
.
├── README.md
├── cnn_workshop.ipynb        # EDA, baseline, CNN, experiment, interpretation
├── requirements.txt
└── artifacts/
    └── results.json          # full metrics (generated when the notebook runs)
```

To run:

```bash
pip install numpy matplotlib tensorflow
jupyter notebook cnn_workshop.ipynb
```

There is a `FAST_MODE` flag in the first cell. Set it to `True` to train on a small subset
and verify everything runs; the results reported here used `False`.

---

## 1. Dataset

**Fashion-MNIST**, loaded from `tf.keras.datasets`. Ten clothing categories, same 28×28
grayscale format as MNIST but harder, because several classes share a similar silhouette
and differ only in texture or fine detail.

| | |
|---|---|
| Total images | 60,000 training · 10,000 test |
| Split used | 54,000 train / 6,000 validation / 10,000 test |
| Image size | 28 × 28, single channel |
| Classes | 10, balanced at 6,000 images each |
| Pixel values | integers 0–255 |

**Why not plain MNIST.** A simple dense network already exceeds 97% on MNIST, leaving
almost no room for an architectural improvement to show up. Fashion-MNIST has the same
format and is harder precisely where convolution should help.

**Why it suits convolutions.** Convolutional layers assume that what matters is local and
that a useful pattern in one region is also useful elsewhere. Both hold for clothing: what
separates a boot from a sandal is the shaft or the gaps between straps, not the
relationship between two distant pixels. Section 2.4 of the notebook confirms this
empirically — pixel correlation drops sharply with distance and is near zero beyond 10 px,
so restricting each filter to a small window does not discard meaningful information.

**Preprocessing:** divide by 255, add the channel dimension, reserve 10% for validation.
No resizing (all images are already the same size), no data augmentation. Keeping
preprocessing identical across models ensures that any performance difference comes from
architecture alone. The test set is used exactly once per model, at the end.

---

## 2. Baseline model (no convolutions)

```
Input (28,28,1) → Flatten(784) → Dense(128, relu) → Dense(64, relu) → Dense(10, linear)
```

**109,386 parameters.**

The key operation is `Flatten`: once the image is flattened, the top-left pixel and the
bottom-right pixel become two arbitrary positions in a vector. The network has no way to
know which pixels were neighbors. Every weight in the first dense layer connects to a
single fixed pixel position — if the garment shifts one pixel, a completely different set
of weights activates.

I deliberately sized this model to be close to the CNN in parameter count. If the CNN won
with twice as many parameters, the comparison would be ambiguous.

| Metric | Value |
|---|---:|
| Train accuracy | 0.9234 |
| Validation accuracy | 0.8890 |
| Test accuracy | 0.8769 |
| Training time | ~13 s |

---

## 3. CNN architecture

```
Input (28,28,1)
  Conv2D(16, 3×3, same, relu)  →  (28,28,16)      160 params
  MaxPooling2D(2×2)            →  (14,14,16)
  Conv2D(32, 3×3, same, relu)  →  (14,14,32)    4,640 params
  MaxPooling2D(2×2)            →  (7,7,32)
  Flatten                      →  1,568
  Dense(64, relu)                             100,416 params
  Dense(10, linear)                               650 params
```

**105,866 parameters** — 3,520 fewer than the baseline.

Something worth noting: the two convolutional layers together hold only 4,800 parameters,
less than 5% of the total. Almost the entire model lives in the final dense layer. Those
4,800 parameters are what make the difference.

### Justification for each decision

| Decision | Choice | Reasoning |
|---|---|---|
| Number of conv layers | 2 | With 2 layers and their pooling, each output neuron sees a 10×10 region of the original 28×28 image — enough to cover a meaningful part of a garment. One layer sees too little; four layers shrink the feature map too aggressively. |
| Kernel size | 3×3 | Pixel correlation drops to near zero within a few pixels, so a small window captures the relevant neighborhood. Two stacked 3×3 layers cover the same receptive field as one 5×5 with fewer parameters. This is what the experiment tests. |
| Filters | 16 → 32 | Doubling filters when pooling halves the spatial map avoids a sudden information bottleneck. Starting at 16 is enough for the first layer, which only needs to detect edges. |
| Stride | 1 in conv, 2 in pooling | Stride 1 in convolution means nothing is skipped; spatial reduction is delegated to pooling, which adds no parameters. |
| Padding | `same` | Without padding, each convolution crops the border, potentially discarding clothing pixels. `same` also keeps the output size independent of kernel size, which is essential for the experiment — the dense layer receives the same 1,568-dimensional vector regardless of kernel. |
| Activation | ReLU | The background is mostly black, so many activations will be zero anyway. ReLU does not saturate for positive values and is cheap to compute. |
| Output layer | linear | Used with `from_logits=True` for numerical stability. Softmax is applied after the model when a probability is needed. |
| Pooling | MaxPooling 2×2 | Reduces computation, enlarges each neuron's receptive field without adding parameters, and provides a few pixels of shift tolerance. Max rather than average because the question is whether a pattern appears at all — averaging dilutes it against the black background. |

| Metric | Value |
|---|---:|
| Train accuracy | 0.9452 |
| Validation accuracy | 0.8993 |
| Test accuracy | **0.9006** |
| Training time | ~53 s |

---

## 4. Experiment: kernel size

One variable changed, everything else fixed: 3×3 vs 5×5 vs 7×7 kernels, with 2 conv
layers, 16 and 32 filters, stride 1, `same` padding, MaxPooling, ReLU, the same dense
layer, Adam lr=1e-3, batch 128, 15 epochs, same data split.

`same` padding is what makes this a clean single-variable experiment: the feature map
entering the dense layer is always 7×7×32 = 1,568, so that layer has identical parameters
across all three configurations. The only thing that changes is the conv parameter count.

Each configuration was run 3 times with different random seeds. On the first pass with a
single run, 5×5 appeared to win; re-running gave a different ranking. With random
initialization, a single run cannot distinguish a real difference from luck.

### Parameter cost per kernel

| Kernel | Conv params | Total params | Receptive field |
|---|---:|---:|---:|
| 3×3 | 4,800 | 105,866 | 10×10 |
| 5×5 | 13,248 | 114,314 | 16×16 |
| 7×7 | 25,920 | 126,986 | 22×22 |

### Results (15 epochs, batch 128, Adam lr=1e-3)

| Model | Params | Train acc | Val acc | Test acc | Time (s) |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 109,386 | 0.9234 | 0.8890 | 0.8769 | 13 |
| CNN | 105,866 | 0.9452 | 0.8993 | **0.9006** | 53 |
| Compact CNN (control) | 23,946 | 0.8609 | 0.8377 | 0.8266 | 66 |

| Kernel | Val acc (avg 3 seeds) | Run-to-run variation | Train–val gap | Time (s) |
|---|---:|---:|---:|---:|
| 3×3 | 0.9057 | 0.0035 | +0.0240 | 38 |
| 5×5 | 0.9045 | 0.0052 | +0.0290 | 50 |
| 7×7 | 0.9033 | 0.0061 | +0.0308 | 78 |

### Analysis

**In accuracy, the three kernels are statistically tied.** The gap between best and worst
is 0.0024, while runs of the same kernel vary by up to 0.0061 due to random initialization
alone. The signal is smaller than the noise.

**The difference is entirely in cost.** Going from 3×3 to 7×7 multiplies convolution
parameters by 5.4 and doubles training time, for the same result. The train–val gap also
grows from +0.024 to +0.031, meaning larger kernels overfit slightly more.

**The hypothesis was confirmed, but not in the way I expected.** I expected 3×3 to be
enough. It is not that 3×3 wins — it is that enlarging the kernel adds nothing while
costing five times more. When two options produce the same result, the cheaper one is
strictly better.

This is consistent with the pixel correlation measurement: correlation at 5 pixels was
still 0.41, so it was reasonable to expect 5×5 not to be worse. But that residual
correlation is not enough to make it better.

| Kernel | Pros | Cons |
|---|---|---|
| 3×3 | Same result, 5× fewer conv params, half the time | Sees a small region per layer; needs stacking to cover more |
| 5×5 | Larger immediate receptive field | 2.8× more parameters for the same accuracy |
| 7×7 | Very wide immediate context | 5.4× more parameters, 2× slower, slightly more overfitting, same result |

**Conclusion:** 3×3 is the right choice — not because it achieves higher accuracy, but
because it achieves the same accuracy far more cheaply. If a larger receptive field were
needed, the right approach is to stack more small layers, not to enlarge the kernel.

---

## 5. Interpretation and architectural reasoning

### Why did the CNN outperform the baseline?

The CNN reached 0.9006 vs 0.8769 for the dense baseline: 2.4 points higher, with 3,520
fewer parameters. Three arguments were tested; two hold and one does not:

1. **With the same parameter count, the CNN wins.** If only capacity mattered, they should
   have tied.
2. **The compact CNN did not reach the baseline.** It ended still improving at epoch 15,
   so it was undertrained — not a valid argument either way.
3. **The CNN tolerates small shifts much better**, with the caveat below.

**Where exactly it wins.** The gains are concentrated in pullover (+9.3 points), T-shirt
(+7.3), and shirt (+2.9) — the torso garments that share a silhouette and differ by
texture. On easy classes (trousers, ankle boot, sneaker) the improvement is under half a
point because there was no room, and on coat the CNN is actually two points worse.

The CNN does not improve uniformly. It improves precisely where fine local detail matters,
which was the prediction made by looking at the images before training.

**The underlying mechanism:** a 3×3 filter is applied at all 784 positions of the image,
so its 9 weights receive gradient signal from every position in every image. A weight in
the dense layer sees only one fixed pixel per image.

**Honest caveat:** 2.4 points is not dramatic, for two reasons. The images are small and
centered, so the variable-position problem barely exists. And 95% of the CNN's parameters
are in the final dense layer — the two models share almost the same structure except at
the very beginning.

### An extra test: shifting the images

Test images were shifted by 1–4 pixels (zero-padded) and both models were evaluated
without retraining.

| Shift | Baseline | CNN | CNN advantage |
|---|---:|---:|---:|
| 0 px | 0.8769 | 0.9006 | +0.024 |
| 1 px | 0.7452 | 0.8357 | +0.091 |
| 2 px | 0.4273 | 0.6623 | **+0.235** |
| 3 px | 0.2265 | 0.3030 | +0.077 |
| 4 px | 0.1458 | 0.1733 | +0.028 |

At 2 pixels the dense model collapses to 0.43 while the CNN holds at 0.66 — a 23-point
gap. But at 3–4 pixels both collapse below 20%, barely above random chance for 10 classes.

The tolerance comes from the two MaxPooling 2×2 layers, which provide a margin of a few
pixels. Beyond that there is nothing to sustain it, because the CNN ends in a Flatten
followed by a dense layer, and that layer does depend on where each activation appeared.

This clarified a common misconception: convolution does not provide translation invariance
for free. It makes the same filter detect the same pattern anywhere in the image — that is
not the same thing. If you flatten afterward and connect to a dense layer, position matters
again. The inductive biases come from the entire architecture, not from a single layer type.

### What inductive biases does convolution introduce?

Choosing an architecture encodes assumptions about the data before the model sees a single
example. Convolution encodes four:

| Assumption | How it is built in | What the experiments showed |
|---|---|---|
| What matters is local | Each neuron connects only to a k×k window | Confirmed: correlation drops to zero by 10 px |
| What works here works elsewhere | The same filter slides over the entire image | Confirmed: wins with fewer parameters |
| Shifting the image does not change what it is | Follows from weight sharing | Partially: holds for 1–2 px, not 4 |
| Complex patterns are built from simple ones | Stacking layers increases receptive field | Confirmed: learned filters are edge detectors |

The most striking observation is that all four are **restrictions**: a convolutional layer
can represent fewer functions than a dense layer of the same size, not more. And yet it
performs better.

The reason: the dense network must search for a solution among an enormous number of
options, most of which make no sense for images (e.g., treating two distant pixels as
related). The CNN has those options ruled out from the start, so it searches in a much
smaller space where almost everything is reasonable. That is why it needs less data to
find a good solution.

This is analogous to hand-crafting features in classical ML — except that instead of
deciding which features to use, we decide the rules by which the network learns them on
its own.

### When is convolution not appropriate?

The assumptions above break down in several common situations:

- **Tabular data.** Column order is arbitrary: there is no reason for age and income to be
  "neighbors," and passing the same filter over columns that measure different things is
  meaningless. A dense network or tree-based model is the right tool.
- **When exact position matters.** If the task is "is there something bright in the upper
  left?", position-invariant responses are exactly what you do not want. This arises in
  medical imaging where anatomical location is part of the diagnosis.
- **When relevant information is far apart.** In text, subject and verb can be separated
  by many words; a CNN would need an impractical number of layers to connect them.
  Attention mechanisms exist for this reason.
- **When the data does not form a grid.** Graphs, social networks, point clouds: there is
  no well-defined "right neighbor."

The more useful question — more useful than "if it's images, use a CNN" — is: **do the
data have neighbors that mean something, and does a pattern that works in one place also
work in another?** If both answers are yes, convolution helps. If either is no, those
assumptions become a constraint rather than an advantage.

---

## Bonus: learned filters

Section 7 of the notebook visualizes the first-layer weights and activation maps. The
filters are not noise: they show clear light-dark patterns in different orientations —
the signature of edge detectors — and nobody programmed them. In the `conv_1` maps the
garment silhouette is still recognizable; in `conv_2` the representations are smaller and
harder to interpret directly.

---

## What remains open

- **Train the compact CNN to convergence.** It was still improving at epoch 15, so the
  control experiment is unresolved. This is the most important open question.
- **Test the compact CNN's shift robustness.** Because it uses GlobalAveragePooling2D
  instead of Flatten, it should tolerate large shifts much better. That is a concrete
  prediction from the analysis that was not tested.
- **Larger images.** It is unclear whether the kernel-size result holds for larger inputs;
  larger kernels are reportedly used in early layers of models trained on high-resolution
  images, but this was not tested here.
- **Regularization on the baseline.** Data augmentation, dropout, and batch normalization
  were deliberately excluded to isolate architectural effects. It remains an open question
  how much the dense baseline would improve with regularization, since it overfits the most.
- **What I would try next:** train the dense model on randomly shifted images, to see
  whether it can learn shift tolerance on its own or whether the architecture must impose it.

# Draft: Paper Introduction
# "Recovering Metric Scale in Pretrained Image-to-3D Generators Without Backbone Retraining"
# Created: 2026-05-19 — first draft, for review/iteration

---

## Introduction

Image-to-3D generative models have reached a level of visual fidelity where a single photograph is
sufficient to produce a detailed mesh or Gaussian splat of an object with plausible geometry and texture.
Foundations such as SAM 3D Objects~\cite{sam3d} and TRELLIS~\cite{trellis} train on millions of synthetic
and real objects, learning rich priors over 3D shape and surface appearance.
Yet these models share a fundamental blind spot: their outputs live in a canonical unit cube,
$[-0.5, 0.5]^3$, with no connection to physical units.
A reconstructed bowl and a reconstructed car are indistinguishable in scale.

This missing dimension — metric scale — is not a cosmetic limitation.
Robotic manipulation requires knowing whether a grasped object is five centimetres or fifty.
Augmented reality demands that a reconstructed mug placed on a physical table displaces
a physically plausible volume.
Scene reconstruction from multiple objects depends on consistent metric embedding.
Without absolute scale, a 3D generative model is an artist's tool, not a spatial computing primitive.

\paragraph{Why metric scale is hard.}
Recovering metric scale from a single image is fundamentally under-constrained without a prior
on object size. The standard workarounds each carry a cost.
\emph{Category-mean priors} assume the object is an average specimen of its class — a heuristic
that incurs $\sim$10.9\% mean absolute percentage error (MAPE) on NOCS-Real275~\cite{nocs}
even when the category is known.
\emph{Separate monocular metric depth models}~\cite{metric3d,zoedepth} can estimate depth
in physical units but require an additional inference pass, introduce a second set of parameters,
and still require a geometric bridge between pixel-depth and object volume.
\emph{Backbone retraining} — modifying the 3D generator itself to reason about physical scale —
demands compute budgets (>100K GPU-hours for models of this class) that are inaccessible to
most researchers and would destroy the generalisation properties built into the pretrained weights.

\paragraph{The hidden anchor.}
We observe that modern image-to-3D pipelines based on monocular depth estimation already contain
the metric signal needed for scale recovery — it is simply being discarded.
SAM 3D Objects uses MoGe~\cite{moge} to lift the input image into a metric pointmap:
a dense field of $(x, y, z)$ coordinates in metres aligned to the camera frame.
From this pointmap the pipeline extracts a global scale statistic (\texttt{pointmap\_scale})
and a vertical offset (\texttt{pointmap\_shift\_z}) to normalise input conditioning —
and then never uses the original metric values again.
The object's true physical extent, encoded in those pointmap statistics and in the spatial
distribution of pointmap tokens, is present at inference time but plays no role in the output.

\paragraph{Our approach.}
We propose a lightweight recipe for injecting metric awareness into a frozen image-to-3D
generator without retraining any backbone component.
A compact \emph{MetricScaleHead} fuses MoGe pointmap statistics with pooled sparse-structure (SS)
latent features into a 1024-dimensional scale token.
This token is injected into the spatial latent (SLAT) flow-matching decoder via cross-attention,
the only part of the generative model that is fine-tuned (approximately 100M of more than one
billion total parameters, with an fp32 upcast recipe that prevents the bf16 attention overflow
we encountered in early experiments).
Separately, a differentiable \emph{aspect-ratio loss} on the SS decoder's soft voxel occupancy
supervises object shape proportions directly, closing the gap between the metric head's
per-axis scale predictions and the mesh's actual bounding-box dimensions.

\paragraph{Results.}
Trained on the real-world NOCS-Real275 subset of OmniNOCS~\cite{omninocs} and evaluated on a
held-out scene split, our method achieves \textbf{1.23\% MAPE} (0.20 cm mean absolute error)
— an order of magnitude below the category-prior baseline and competitive with specialised
monocular metric depth methods that require a separate inference pass.
When the same recipe is extended to mixed-source training on Objectron~\cite{objectron} and
ARKitScenes~\cite{arkitscenes}, the resulting model predicts metric scale across object categories,
viewpoints, and indoor environments without sacrificing per-category accuracy on the original
evaluation set.

\paragraph{Contributions.}
We contribute: (i) the observation that metric pointmap statistics are a latent anchor already
present in image-to-3D pipelines and amenable to lightweight supervision;
(ii) a practical recipe — MetricScaleHead, scale-token cross-attention injection, and soft
aspect-ratio supervision — that adds metric scale to a frozen 3D generator with $\sim$170M
trainable parameters and no backbone modification;
(iii) a training stability protocol (fp32 cross-attention upcast, linear lr warmup)
that makes fine-tuning large sparse flow-matching decoders viable without NaN instabilities;
and (iv) empirical validation across three real-world object datasets demonstrating sub-2\%
MAPE with generalisation to unseen object categories.

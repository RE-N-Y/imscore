# Imscore
**Work by RE-N-Y and friends @ [krea.ai](https://krea.ai)**

![teaser](teaser.png)

Imscore is a minimal library curating a set of **fully differentiable** aesthetic and preference scorers for images.
We provide a set of popular scorers such as PickScore, MPS, HPSv2, and LAION aesthetic scorer as well as our own models trained on open source preference datasets.

`imscore` allows ...

1. **Benchmarking** your generative models on aesthetic and preference scorers.
2. **Post training** your generative models to align with human preference.
3. **Prevent headaches** of porting over models from different repositories.


## Installation

```bash
pip install imscore
pip install transformers<5
```

## Usage

```python
from imscore.aesthetic.model import ShadowAesthetic, LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse
from imscore.hpsv3.model import HPSv3

import torch
import numpy as np
from PIL import Image
from einops import rearrange

# popular aesthetic/preference scorers
model = ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2") # ShadowAesthetic aesthetic scorer (my favorite)
model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14") # CLIPScore
model = PickScorer.from_pretrained("RE-N-Y/pickscore") # PickScore preference scorer
model = MPS.from_pretrained("RE-N-Y/mpsv1") # MPS (ovreall) preference scorer
model = HPSv2.from_pretrained("RE-N-Y/hpsv21") # HPSv2.1 preference scorer
model = ImageReward.from_pretrained("RE-N-Y/ImageReward") # ImageReward aesthetic scorer
model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic") # LAION aesthetic scorer
model = CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-Combo') # CycleReward preference scorer.
model = VQAScore.from_pretrained("RE-N-Y/clip-t5-xxl")
model = EvalMuse.from_pretrained("RE-N-Y/evalmuse")
mode = HPSv3.from_pretrained("RE-N-Y/hpsv3") # NOTE: use with caution, expect 2~3% parity with original implementation. Requires a fix.

# multimodal (pixels + text) preference scorers trained on PickaPicv2 dataset 
model = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip")

prompt = "a photo of a cat"
pixels = Image.open("cat.jpg")
pixels = np.array(pixels)
pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0

# prompts and pixels should have the same batch dimension
# pixels should be in the range [0, 1]
# score == logits
score = model.score(pixels, [prompt]) # full differentiable reward
```

## Post Training for Generative Models

```python
import torch
from imscore.preference.model import SiglipPreferenceScorer

G = model() # your generative model
dataloader = ... # your dataloader with conditioning (ex.prompts)

rm = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip") # pretrained preference model
optim = torch.optim.AdamW(G.parameters(), lr=3e-4)

# post training
for prompts in dataloader:
    optim.zero_grad()
    
    images = G(prompts)
    scores = rm.score(images, prompts) # ensure images are in the range [0, 1]
    loss = -scores.mean() # maximise reward

    loss.backward()
    optim.step()
```

## List of available models

### Aesthetic Scorers
```python
from imscore.aesthetic.model import ShadowAesthetic, CLIPAestheticScorer, SiglipAestheticScorer, Dinov2AestheticScorer, LAIONAestheticScorer

# pixel only scorers trained on imscore dataset's aesthetic rating
SiglipAestheticScorer.from_pretrained("RE-N-Y/imreward-fidelity_rating-siglip")
CLIPAestheticScorer.from_pretrained("RE-N-Y/imreward-fidelity_rating-clip")
Dinov2AestheticScorer.from_pretrained("RE-N-Y/imreward-fidelity_rating-dinov2")

# pixel only scorers trained on imreward dataset's overall rating
SiglipAestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-siglip")
CLIPAestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-clip")
Dinov2AestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-dinov2")

# pixel only scorers trained on AVA dataset
CLIPAestheticScorer.from_pretrained("RE-N-Y/ava-rating-clip-sampled-True")
CLIPAestheticScorer.from_pretrained("RE-N-Y/ava-rating-clip-sampled-False")
SiglipAestheticScorer.from_pretrained("RE-N-Y/ava-rating-siglip-sampled-True")
SiglipAestheticScorer.from_pretrained("RE-N-Y/ava-rating-siglip-sampled-False")
Dinov2AestheticScorer.from_pretrained("RE-N-Y/ava-rating-dinov2-sampled-True")
Dinov2AestheticScorer.from_pretrained("RE-N-Y/ava-rating-dinov2-sampled-False")

# Common aesthetic scorers
LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic") # LAION aesthetic scorer
ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2") # ShadowAesthetic aesthetic scorer for anime images
```

### Preference Scorers
```python
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.imreward.model import ImageReward
from imscore.preference.model import SiglipPreferenceScorer, CLIPPreferenceScorer
from imscore.pickscore.model import PickScorer
from imscore.cyclereward.model import CycleReward

HPSv2.from_pretrained("RE-N-Y/hpsv21") # HPSv2.1 preference scorer
MPS.from_pretrained("RE-N-Y/mpsv1") # MPS (ovreall) preference scorer
PickScorer("yuvalkirstain/PickScore_v1") # PickScore preference scorer
ImageReward.from_pretrained("RE-N-Y/ImageReward") # ImageReward preference scorer
CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-Combo') # CycleReward preference scorer trained on combined CyclePrefDB
CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-T2I')# CycleReward preference scorer trained on CyclePrefDB-T2I only
CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-I2T')
# CycleReward preference scorer trained on CyclePrefDB-I2T only

# multimodal scorers trained on PickAPicv2 dataset
SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip")
CLIPPreferenceScorer.from_pretrained("RE-N-Y/pickscore-clip")
```

## Benchmark on Preference datasets

| Model                           |  HPD v2 Accuracy      | ImageReward Accuracy       |   Pickapicv2 Accuracy        |       Average     |
|:--------------------------------|----------------------:|---------------------------:|-----------------------------:|------------------:|
| pickscore                       |              0.792157 |                   0.667448 |                     0.801887 |          0.753831 |
| hpsv21                          |              0.833464 |                   0.674793 |                     0.693396 |          0.733884 |
| mpsv1                           |              0.838562 |                   0.677762 |                     0.650943 |          0.722423 |
| imreward                        |              0.740131 |                   0.657915 |                     0.608491 |          0.668846 |
| laion-aesthetic                 |              0.736013 |                   0.566807 |                     0.551887 |          0.618236 |
| clipscore                       |              0.626078 |                   0.571652 |                     0.606132 |          0.601287 |

For full benchmark results and methodology, please refer to [bench.md](bench.md).

## Differenes between original and ported versions

| Model | Mean Error | Mean Error % |
|-------|------------|--------------|
| pickscore | 0.0364375 | 0.1697% |
| mps | 0.1221515 | 1.3487% |
| hps | 0.0010474 | 0.3733% |
| laion | 0.0202606 | 0.3461% |
| imreward | 0.0135808 | 0.7608% |
| clip-t5-xxl | 0.0111 | 2.0091% |
| evalmuse | xxx | xxx |
| hpsv3 | xxx | xxx |

`imscore` library ports popular scorers such as PickScore, MPS, HPSv2, etc. In order to ensure that `.score` function is (1) fully differentiable and (2) takes pixels of range [0, 1], the image processing pipeline had to be modified. The above table reports the mean and standard error between the original and ported versions. 

Most ported models have a mean absolute error less than < 1% w.r.t original output. These statistics were computed on PickAPicv2 test unique set images.

## Why did I make this?

1. To save myself headaches.
2. To provide a common interface for dataset filtering, posttraining, and image model benchmarking.

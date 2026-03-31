import math
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch
import torch.nn.functional as F

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {prompt}


""" + """
Please provide the overall ratings of this image: <|Reward|>

END
"""


def resize(height: int, width: int, factor: int, minpix:int, maxpix:int) -> tuple[int, int]:
    hbar = max(factor, round(height / factor) * factor)
    wbar = max(factor, round(width / factor) * factor)
    if hbar * wbar > maxpix:
        beta = math.sqrt((height * width) / maxpix)
        hbar = math.floor((height / beta) / factor) * factor
        wbar = math.floor((width / beta) / factor) * factor
    elif hbar * wbar < minpix:
        beta = math.sqrt(minpix / (height * width))
        hbar = math.ceil((height * beta) / factor) * factor
        wbar = math.ceil((width * beta) / factor) * factor
    return hbar, wbar


def fetch(ele: dict, factor:int = 28) -> Image.Image:
    image = ele["image"]
    w, h = image.size
    newh, neww = resize( h, w, factor=factor, minpix=ele["minpix"], maxpix=ele["maxpix"])
    image = TF.resize(image, (newh, neww), interpolation=InterpolationMode.BICUBIC)
    return image

def resize_tensor(image: torch.Tensor, factor: int, minpix: int, maxpix: int) -> torch.Tensor:
    """
    image: Tensor [C, H, W] in [0,1] (float)
    factor: int, round to multiple of this
    minpix, maxpix: pixel constraints (H*W)
    """
    _, h, w = image.shape

    # Convert to float to allow gradient flow
    h = torch.tensor(h, dtype=torch.float32, device=image.device)
    w = torch.tensor(w, dtype=torch.float32, device=image.device)

    # Round to multiple of factor
    hbar = torch.clamp((h / factor).round() * factor, min=factor)
    wbar = torch.clamp((w / factor).round() * factor, min=factor)

    # Compute area
    area = hbar * wbar

    # If too large -> scale down
    if area > maxpix:
        beta = torch.sqrt((h * w) / maxpix)
        hbar = torch.floor((h / beta) / factor) * factor
        wbar = torch.floor((w / beta) / factor) * factor
    # If too small -> scale up
    elif area < minpix:
        beta = torch.sqrt(minpix / (h * w))
        hbar = torch.ceil((h * beta) / factor) * factor
        wbar = torch.ceil((w * beta) / factor) * factor

    # Ensure integers
    newh, neww = int(hbar.item()), int(wbar.item())

    # Resize with differentiable interpolation
    image = image.unsqueeze(0)  # [1, C, H, W]
    image = F.interpolate(image, size=(newh, neww), mode="bicubic", align_corners=False)
    return image.squeeze(0)  # [C, newh, neww]


def process(conversations: list[dict] | list[list[dict]]) -> list:
    infos = []
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "image" in ele:
                        infos.append(fetch(ele))
    return infos

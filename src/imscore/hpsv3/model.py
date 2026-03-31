
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import Tensor
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from .utils import process, INSTRUCTION, resize_tensor


class HPSv3(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.rm_head = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
        self.processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side="right")
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
        self.special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(["<|Reward|>"])
        self.reward_token = "special"


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None= None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(input_ids=None, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states)

        hiddens = outputs[0]  # [B, L, D]
        with torch.autocast('cuda', dtype=torch.float32):
            logits = self.rm_head(hiddens)  # [B, L, N]

        b, *_ = inputs_embeds.shape
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == special_token_id)
        pooled = logits[special_token_mask, ...]
        pooled = pooled.view(b, 1, -1)  # [B, 3, N] assert 3 attributes
        pooled = pooled.view(b, -1)

        return { "logits": pooled }
    
    def prepare(self, images, prompts):
        messages = []
        # tform = T.ToPILImage()
        # images = [tform(im) for im in images]
        images = [im for im in images]
        for text, image in zip(prompts, images):
            m = [{ 
                "role": "user",
                "content": [ 
                    { "type": "image", "image": image },
                    { "type": "text",  "text": INSTRUCTION.format(prompt=text) }
                ]
            }]

            messages.append(m)

        # images = process(messages)
        images = [resize_tensor(img, minpix=256*28*28, maxpix=256*28*28, factor=28) for img in images]
        batch = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=images,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        batch = { k : v.to(self.device) for k,v in batch.items() }
        return batch
    
    def score(self, images:list[Tensor], prompts:list[str]):
        batch = self.prepare(images, prompts)
        rewards = self.forward(**batch)["logits"]
        mean, std = rewards.chunk(chunks=2, dim=-1)
        return mean

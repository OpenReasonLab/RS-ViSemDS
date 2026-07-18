from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LOCAL_MLLM_IMPLEMENTATION_VERSION = "2026-07-12-internvl-qwen3-context-v2"


class TransformersVisionLLM:
    """Local Hugging Face Transformers backend for vision-language generation.

    Supports Llama-3.2-Vision, Qwen2/2.5/3-VL, Gemma-3 vision models,
    InternVL chat models, and other AutoModelForImageTextToText-compatible VLMs.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = dtype_map[torch_dtype]
        self.model_family = self._model_family(model_id)
        self.processor = None
        self.tokenizer = None

        if self.model_family == "internvl":
            self._init_internvl(model_id, device_map)
            return

        processor_kwargs: dict[str, Any] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            **processor_kwargs,
        )
        model_cls = self._resolve_model_class(self.model_family, AutoModelForImageTextToText)
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def _init_internvl(self, model_id: str, device_map: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        try:
            import flash_attn  # noqa: F401

            self.use_flash_attn = True
        except (ImportError, OSError):
            self.use_flash_attn = False
        print(
            "InternVL attention backend: "
            + ("FlashAttention2" if self.use_flash_attn else "standard attention fallback")
        )
        # InternVLChatModel does not expose all_tied_weights_keys in some
        # Transformers versions, so accelerate's device_map="auto" path fails.
        # The 8B model fits on a single GPU in bf16 on typical AutoDL cards.
        internvl_device_map: str | dict[str, int] = device_map
        if device_map == "auto":
            internvl_device_map = {"": 0}

        original_module_getattr = torch.nn.Module.__getattr__

        def patched_module_getattr(module, name):
            if name == "all_tied_weights_keys":
                for cls in type(module).mro():
                    if "_tied_weights_keys" in cls.__dict__:
                        tied = cls.__dict__["_tied_weights_keys"]
                        return {key: key for key in tied} if tied else {}
                return {}
            return original_module_getattr(module, name)

        torch.nn.Module.__getattr__ = patched_module_getattr
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=internvl_device_map,
                use_flash_attn=self.use_flash_attn,
            ).eval()
        finally:
            torch.nn.Module.__getattr__ = original_module_getattr

    @staticmethod
    def _model_family(model_id: str) -> str:
        name = model_id.lower().replace("_", "-")
        if "internvl" in name:
            return "internvl"
        if "qwen3-vl" in name or "qwen3vl" in name:
            return "qwen3_vl"
        if "qwen2.5-vl" in name or "qwen2-5-vl" in name or "qwen25-vl" in name:
            return "qwen2_5_vl"
        if "qwen2-vl" in name:
            return "qwen2_vl"
        if "llama-3.2" in name or "mllama" in name:
            return "mllama"
        return "auto_image_text"

    @staticmethod
    def _resolve_model_class(model_family: str, auto_cls):
        if model_family == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration

                return Qwen3VLForConditionalGeneration
            except ImportError:
                return auto_cls
        if model_family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        if model_family == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        if model_family == "mllama":
            from transformers import MllamaForConditionalGeneration

            return MllamaForConditionalGeneration
        return auto_cls

    def _input_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def generate_from_messages(self, messages: list[dict], images: list[Image.Image]) -> str:
        if self.model_family == "internvl":
            return self._generate_internvl(messages, images)
        return self._generate_standard(messages, images)

    def _generate_standard(self, messages: list[dict], images: list[Image.Image]) -> str:
        import torch

        assert self.processor is not None
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=images if len(images) != 1 else images[0],
            text=prompt,
            return_tensors="pt",
        )
        device = self._input_device()
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_token_count = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][input_token_count:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _generate_internvl(self, messages: list[dict], images: list[Image.Image]) -> str:
        import torch

        assert self.tokenizer is not None
        question = self._internvl_question(messages, len(images))
        pixel_values, num_patches_list = self._internvl_pixel_values(images)
        device = self._input_device()
        pixel_values = pixel_values.to(device=device, dtype=self._torch_tensor_dtype())
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            ).strip()

    def _torch_tensor_dtype(self):
        import torch

        if self.torch_dtype == "auto":
            param = next(self.model.parameters())
            return param.dtype if param.dtype.is_floating_point else torch.bfloat16
        return self.torch_dtype

    @staticmethod
    def _internvl_question(messages: list[dict], image_count: int) -> str:
        text_parts: list[str] = []
        seen_images = 0
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "image":
                    seen_images += 1
                    if image_count == 1:
                        text_parts.append("<image>")
                    else:
                        text_parts.append(f"Image-{seen_images}: <image>")
                elif part_type == "text":
                    text_parts.append(str(part.get("text", "")))
        if seen_images == 0:
            prefix = "\n".join(
                "<image>" if image_count == 1 else f"Image-{index}: <image>"
                for index in range(1, image_count + 1)
            )
            text_parts.insert(0, prefix)
        return "\n".join(part for part in text_parts if part).strip()

    @staticmethod
    def _internvl_pixel_values(images: list[Image.Image]):
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        tensors = [transform(image.convert("RGB")) for image in images]
        num_patches_list = [1 for _ in tensors]
        return torch.stack(tensors), num_patches_list


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def should_use_contact_sheet(model_id: str, image_count: int, mode: str) -> bool:
    if image_count <= 1:
        return False
    if mode == "contact_sheet":
        return True
    if mode == "native_multi":
        return False
    model_name = model_id.lower()
    return "llama-3.2" in model_name or "mllama" in model_name


def make_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    tile_size: int = 336,
    columns: int = 4,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("At least one image is required")

    font = ImageFont.load_default()
    label_height = 28
    gap = 12
    rows = (len(images) + columns - 1) // columns
    width = columns * tile_size + (columns + 1) * gap
    height = rows * (tile_size + label_height) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (image, label) in enumerate(zip(images, labels)):
        row = index // columns
        column = index % columns
        x = gap + column * (tile_size + gap)
        y = gap + row * (tile_size + label_height + gap)
        draw.text((x, y), label[:80], fill="black", font=font)
        tile = image.copy()
        tile.thumbnail((tile_size, tile_size))
        tile_canvas = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
        offset = ((tile_size - tile.width) // 2, (tile_size - tile.height) // 2)
        tile_canvas.paste(tile, offset)
        sheet.paste(tile_canvas, (x, y + label_height))
        draw.rectangle(
            [x, y + label_height, x + tile_size - 1, y + label_height + tile_size - 1],
            outline=(180, 180, 180),
            width=1,
        )
    return sheet


def image_part() -> dict:
    return {"type": "image"}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}

import re
from io import BytesIO
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch
from omegaconf import DictConfig

from qwen_vl_utils import process_vision_info as qwen_process_vision_info
from verl.utils.dataset import RLHFDataset, SFTDataset

class VTGDataset(RLHFDataset):

    def _build_messages(self, example: dict) -> List[Dict[str, Any]]:
        """Support video arguments and pre-process vision info.
        """
        messages: list = example[self.prompt_key]
        images = example.pop(self.image_key, [])
        videos = example.pop(self.video_key, [])

        assert len(videos) == 1
        video = {
            "type": "video",
            "video": videos[0],
            "total_pixels": 3584 * (self.image_patch_size * 2) ** 2,
            "min_pixels": 4 * (self.image_patch_size * 2) ** 2,
            "max_frames": 768,
        }
        if example.get("video_start", None):
            video["video_start"] = example["video_start"]
        if example.get("video_end", None):
            video["video_end"] = example["video_end"]

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                    elif isinstance(image, dict) and "bytes" in image:
                        image["image"] = Image.open(BytesIO(image["bytes"]))
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append(video)
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"

        # --- 在此处直接处理 Vision Info 并缓存 ---
        cached_images, cached_videos = qwen_process_vision_info(
            messages, 
            image_patch_size=self.image_patch_size, 
            return_video_metadata=True
        )
        if cached_videos is not None:
            video_data, video_metadata = cached_videos[0]
            start_idx = video_metadata["frames_indices"][0]
            video_metadata["frames_indices"] = [idx - start_idx for idx in video_metadata["frames_indices"]]
            cached_videos = [(video_data, video_metadata)]
        
        if messages:
            messages[0]['_cached_vision_info'] = (cached_images, cached_videos)

        return messages

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.
        """
        
        # --- 读取缓存 ---
        if messages and isinstance(messages[0], dict) and '_cached_vision_info' in messages[0]:
            return messages[0]['_cached_vision_info']

        images, videos = qwen_process_vision_info(
            messages, 
            image_patch_size=self.image_patch_size, 
            return_video_metadata=True
        )

        if videos is not None:
            video_data, video_metadata = videos[0]
            start_idx = video_metadata["frames_indices"][0]
            video_metadata["frames_indices"] = [idx - start_idx for idx in video_metadata["frames_indices"]]
            videos = [(video_data, video_metadata)]

        return images, videos

class VTGSFTDataset(SFTDataset):
    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

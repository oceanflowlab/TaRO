import torch
import torch.multiprocessing as mp
# (新) 导入 DataLoader 和 Dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams # (vLLM) 导入 vLLM
from dataset_config import VTG

import json
import os
import re
import numpy as np
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import trange, tqdm # (修改) 导入 tqdm

import logging
logging.basicConfig(level=logging.ERROR)

PROMPT = """To accurately pinpoint the event "{query}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example:

<think>
I need to find the specific moment a person eats from a box.
From 0.0s to 5.2s, a person is holding a box of food and talking to the camera, but they are not eating yet.
From 5.3s to 8.1s, the person opens the box and looks inside. Still no eating action.
At 8.2s, the person picks up a piece of food from the box.
From 8.5s to 12.4s, the person puts the food in their mouth and chews while holding the box. This clearly matches "eating from a box."
From 12.5s to 15.0s, the person puts the box down and wipes their mouth. The action has ended.
Therefore, the relevant segment starts when the food approaches the mouth and ends when the chewing action concludes or the box is lowered.
</think>
<answer>
8.50s to 12.40s
</answer>"""

def check_device():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"
    
def extract_time(paragraph):
    numbers = re.findall(r'\d+(?:\.\d+)?', paragraph)
    numbers = [float(num) for num in numbers]

    if len(numbers) >= 2:
        last_two = numbers[-2:]
        if last_two[0] > last_two[1]:
            last_two[0], last_two[1] = last_two[1], last_two[0]
        
        return last_two
    else:
        return [None, None]

def calculate_iou(gt_segment, pred_segment):
    if pred_segment[0] is None or pred_segment[1] is None:
        return 0.0

    gt_start, gt_end = gt_segment
    pred_start, pred_end = pred_segment

    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter_len = max(0, inter_end - inter_start)

    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union_len = union_end - union_start

    if union_len == 0:
        return 0.0

    iou = inter_len / union_len
    return iou


class VideoDataset(Dataset):
    def __init__(self, data, model_path, video_prefix, prompt, total_video_tokens=4096, max_frames=1024, fps=2):
        self.data = data
        self.model_path = model_path
        self.video_prefix = video_prefix
        self.prompt_template = prompt 
        self.total_video_tokens = total_video_tokens
        self.max_frames = max_frames
        self.fps = fps
        
        self.processor = None

    def _get_processor(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        return self.processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        processor = self._get_processor()
        d = self.data[idx]

        try:
            video_path = os.path.join(self.video_prefix, d['video_file'])
            if not os.path.exists(video_path):
                print(f"Worker PID {os.getpid()}: Video file not found, skipping: {video_path}")
                return None, None
            user_prompt = self.prompt_template.format(query=d['query'])

            video_info = {
                "type": "video",
                "video": video_path,
                "total_pixels": self.total_video_tokens * (processor.image_processor.patch_size * 2) ** 2,
                "min_pixels": 4 * (processor.image_processor.patch_size * 2) ** 2,
                "max_frames": self.max_frames,
                "fps": self.fps,
            }

            if d.get("video_start", None) is not None:
                video_info["video_start"] = float(d["video_start"])

            if d.get("video_end", None) is not None:
                video_info["video_end"] = float(d["video_end"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        video_info,
                        {"type": "text", "text": user_prompt}
                    ],
                }
            ]
            raw_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos, video_kwargs = process_vision_info(
                messages, 
                image_patch_size=processor.image_processor.patch_size, 
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if videos is not None:
                video_data, video_metadata = videos[0]
                start_idx = video_metadata["frames_indices"][0]
                video_metadata["frames_indices"] = [idx - start_idx for idx in video_metadata["frames_indices"]]
                videos = [(video_data, video_metadata)]
            else:
                video_metadatas = None

            multi_modal_data = {"video": videos}
            prompt = {"prompt":raw_prompt, "multi_modal_data":multi_modal_data, 'mm_processor_kwargs': video_kwargs}

            return prompt, d

        except Exception as e:
            print(f"Worker PID {os.getpid()}: Error preparing inputs for {d.get('video_file')}: {e}")
            return None, None


def vllm_collate_fn(batch):
    batch_inputs = []
    batch_original_data = []
    
    for inp, d in batch:
        if inp is not None and d is not None:
            batch_inputs.append(inp)
            batch_original_data.append(d)
            
    return batch_inputs, batch_original_data


def aggregate_and_calculate(output_dir, world_size):
    print("Main process: Aggregating results...")
    
    all_results = []
    
    for rank in range(world_size):
        result_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {result_file}: {line}")
        except FileNotFoundError:
            print(f"Warning: Result file not found, skipping: {result_file}")

    if not all_results:
        print("Error: No results found to aggregate.")
        return

    print(f"Total aggregated results: {len(all_results)}")

    aggregated_file = os.path.join(output_dir, "all_results_aggregated.jsonl")
    with open(aggregated_file, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + '\n')
    print(f"Aggregated results saved to {aggregated_file}")
    
    ious = []
    recalls_03 = []
    recalls_05 = []
    recalls_07 = []

    for item in all_results:
        gt_segments = item['relevant_windows']
        pred_segment = item['pred_relevant_window']
        
        max_iou = 0
        for gt_segment in gt_segments:
            iou = calculate_iou(gt_segment, pred_segment)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
        
        recalls_03.append(1 if max_iou >= 0.3 else 0)
        recalls_05.append(1 if max_iou >= 0.5 else 0)
        recalls_07.append(1 if max_iou >= 0.7 else 0)

    metrics = {
        "mIoU": np.mean(ious),
        "Recall@0.3": np.mean(recalls_03),
        "Recall@0.5": np.mean(recalls_05),
        "Recall@0.7": np.mean(recalls_07),
        "total_samples": len(all_results),
    }

    print("\n--- Final Metrics ---")
    print(json.dumps(metrics, indent=2))
    print("---------------------\n")

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


def evaluate_worker(rank, world_size, all_data, model_path, video_prefix, output_dir, prompt, num_workers=4, batch_size=16, total_video_tokens=4096, max_frames=1024, fps=2):
    if check_device() == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    print(f"Rank {rank} (PID {os.getpid()}): Initializing vLLM...")

    patch_size = 16 if "Qwen3" in model_path else 14
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=16384,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=0.8,
            disable_mm_preprocessor_cache=True, 
            limit_mm_per_prompt={"image": 0, "video": 1}
        )
    except Exception as e:
        print(f"Rank {rank}: Error loading vLLM model: {e}")
        return
        
    print(f"Rank {rank}: vLLM Model loaded.")

    num_samples = len(all_data)
    samples_per_gpu = (num_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, num_samples)
    my_data = all_data[start_idx:end_idx]

    print(f"Rank {rank}: Processing {len(my_data)} samples (indices {start_idx} to {end_idx-1}).")

    dataset = VideoDataset(
        data=my_data,
        model_path=model_path,
        video_prefix=video_prefix,
        prompt=prompt,
        total_video_tokens=total_video_tokens,
        max_frames=max_frames,
        fps=fps,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=vllm_collate_fn,
        pin_memory=False
    )
    
    output_file = open(os.path.join(output_dir, f"results_rank_{rank}.jsonl"), "w")
    
    for batch_inputs, batch_original_data in tqdm(dataloader, desc=f"Rank {rank} Processing"):

        if not batch_inputs:
            continue
            
        sampling_params = SamplingParams(
            repetition_penalty=1.05, 
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            stop=None,
            stop_token_ids=[151645, 151643], 
            max_tokens=1024,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

        try:
            outputs = llm.generate(
                prompts=batch_inputs,
                sampling_params=sampling_params,
                use_tqdm=False
            )
        except Exception as e:
            print(f"Rank {rank}: Error during llm.generate: {e}")
            continue

        for j, output in enumerate(outputs):
            d = batch_original_data[j]
            output_text = output.outputs[0].text
            
            pred_start, pred_end = extract_time(output_text)
            
            result = {
                "original_data": d,
                "model_output": output_text,
                "predicted_start": pred_start,
                "pred_relevant_window": [pred_start, pred_end], 
                "relevant_windows": d["relevant_windows"]
            }
            print(json.dumps(result), file=output_file)
            output_file.flush()
    
    output_file.close()

def main(args):
    world_size = torch.accelerator.device_count()
    print(f"Found {world_size} devices.")

    os.makedirs(args.output_dir, exist_ok=True)

    data_file = VTG[args.dataset]["annotation"]
    video_dir_prefix = VTG[args.dataset]["video_folder"]
    try:
        with open(data_file) as f:
            all_data = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(all_data)} samples from {data_file}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    if world_size > 1:
        mp.spawn(
            evaluate_worker,
            args=(world_size, all_data, args.model_path, video_dir_prefix, args.output_dir, PROMPT, args.num_workers, args.batch_size, args.total_video_tokens, args.max_frames, args.fps),
            nprocs=world_size,
            join=True
        )
    else:
        evaluate_worker(0, world_size, all_data, args.model_path, video_dir_prefix, args.output_dir, PROMPT, args.num_workers, args.batch_size, args.total_video_tokens, args.max_frames, args.fps)

    print("\nAll worker processes finished.")
    aggregate_and_calculate(args.output_dir, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Charades", choices=list(VTG.keys()))
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--total_video_tokens", type=int, default=4096)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for data loading")
    parser.add_argument("--batch_size", type=int, default=8, help="vLLM batch size")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, os.path.split(args.model_path)[-1], args.dataset)
    main(args)
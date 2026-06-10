import numpy as np
import torch
import random

def shuffle_boundary_frames(video_tensor, meta_data, gt_start, gt_end, time_radius=1.0):
    fps = meta_data['fps']
    frame_indices = np.array(meta_data['frames_indices'])
    frame_indices = frame_indices - frame_indices[0]
    
    start_min_frame = (gt_start - time_radius) * fps
    start_max_frame = (gt_start + time_radius) * fps
    
    end_min_frame = (gt_end - time_radius) * fps
    end_max_frame = (gt_end + time_radius) * fps
    
    tensor_indices_near_start = np.where(
        (frame_indices >= start_min_frame) & 
        (frame_indices <= start_max_frame)
    )[0]
    
    tensor_indices_near_end = np.where(
        (frame_indices >= end_min_frame) & 
        (frame_indices <= end_max_frame)
    )[0]
    

    def apply_shuffle(indices, tensor):
        if len(indices) < 2:
            return
            
        indices_torch = torch.from_numpy(indices).long()
        
        perm = torch.randperm(len(indices))
        shuffled_indices = indices_torch[perm]
        
        tensor[indices_torch] = tensor[shuffled_indices]


    apply_shuffle(tensor_indices_near_start, video_tensor)
    apply_shuffle(tensor_indices_near_end, video_tensor)
    
    return video_tensor


def sample_reasoning(captions, query):
    if not captions:
        return None
    
    reasoning = f"<think>\nI need to find the specific moment {query}.\n"

    n = len(captions)
    k = random.randint(1, n)
    sampled_indices = sorted(random.sample(range(n), k))
    selected_captions = [captions[i] for i in sampled_indices]
    # selected_captions = captions
    reasoning = reasoning + "\n".join(selected_captions)
    
    return reasoning
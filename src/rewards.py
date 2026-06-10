import re


def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    # print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*)s (to|and) (\d+\.?\d*)s", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward_v2(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    **kwargs,
):
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    duration = extra_info.get("duration")
    split = extra_info.get("split", "train")
    reward = 0.0
    iou = 0.0
    parsed_times = parse_timestamp_output(solution_str)
    start_time, end_time = 0, 0
    gt_start, gt_end = ground_truth
    s, e = max(gt_start, 0), min(gt_end, duration)
    if parsed_times:
        start_time, end_time = parsed_times
        from_number = start_time
        to_number = end_time

        intersection = max(0, min(to_number, e) - max(from_number, s))
        union = max(to_number, e) - min(from_number, s)
        if union > 0:
            iou = intersection / union  # 0.1 0.3

        gt_start_norm = 1.0 * s / duration
        gt_end_norm = 1.0 * e / duration
        pred_start_norm = 1.0 * start_time / duration
        pred_end_norm = 1.0 * end_time / duration
        if split == "train":
            reward = (
                iou
                * (1 - abs(gt_start_norm - pred_start_norm))
                * (1 - abs(gt_end_norm - pred_end_norm))
            )
        else:
            reward = iou

    return reward, iou


def format_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    **kwargs,
):
    """Reward function that checks if the completion has a specific format."""
    content = solution_str.strip()
    
    if kwargs.get("nothink", False):
        pattern = re.compile(
            r"^<answer>.*?</answer>$", 
            re.DOTALL
        )
        match = re.fullmatch(pattern, content)
    
        return 1.0 if match else 0.0

    required_tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in required_tags:
        if content.count(tag) != 1:
            return 0.0
    pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?</answer>$", 
        re.DOTALL
    )
    match = re.fullmatch(pattern, content)
    
    return 1.0 if match else 0.0

def temporal_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    temp_margin,
    temp_alpha,
    **kwargs,
):
    """Reward function that checks if the completion has a specific format."""
    diff_log_prob = extra_info["diff_log_prob"]
    diff_log_prob_thresh = extra_info["diff_log_prob_thresh"]
    if diff_log_prob > diff_log_prob_thresh + temp_margin:
        return temp_alpha
    else:
        return 0.0


def compute_vtg_score(**kwargs):
    format_score = format_reward(**kwargs)
    iou_score, iou = iou_timestamp_reward_v2(**kwargs)
    score = format_score + iou_score

    if "diff_log_prob" in kwargs["extra_info"]:
        temp_score = temporal_reward(**kwargs)
        if iou >= kwargs["temp_iou_thresh"]:
            score = score + temp_score
    else:
        temp_score = 0.0

    return {
        "score": score,
        "format_score": format_score,
        "iou_score": iou_score,
        "temp_score": temp_score,
        "iou": iou,
    }

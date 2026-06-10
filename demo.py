import gradio as gr
import re
import logging
import argparse
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.ERROR)

TOTAL_VIDEO_TOKENS = 4096
MAX_FRAMES = 1024
FPS = 2

PROMPT = """To accurately pinpoint the event "{query}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx).

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

def process_output(output_text):
    think_content = re.sub(r'<think>(.*?)</think>', r'**Thought:** \1\n', output_text, flags=re.DOTALL)
    final_output = re.sub(r'<answer>(.*?)</answer>', r'**Answer:** \1\n', think_content, flags=re.DOTALL)
    return final_output

def create_demo(processor, llm):
    def predict(video_file_path, user_input, chat_history_state, display):
        if video_file_path is None:
            gr.Warning("Please upload a video first.")
            return chat_history_state, display

        if not user_input:
            gr.Warning("Please enter your query.")
            return chat_history_state, display

        formatted_query = PROMPT.format(query=user_input)

        try:
            video_info = {
                "type": "video",
                "video": video_file_path,
                "total_pixels": TOTAL_VIDEO_TOKENS * (processor.image_processor.patch_size * 2) ** 2,
                "min_pixels": 4 * (processor.image_processor.patch_size * 2) ** 2,
                "max_frames": MAX_FRAMES,
                "fps": FPS,
            }
            
            messages = [{
                "role": "user",
                "content": [
                    video_info,
                    {"type": "text", "text": formatted_query}
                ],
            }]

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
                
            multi_modal_data = {"video": videos}
            prompt_data = {
                "prompt": raw_prompt, 
                "multi_modal_data": multi_modal_data, 
                'mm_processor_kwargs': video_kwargs
            }

            sampling_params = SamplingParams(
                repetition_penalty=1.05, 
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                stop_token_ids=[151645, 151643], 
                max_tokens=1024,
                include_stop_str_in_output=False,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )

            outputs = llm.generate(
                prompts=[prompt_data],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            bot_response = outputs[0].outputs[0].text.strip()
        except Exception as e:
            gr.Error(f"Error during processing or model inference: {e}")
            print(f"Error in execution: {e}")
            return chat_history_state, display

        chat_history_state.append((formatted_query, bot_response))
        display.append({"role": "user", "content": user_input})
        display.append({"role": "assistant", "content": process_output(bot_response)})
        
        return chat_history_state, display

    def clear_all(video_file):
        return None, [], []

    with gr.Blocks() as demo:
        gr.Markdown("# TaRO Demo")

        chat_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.Video(label="Upload Video")

            with gr.Column(scale=2):
                chatbot_display = gr.Chatbot(label="Messages", height=500, allow_tags=True)
                
                with gr.Row():
                    user_textbox = gr.Textbox(
                        label="Query",
                        placeholder="a woman opens the door",
                        scale=4
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear")
        
        gr.Examples(
            examples=[
                ["assets/demo.mp4", "The man drank water for the first time."],
                ["assets/demo.mp4", "The man drank water for the second time."],
                ["assets/demo.mp4", "The man drank water for the third time."],
            ],
            inputs=[video_file, user_textbox],
            label="Example Queries"
        )

        submit_btn.click(
            fn=predict,
            inputs=[video_file, user_textbox, chat_history_state, chatbot_display],
            outputs=[chat_history_state, chatbot_display]
        )
        
        user_textbox.submit(
            fn=predict,
            inputs=[video_file, user_textbox, chat_history_state, chatbot_display],
            outputs=[chat_history_state, chatbot_display]
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[video_file], 
            outputs=[video_file, chatbot_display, chat_history_state]
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Gradio Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    print(f"Loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading vLLM model from: {args.model} ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.8,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"image": 0, "video": 1}
    )
    print("Model and Processor loaded. Gradio is starting...")

    demo = create_demo(processor, llm)
    demo.launch(share=True, debug=True)
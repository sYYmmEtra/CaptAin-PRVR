import json
import time
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from AutoPrompt import AutoPrompt
from pydantic import BaseModel

# 模型加载
model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class MergeOutput(BaseModel):
    start_time: float
    end_time: float
    caption: str

class MergedOutputList(BaseModel):
    merged: list[MergeOutput]

# Prompt模板
prompt = AutoPrompt(
    prompt_template="""
Given a list of video caption segments (each with start_time, end_time, and caption) which is in chronological order, merge only adjacent segments whose captions are semantically overlapping or describe the same event. Do not merge segments that are just temporally adjacent but describe different content.

After merging, ensure the total number of captions is at least 2(even if the two merged segments have overlapping content). For each merged segment, combine all information from the original captions(must avoiding information loss!), avoid redundancy, and write fluent, simple English.

Input segments (JSON array):
{segments}

Only return the merged captions in the following JSON format:
""",
    output_basemodel=MergedOutputList
)

# 提取JSON
import re
def extract_json_from_response(response):
    try:
        # Support ```json and ```schema_json fenced blocks
        json_match = re.search(r'```(?:json|schema_json)\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_data = json.loads(json_str)
            return json_data
        # 兼容无markdown包裹的情况
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            json_data = json.loads(json_str)
            return json_data
        return None
    except Exception as e:
        print(f"JSON提取错误: {e}")
        return None

def chat(segments: list[dict], max_retries: int = 10):
    """
    Merge a list of caption segments (with start/end/caption) into merged captions using the language model.
    segments: List[dict] with 'start_time', 'end_time', 'caption'
    Returns: List[dict] with 'start_time', 'end_time', 'caption'
    """
    seg_text = json.dumps(segments, ensure_ascii=False, indent=2)
    min_start = min(seg['start_time'] for seg in segments)
    max_end = max(seg['end_time'] for seg in segments)
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "user", "content": prompt.format(segments=seg_text)}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)
            json_data = extract_json_from_response(response)
            # Expecting {"merged": [...]}
            if (
                isinstance(json_data, dict)
                and "merged" in json_data
                and isinstance(json_data["merged"], list)
                and all('caption' in d and 'start_time' in d and 'end_time' in d for d in json_data["merged"])
            ):
                all_valid = True
                for d in json_data["merged"]:
                    s = float(d["start_time"])
                    e = float(d["end_time"])
                    if s < min_start or e > max_end or s > e or (e - s) > 0.8 :
                        all_valid = False
                        break
                if all_valid:
                    return json_data["merged"]
            print(f"Attempt {attempt + 1}: Failed to extract valid merged captions, retrying...\nResponse: {response}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Generation failed: {e}")
        time.sleep(2)
    print("All retries failed, returning original segments as fallback.")
    # fallback: return original segments
    return [
        {
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "caption": seg["caption"]
        } for seg in segments
    ]

def process_json(input_path: str, output_path: str):
    """
    Read the input json where each video has multiple segment captions, merge them
    into fewer captions (but at least 2) using the LLM, and write the result to *output_path*.
    The output format for every video is a list of objects:
    {
        "start_time": <float>,
        "end_time": <float>,
        "caption": "<merged caption>"
    }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data: dict[str, list[dict]] = json.load(f)
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            new_data: dict = json.load(f)
        print(f"Detected existing output file, {len(new_data)} videos already processed.")
    except FileNotFoundError:
        new_data = {}
    for vid, items in tqdm(data.items(), desc="Merging captions"):
        if vid in new_data:
            continue
        merged_results = chat(items)
        new_data[vid] = merged_results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Finished processing. Results saved to {output_path}")

if __name__ == "__main__":
    process_json(
        input_path="./anet_test_020.json",
        output_path="./anet_test_020_merged.json"
    ) 
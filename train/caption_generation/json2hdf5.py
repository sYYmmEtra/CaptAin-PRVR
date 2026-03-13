import torch
import h5py
import json
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import random

# ===== 加载 tokenizer 和模型 =====
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta = RobertaModel.from_pretrained('roberta-large').to('cuda')
roberta.eval()

# ===== 读取已清洗的 JSON 文件 =====
# 

json_path = "path_to_your_json_file" 
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ===== 输出文件路径 =====
output_h5_path = "path_of_output_hdf5_file"
h5f = h5py.File(output_h5_path, 'w')

# ===== 设置 batch size =====
BATCH_SIZE = 32

# ===== 收集所有 (key, caption) 对 =====
items = []
for video_id, segments in data.items():
    for seg_idx, seg in enumerate(segments):
        start = seg["start_time"]
        end = seg["end_time"]
        caption = seg["caption"]
        h5_key = f"{video_id}_{start:.2f}-{end:.2f}"
        items.append((h5_key, caption))

# ===== 按 batch 编码并写入 HDF5 =====
for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i + BATCH_SIZE]
    keys = [x[0] for x in batch]
    captions = [x[1] for x in batch]

    # 编码 batch captions
    encoded = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    with torch.no_grad():
        outputs = roberta(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

    # 写入每个句子的特征
    for j, key in enumerate(keys):
        # 每一句的 token 数量可能不同，所以按 j 分别写入
        length = attention_mask[j].sum().item()  # 真实 token 数
        token_vecs = token_embeddings[j, :length, :].cpu().numpy().astype(np.float32)
        h5f.create_dataset(key, data=token_vecs)

h5f.close()

print(f"全部特征成功写入：{output_h5_path}")
print(f"总共写入的caption数量: {len(items)}")
print(f"总共写入的video数量: {len(data)}")

import os
import re
import matplotlib.pyplot as plt
import argparse

def parse_log_file(log_path, loss_names):
    loss_data = {name: [] for name in loss_names}
    epoch_list = []
    
    print("文件是否存在：", os.path.exists(log_path))
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[Loss]' in line and '[Epoch]' in line:
                # 提取 Epoch 数字
                epoch_match = re.search(r'\[Epoch\]\s*(\d+)', line)
                if not epoch_match:
                    continue
                epoch = int(epoch_match.group(1))
                epoch_list.append(epoch)

                # 用空格分隔所有词（键和值交替出现）
                tokens = line.strip().split()
                token_dict = {}
                for i in range(len(tokens) - 1):
                    if tokens[i] in loss_names:
                        try:
                            token_dict[tokens[i]] = float(tokens[i + 1])
                        except ValueError:
                            token_dict[tokens[i]] = None

                for loss_name in loss_names:
                    loss_data[loss_name].append(token_dict.get(loss_name, None))

    return epoch_list, loss_data

def plot_losses(epoch_list, loss_data, results_dir):
    plt.figure(figsize=(10, 6))
    for loss_name, values in loss_data.items():
        if all(v is None for v in values):
            print(f"Warning: Loss '{loss_name}' not found in log.")
            continue
        plt.plot(epoch_list, values, marker='o', label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    results_dir = os.path.join(results_dir, 'loss_curve.png')
    plt.savefig(results_dir)
    print(f'图像已保存至: {results_dir}')

def get_path(results_dir,losses):
    log_path = os.path.join(results_dir, "train.log.txt")
    epoch_list, loss_dict = parse_log_file(log_path, losses)
    plot_losses(epoch_list, loss_dict, results_dir)

if __name__ == "__main__":
    results_dir = "path_to_your_result"
    losses = ["loss_overall"]
    losses = ["clip_nce_loss", "clip_trip_loss","frame_nce_loss","frame_trip_loss","caption_hinge_loss", "loss_overall"]
    get_path(results_dir,losses)
    # train.py:     import plot_loss plot_loss.get_path(results_dir, losses)
    # config.py：   parser.add_argument("--losses", nargs='+', default=["loss_overall"], help="the loss names to plot, separated by spaces")
    # .sh 调用：    --losses clip_nce_loss clip_trip_loss loss_overall
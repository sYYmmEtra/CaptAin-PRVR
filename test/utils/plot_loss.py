import os
import re
import matplotlib.pyplot as plt

class LossPlotter:
    def __init__(self, results_dir, loss_names):
        self.results_dir = results_dir
        self.loss_names = loss_names
        self.log_path = os.path.join(results_dir, "train.log.txt")

    def parse_log_file(self):
        loss_data = {name: [] for name in self.loss_names}
        epoch_list = []

        print("日志文件存在：" if os.path.exists(self.log_path) else "找不到日志文件：", self.log_path)

        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '[Loss]' in line and '[Epoch]' in line:
                    epoch_match = re.search(r'\[Epoch\]\s*(\d+)', line)
                    if not epoch_match:
                        continue
                    epoch = int(epoch_match.group(1))
                    epoch_list.append(epoch)

                    tokens = line.strip().split()
                    token_dict = {}
                    for i in range(len(tokens) - 1):
                        if tokens[i] in self.loss_names:
                            try:
                                token_dict[tokens[i]] = float(tokens[i + 1])
                            except ValueError:
                                token_dict[tokens[i]] = None

                    for loss_name in self.loss_names:
                        loss_data[loss_name].append(token_dict.get(loss_name, None))

        return epoch_list, loss_data

    def plot_losses(self):
        epoch_list, loss_data = self.parse_log_file()

        plt.figure(figsize=(10, 6))
        for loss_name, values in loss_data.items():
            if all(v is None for v in values):
                print(f"⚠️ Warning: Loss '{loss_name}' not found in log.")
                continue
            plt.plot(epoch_list, values, marker='o', label=loss_name)

        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('a+b+c')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.results_dir, 'a+b+c.png')
        plt.savefig(output_path)
        print(f'图像已保存至: {output_path}')

# ======================= demo 测试入口 =======================
if __name__ == "__main__":
    results_dir = "path_to_your_model_ckpt"
    loss_names = ["clip_nce_loss","clip_trip_loss","frame_nce_loss","frame_trip_loss","local_hinge_loss","global_soft_pos_loss","video_grounding_loss"]
    
    plotter = LossPlotter(results_dir, loss_names)
    plotter.plot_losses()

    # train.py:     import plot_loss plot_loss.get_path(results_dir, plot_loss)
    # config.py：   parser.add_argument("--plot_losses", nargs='+', default=["loss_overall"], help="the loss names to plot, separated by spaces")
    # .sh 调用：    --plot_losses clip_nce_loss clip_trip_loss loss_overall
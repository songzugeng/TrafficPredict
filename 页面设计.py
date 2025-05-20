import tkinter as tk
from tkinter import ttk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# 配置中文显示
try:
    rcParams['font.sans-serif'] = ['Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体设置失败: {e}")

matplotlib.use('TkAgg')


class TrafficForecastApp:
    def __init__(self, master):
        self.master = master
        master.title("交通流量预测系统超级无敌终极版 v12.3")
        master.geometry("1200x800")

        # 初始化变量
        self.horizon = 12
        self.selected_model = None
        self.sensor_ids = None
        self.results = {}
        self.scaler = MinMaxScaler()
        self.adj = None

        # 创建控件
        self.create_widgets()

        # 状态标志
        self.training = False
        self.trained = False

    def create_widgets(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.master, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 预测时长选择
        ttk.Label(control_frame, text="预测时长选择:").pack(pady=5)
        self.horizon_var = tk.StringVar(value="12")
        ttk.Combobox(control_frame, textvariable=self.horizon_var,
                     values=["12", "24"], state="readonly").pack()

        # 开始训练按钮
        ttk.Button(control_frame, text="开始训练",
                   command=self.start_training).pack(pady=10)

        # 模型选择
        model_frame = ttk.LabelFrame(control_frame, text="模型选择")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.model_var = tk.StringVar()
        models = ["LiteSTGCN", "StandardSTGCN", "LSTM", "RNN"]
        for model in models:
            ttk.Radiobutton(model_frame, text=model, variable=self.model_var,
                            value=model).pack(anchor=tk.W)

        # 可视化选项
        vis_frame = ttk.LabelFrame(control_frame, text="可视化选项")
        vis_frame.pack(fill=tk.X, padx=5, pady=5)

        buttons = [
            ("训练损失曲线", self.show_loss),
            ("随机传感器预测", self.show_sensors),
            ("传感器335对比", self.show_335),
            ("性能指标", self.show_metrics)
        ]
        for text, cmd in buttons:
            ttk.Button(vis_frame, text=text, command=cmd).pack(fill=tk.X, pady=2)

        # 状态栏
        self.status = ttk.Label(control_frame, text="就绪")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # 绘图区域
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def start_training(self):
        if self.training:
            self.status.config(text="正在训练中...")
            return

        self.horizon = int(self.horizon_var.get())
        self.status.config(text="初始化训练...")
        self.training = True
        self.trained = False

        # 后台线程运行训练
        threading.Thread(target=self.run_training).start()

    def run_training(self):
        # 数据预处理
        try:
            df = pd.read_excel('traffic.xlsx', index_col='date', parse_dates=True)
            n_sensors = 350

            # 构建邻接矩阵
            self.adj = np.eye(n_sensors, k=1) + np.eye(n_sensors, k=-1)
            self.adj = torch.FloatTensor(self.adj / self.adj.sum(axis=1, keepdims=True))

            # 数据归一化
            data = self.scaler.fit_transform(df.values.astype(np.float32))

            # 时间特征
            time_feat = np.array([[np.sin(2 * np.pi * h / 24)] for h in df.index.hour], dtype=np.float32)

            # 生成随机传感器ID（保证所有模型使用同一组）
            self.sensor_ids = np.random.choice(n_sensors, 5, replace=False)

            # 创建数据集
            class TrafficDataset(Dataset):
                def __init__(self, data, time_feat, window=24, horizon=12):
                    self.window = window
                    self.horizon = horizon
                    self.data = data
                    self.time = time_feat

                def __len__(self):
                    return len(self.data) - self.window - self.horizon

                def __getitem__(self, idx):
                    x = self.data[idx:idx + self.window].T
                    t = self.time[idx:idx + self.window]
                    y = self.data[idx + self.window:idx + self.window + self.horizon].T
                    return torch.FloatTensor(x), torch.FloatTensor(t), torch.FloatTensor(y)

            dataset = TrafficDataset(data, time_feat, 24, self.horizon)

            # 数据集划分
            total_length = len(dataset)
            train_size = int(0.6 * total_length)
            val_size = (total_length - train_size) // 2
            test_size = total_length - train_size - val_size
            train_set, val_set, test_set = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size])

            # 初始化模型
            models = {
                "LiteSTGCN": LiteSTGCN(self.adj),
                "StandardSTGCN": StandardSTGCN(self.adj),
                "LSTM": LSTM_Model(),
                "RNN": RNN_Model()
            }

            # 训练和评估每个模型
            for name, model in models.items():
                self.status.config(text=f"正在训练 {name}...")
                train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
                test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

                # 训练模型
                model, train_loss, val_loss = self.train_model(model, train_loader, val_loader)

                # 评估模型
                pred, true, metrics = self.evaluate_model(model, test_loader)

                # 保存结果
                self.results[name] = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "pred": pred,
                    "true": true,
                    "metrics": metrics
                }

            self.training = False
            self.trained = True
            self.status.config(text="训练完成！")

        except Exception as e:
            self.status.config(text=f"错误: {str(e)}")
            self.training = False

    def train_model(self, model, train_loader, val_loader):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        train_losses, val_losses = [], []

        for epoch in range(50):
            # 训练阶段
            model.train()
            total_loss = 0
            for x, t, y in train_loader:
                optimizer.zero_grad()
                pred = model(x, t) if isinstance(model, (StandardSTGCN, LiteSTGCN)) else model(x)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, t_val, y_val in val_loader:
                    pred_val = model(x_val, t_val) if isinstance(model, (StandardSTGCN, LiteSTGCN)) else model(x_val)
                    val_loss += criterion(pred_val, y_val).item()

            train_losses.append(total_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

        return model, train_losses, val_losses

    def evaluate_model(self, model, test_loader):
        preds, trues = [], []
        with torch.no_grad():
            for x, t, y in test_loader:
                pred = model(x, t) if isinstance(model, (StandardSTGCN, LiteSTGCN)) else model(x)
                preds.append(pred.numpy())
                trues.append(y.numpy())

        # 逆标准化
        preds = np.concatenate(preds, axis=0).transpose(0, 2, 1).reshape(-1, 350)
        preds = self.scaler.inverse_transform(preds).reshape(-1, self.horizon, 350).transpose(0, 2, 1)

        trues = np.concatenate(trues, axis=0).transpose(0, 2, 1).reshape(-1, 350)
        trues = self.scaler.inverse_transform(trues).reshape(-1, self.horizon, 350).transpose(0, 2, 1)

        # 计算指标
        mae = mean_absolute_error(trues.ravel(), preds.ravel())
        rmse = np.sqrt(mean_squared_error(trues.ravel(), preds.ravel()))

        return preds, trues, {"mae": mae, "rmse": rmse}

    def show_loss(self):
        if not self.check_ready():
            return

        model = self.model_var.get()
        data = self.results.get(model)
        if not data:
            self.status.config(text="该模型未训练")
            return

        fig = plt.figure(figsize=(10, 5))
        plt.plot(data['train_loss'], label='训练损失')
        plt.plot(data['val_loss'], label='验证损失', linestyle='--')
        plt.title(f'{model} 损失曲线')
        plt.legend()
        self.display_plot(fig)

    def show_sensors(self):
        if not self.check_ready():
            return

        model = self.model_var.get()
        data = self.results.get(model)
        if not data:
            self.status.config(text="该模型未训练")
            return

        fig = plt.figure(figsize=(15, 10))
        for i, sid in enumerate(self.sensor_ids, 1):
            plt.subplot(5, 1, i)
            plt.plot(data['pred'][-1, sid], label='预测')
            plt.plot(data['true'][-1, sid], label='真实', alpha=0.7)
            plt.title(f'传感器 {sid}', fontsize=10)
            plt.legend()
        plt.tight_layout()
        self.display_plot(fig)

    def show_335(self):
        if not self.check_ready():
            return

        fig = plt.figure(figsize=(12, 6))
        for model in self.results:
            plt.plot(self.results[model]['pred'][-1, 334], label=model)
        plt.plot(self.results["LiteSTGCN"]['true'][-1, 334], label='真实值', linestyle='--')
        plt.title('传感器335预测对比')
        plt.legend()
        self.display_plot(fig)

    def show_metrics(self):
        if not self.check_ready():
            return

        model = self.model_var.get()
        data = self.results.get(model)
        if not data:
            self.status.config(text="该模型未训练")
            return

        win = tk.Toplevel()
        win.title("性能指标")
        text = f"MAE: {data['metrics']['mae']:.2f}\nRMSE: {data['metrics']['rmse']:.2f}"
        ttk.Label(win, text=text, font=('Arial', 14)).pack(padx=20, pady=20)

    def check_ready(self):
        if not self.trained:
            self.status.config(text="请先完成训练！")
            return False
        if not self.model_var.get():
            self.status.config(text="请选择模型！")
            return False
        return True

    def display_plot(self, fig):
        # 清除旧图形
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # 嵌入新图形
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# 模型定义（与原始代码保持一致）
class StandardSTGCN(nn.Module):
    def __init__(self, adj, in_dim=1, hid_dim=8):
        super().__init__()
        self.adj = adj
        self.hid = hid_dim
        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, (1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, (1, 1)),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, (1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, (1, 1)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hid_dim * 24 * 350, 512),
            nn.ReLU(),
            nn.Linear(512, 350 * 12)
        )

    def forward(self, x, t):
        x = x.unsqueeze(1)
        out1 = self.block1(x)
        out2 = self.block2(out1)
        res_out = out1 + out2
        res_out = res_out * t.permute(0, 2, 1).unsqueeze(2)
        return self.fc(res_out).view(-1, 350, 12)


class LiteSTGCN(nn.Module):
    def __init__(self, adj, in_dim=1, hid_dim=32):
        super().__init__()
        self.adj = adj
        self.hid_dim = hid_dim
        self.temp_conv = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim * 2, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(hid_dim * 2, hid_dim, (1, 1)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hid_dim * 24 * 350, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 350 * 12)
        )

    def forward(self, x, t):
        x = x.unsqueeze(1)
        x = self.temp_conv(x)

        # 修正 expand 参数
        adj = self.adj.unsqueeze(0).unsqueeze(0).expand(
            x.size(0),
            self.hid_dim,
            -1,
            -1
        )

        x = torch.einsum('bhnw,bhcn->bhcw', x, adj)
        x = self.spatial_conv(x)
        x = x * t.permute(0, 2, 1).unsqueeze(2)
        return self.fc(x).view(-1, 350, 12)


class LSTM_Model(nn.Module):
    def __init__(self, input_size=350, hidden_size=8, output_size=350, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * 12)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(-1, 350, 12)


class RNN_Model(nn.Module):
    def __init__(self, input_size=350, hidden_size=8, output_size=350, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * 12)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).view(-1, 350, 12)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficForecastApp(root)
    root.mainloop()
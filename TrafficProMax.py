import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import gc
import matplotlib
from matplotlib import rcParams

try:
    rcParams['font.sans-serif'] = ['Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体设置失败: {e}")

# 内存优化配置
torch.manual_seed(42)
torch.backends.optimized_for_memory = True


def main():
    # 1. 数据预处理
    print("Step 1/8: 数据加载与预处理...")
    df = pd.read_excel('traffic.xlsx', index_col='date', parse_dates=True)
    n_sensors = 350

    # 构建邻接矩阵
    adj = np.eye(n_sensors, k=1) + np.eye(n_sensors, k=-1)
    adj = torch.FloatTensor(adj / adj.sum(axis=1, keepdims=True))

    # 数据归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values.astype(np.float32))

    # 时间特征生成
    time_feat = np.array([[np.sin(2 * np.pi * h / 24)] for h in df.index.hour], dtype=np.float32)

    # 2. 数据集构建
    class TrafficDataset(Dataset):
        def __init__(self, data, time_feat, window=24, horizon=24):
            self.window = window
            self.horizon = horizon
            self.data = data
            self.time = time_feat

        def __len__(self):
            return len(self.data) - self.window - self.horizon

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.window].T  # [n_sensors, window]
            t = self.time[idx:idx + self.window]  # [window, 1]
            y = self.data[idx + self.window:idx + self.window + self.horizon].T
            return torch.FloatTensor(x), torch.FloatTensor(t), torch.FloatTensor(y)

    window, horizon = 24, 24
    dataset = TrafficDataset(data, time_feat, window, horizon)

    # 数据集划分
    total_length = len(dataset)
    train_size = int(0.6 * total_length)
    remaining = total_length - train_size
    val_size = remaining // 2
    test_size = remaining - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # 3. 模型定义
    class StandardSTGCN(nn.Module):
        def __init__(self, adj, in_dim=1, hid_dim=8):
            super().__init__()
            self.adj = adj
            self.hid = hid_dim

            # 第一时空卷积块
            self.block1 = nn.Sequential(
                nn.Conv2d(in_dim, hid_dim, (1, 3), padding=(0, 1)),
                nn.ReLU(),
                self.spatial_conv(hid_dim, adj),
                nn.ReLU()
            )

            # 第二时空卷积块（带残差）
            self.block2 = nn.Sequential(
                nn.Conv2d(hid_dim, hid_dim, (1, 3), padding=(0, 1)),
                nn.ReLU(),
                self.spatial_conv(hid_dim, adj),
                nn.ReLU()
            )

            # 输出层
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hid_dim * window * n_sensors, 512),
                nn.ReLU(),
                nn.Linear(512, n_sensors * horizon)
            )

        def spatial_conv(self, hid_dim, adj):
            return nn.Sequential(
                nn.Conv2d(hid_dim, hid_dim, (1, 1)),
                nn.ReLU()
            )

        def forward(self, x, t):
            x = x.unsqueeze(1)  # [B, 1, N, W]
            out1 = self.block1(x)
            out2 = self.block2(out1)
            res_out = out1 + out2  # 残差连接
            res_out = res_out * t.permute(0, 2, 1).unsqueeze(2)
            return self.fc(res_out).view(-1, n_sensors, horizon)

    class LiteSTGCN(nn.Module):
        def __init__(self, adj, in_dim=1, hid_dim=32):
            super().__init__()
            self.adj = adj
            self.hid = hid_dim

            # 时间卷积层
            self.temp_conv = nn.Sequential(
                nn.Conv2d(in_dim, hid_dim, (1, 5), padding=(0, 2)),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            # 空间卷积层（带瓶颈结构）
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(hid_dim, hid_dim * 2, (1, 1)),
                nn.ReLU(),
                nn.Conv2d(hid_dim * 2, hid_dim, (1, 1)),
                nn.ReLU()
            )

            # 输出层
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hid_dim * window * n_sensors, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, n_sensors * horizon)
            )

        def forward(self, x, t):
            x = x.unsqueeze(1)  # [B, 1, N, W]
            x = self.temp_conv(x)
            adj = self.adj.unsqueeze(0).unsqueeze(0).expand(x.size(0), self.hid, -1, -1)
            x = torch.einsum('bhnw,bhcn->bhcw', x, adj)
            x = self.spatial_conv(x)
            x = x * t.permute(0, 2, 1).unsqueeze(2)
            return self.fc(x).view(-1, n_sensors, horizon)

    class LSTM_Model(nn.Module):
        # 原有LSTM结构保持不变
        def __init__(self, input_size=350, hidden_size=8, output_size=350, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size * horizon)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).view(-1, n_sensors, horizon)

    class RNN_Model(nn.Module):
        # 原有RNN结构保持不变
        def __init__(self, input_size=350, hidden_size=8, output_size=350, num_layers=1):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size * horizon)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).view(-1, n_sensors, horizon)

    # 4. 训练函数（保持不变）
    def train_model(model, model_name, train_loader, val_loader):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        train_losses, val_losses = [], []

        print(f"\nTraining {model_name}...")
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

            # 记录损失
            train_loss = total_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失', linestyle='--')
        plt.title(f'{model_name} 损失曲线')
        plt.legend()
        plt.savefig(f'{model_name}_loss.png')
        plt.close()

        return model

    # 5. 训练所有模型
    print("\nStep 2/8: 初始化模型...")
    batch_size = 256
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    # 训练模型
    stgcn = LiteSTGCN(adj)
    stgcn = train_model(stgcn, "LiteSTGCN", train_loader, val_loader)

    standard_stgcn = StandardSTGCN(adj)
    standard_stgcn = train_model(standard_stgcn, "StandardSTGCN", train_loader, val_loader)

    lstm = LSTM_Model()
    lstm = train_model(lstm, "LSTM", train_loader, val_loader)

    rnn = RNN_Model()
    rnn = train_model(rnn, "RNN", train_loader, val_loader)

    # 6. 测试函数（保持不变）
    def evaluate_model(model, test_set, model_name):
        test_loader = DataLoader(test_set, batch_size=2)
        preds, trues = [], []

        with torch.no_grad():
            for x, t, y in test_loader:
                pred = model(x, t) if isinstance(model, (StandardSTGCN, LiteSTGCN)) else model(x)
                preds.append(pred.numpy())
                trues.append(y.numpy())

        # 数据逆标准化
        preds = np.concatenate(preds, axis=0).transpose(0, 2, 1).reshape(-1, n_sensors)
        preds = scaler.inverse_transform(preds).reshape(-1, horizon, n_sensors).transpose(0, 2, 1)

        trues = np.concatenate(trues, axis=0).transpose(0, 2, 1).reshape(-1, n_sensors)
        trues = scaler.inverse_transform(trues).reshape(-1, horizon, n_sensors).transpose(0, 2, 1)

        # 计算指标
        mae = mean_absolute_error(trues.ravel(), preds.ravel())
        rmse = np.sqrt(mean_squared_error(trues.ravel(), preds.ravel()))
        print(f"\n{model_name} 测试结果: MAE={mae:.2f}, RMSE={rmse:.2f}")
        return preds, trues

    # 7. 评估所有模型
    print("\nStep 3/8: 评估模型...")
    stgcn_pred, stgcn_true = evaluate_model(stgcn, test_set, "LiteSTGCN")
    standard_pred, standard_true = evaluate_model(standard_stgcn, test_set, "StandardSTGCN")
    lstm_pred, lstm_true = evaluate_model(lstm, test_set, "LSTM")
    rnn_pred, rnn_true = evaluate_model(rnn, test_set, "RNN")

    # 8. 可视化对比
    print("\nStep 4/8: 可视化对比...")
    sensor_ids = np.random.choice(n_sensors, 5, replace=False)

    def plot_comparison(preds, trues, model_name):
        plt.figure(figsize=(15, 10))
        for i, sid in enumerate(sensor_ids, 1):
            plt.subplot(5, 1, i)
            plt.plot(preds[-1, sid], label='预测值')
            plt.plot(trues[-1, sid], label='真实值', alpha=0.7)
            plt.title(f'{model_name} - 传感器 {sid}', fontsize=10)
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_name}_sensors.png')
        plt.close()

    plot_comparison(stgcn_pred, stgcn_true, "LiteSTGCN")
    plot_comparison(standard_pred, standard_true, "StandardSTGCN")
    plot_comparison(lstm_pred, lstm_true, "LSTM")
    plot_comparison(rnn_pred, rnn_true, "RNN")

    # 9. 传感器335对比
    print("\nStep 5/8: 传感器335对比...")

    def plot_sensor_335(pred_dict):
        plt.figure(figsize=(12, 6))
        for model_name, preds in pred_dict.items():
            linestyle = '-' if '预测' in model_name else '--'
            plt.plot(preds[-1, 334], label=model_name, linestyle=linestyle)
        plt.title('传感器335预测对比')
        plt.legend()
        plt.savefig('sensor335_comparison.png')
        plt.close()

    pred_dict = {
        "LiteSTGCN预测": stgcn_pred,
        "StandardSTGCN预测": standard_pred,
        "LSTM预测": lstm_pred,
        "RNN预测": rnn_pred,
        "真实值": stgcn_true
    }
    plot_sensor_335(pred_dict)


if __name__ == '__main__':
    main()
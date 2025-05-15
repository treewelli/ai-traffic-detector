import os
import yaml
import torch
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from torch.nn.functional import softmax
import torch.nn.functional as F


class RobustLSTMInference:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.logger = self.init_logger(self.config)
        self.label_map = self.config.get("label_map", {})
        self.index_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        self.device = torch.device(self.config['lstm_params']['training']['device'])

        self.model_config = self.config['lstm_params']['model']
        self.infer_config = self.config['lstm_params'].get('inference', {})
        self.dtype = torch.float32

        self.chunk_size = self.infer_config.get("chunk_size", 15000)
        self.chunk_overlap = self.infer_config.get("chunk_overlap", 0.0)
        self.topk = self.infer_config.get("topk", 1)
        self.alert_threshold = self.infer_config.get("distribution_alert_threshold", 0.3)

        self.valid_classes = self.config['lstm_params'].get('valid_classes')

        self.model = self._load_or_init_model()

    def init_logger(self, config):
        logger = logging.getLogger("RobustLSTMInference")
        logger.setLevel(config['logging']['level'])
        log_file = os.path.join(config['paths']['log_dir'], "lstm_inference.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(logging.StreamHandler())
        logger.propagate = False
        return logger

    class Attention(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = torch.nn.Linear(hidden_dim, 1)

        def forward(self, lstm_output):
            # lstm_output: [batch, seq_len, hidden_dim]
            attn_weights = F.softmax(self.attn(lstm_output), dim=1)  # [batch, seq_len, 1]
            context = torch.sum(attn_weights * lstm_output, dim=1)   # [batch, hidden_dim]
            return context

    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes, bidirectional=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.attn = RobustLSTMInference.Attention(hidden_size * (2 if bidirectional else 1))
            self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)              # [batch, seq_len, hidden]
            attn_out = self.attn(lstm_out)          # [batch, hidden]
            logits = self.fc(attn_out)              # [batch, num_classes]
            return logits


    def _load_or_init_model(self):
        model = self.LSTMModel(
            input_size=self.model_config['feature_dim'],
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout'],
            num_classes=self.num_classes,
            bidirectional=self.model_config['bidirectional']
        ).to(self.device)

        best_model_path = os.path.join(self.config['paths']['model_best'], 'best.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info(f"加载模型成功: {best_model_path}")
        else:
            self.logger.warning("未找到 best.pt，使用初始化模型")

        return model

    def _load_npz(self, file_path):
        data = np.load(file_path)
        features = data['features']
        labels = data['labels'][:, 0]
        return torch.tensor(features, dtype=self.dtype), torch.tensor(labels, dtype=torch.long)

    def _ensemble_predict(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = softmax(logits, dim=1)
            if self.valid_classes:
                mask = torch.full_like(probs, -float('inf'))
                for cls in self.valid_classes:
                    mask[:, cls] = probs[:, cls]
                probs = softmax(mask, dim=1)
            top_probs, top_classes = torch.topk(probs, k=self.topk, dim=1)
        return probs, top_probs, top_classes

    def _evaluate(self, y_true, y_pred, y_prob):
        unique_labels = np.unique(y_true)
        label_names = [self.index_to_label.get(int(i), str(i)) for i in unique_labels]

        self.logger.info("分类报告:")
        self.logger.info(classification_report(
            y_true, y_pred,
            labels=unique_labels,
            target_names=label_names,
            digits=4, zero_division=0
        ))

        if len(unique_labels) < 2:
            self.logger.warning("无法计算 ROC-AUC，只有一个类别")
            return

        try:
            y_true_oh = np.eye(self.num_classes)[y_true]
            y_true_oh_filtered = y_true_oh[:, unique_labels]
            y_prob_filtered = y_prob[:, unique_labels]
            roc_auc = roc_auc_score(y_true_oh_filtered, y_prob_filtered, average='macro', multi_class='ovr')
            self.logger.info(f"ROC-AUC (macro-ovr): {roc_auc:.4f}")
        except Exception as e:
            self.logger.warning(f"无法计算 ROC-AUC: {e}")

    def infer_file(self, file_path):
        self.logger.info(f"正在推理文件: {file_path}")

        features, labels = self._load_npz(file_path)
        features = features.to(self.device)
        labels = labels.to(self.device)

        y_preds, y_trues, y_probs = [], [], []

        step_size = int(self.chunk_size * (1 - self.chunk_overlap))

        for i in tqdm(range(0, len(features) - self.chunk_size + 1, step_size)):
            batch_x = features[i:i + self.chunk_size]
            batch_y = labels[i:i + self.chunk_size]

            prob, top_probs, top_classes = self._ensemble_predict(batch_x)
            preds = top_classes[:, 0]

            y_preds.append(preds.cpu())
            y_trues.append(batch_y.cpu())
            y_probs.append(prob.cpu())

            # 打印Top-K
            for j in range(min(3, len(batch_x))):
                top_k = [(self.index_to_label.get(top_classes[j, k].item()), top_probs[j, k].item()) for k in range(self.topk)]
                

        y_pred = torch.cat(y_preds).numpy()
        y_true = torch.cat(y_trues).numpy()
        y_prob = torch.cat(y_probs).numpy()

        # 类别分布检测
        unique, counts = np.unique(y_pred, return_counts=True)
        total = sum(counts)
        for cls_id, count in zip(unique, counts):
            cls_name = self.index_to_label.get(cls_id, str(cls_id))
            ratio = count / total
            self.logger.info(f"类别 {cls_name} 占比: {ratio:.2%}")
            if ratio > self.alert_threshold:
                self.logger.warning(f"⚠️ 警告: 类别 {cls_name} 占比超出 {self.alert_threshold:.0%} 阈值")

        self.logger.info(f"完成推理，总样本数: {len(y_true)}")
        self._evaluate(y_true, y_pred, y_prob)

    def infer_all(self):
        test_dir = self.config["paths"]["test_data"]
        npz_files = sorted(glob(os.path.join(test_dir, "*.npz")))

        if not npz_files:
            self.logger.warning("未找到测试数据文件。")
            return

        for file_path in npz_files:
            self.infer_file(file_path)


if __name__ == "__main__":
    infer = RobustLSTMInference("config/patterns.yaml")
    infer.infer_all()

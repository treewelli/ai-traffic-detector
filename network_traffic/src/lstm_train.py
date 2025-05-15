import os
import sys
import yaml
import zarr
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import threading, queue
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, classification_report
from collections import Counter
from tqdm import tqdm

from torch.utils.data import IterableDataset
import threading, queue

class DoubleBufferedZarrDataset(IterableDataset):
    def __init__(self, zarr_path, batch_size, prefetch=8):
        self.zarr_file = zarr.open(zarr_path, mode='r')
        self.features = self.zarr_file['features']
        self.labels = self.zarr_file['labels']
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.q = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()

    def data_producer(self):
        for i in range(0, len(self.labels), self.batch_size):
            if self._stop.is_set():
                break
            x = self.features[i:i+self.batch_size]
            y = self.labels[i:i+self.batch_size]
            x_tensor = torch.from_numpy(np.array(x)).float()
            y_tensor = torch.from_numpy(np.array(y)).long()
            self.q.put((x_tensor, y_tensor))
        self.q.put(None)  # 结束标志

    def __iter__(self):
        self._stop.clear()
        t = threading.Thread(target=self.data_producer, daemon=True)
        t.start()
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item

    def __del__(self):
        self._stop.set()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_f1 = None
        self.patience_counter = 0

    def __call__(self, f1, model):
        if self.best_f1 is None or f1 > self.best_f1 + self.delta:
            self.best_f1 = f1
            self.patience_counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        alpha = self.alpha[targets].unsqueeze(1) if self.alpha is not None else 1.0
        focal_weight = alpha * (1 - probs) ** self.gamma
        loss = -focal_weight * log_probs * targets_one_hot
        return loss.sum(dim=1).mean()


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=bidirectional)
        d = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Linear(d, 1)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = x.contiguous()
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, T, H)
        attn_scores = self.attn(lstm_out)             # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out).squeeze(1)  # (B, H)
        logits = self.fc(context)                             # (B, C)
        return logits


class LSTMTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.paths = self.config['paths']
        self.logger = self.init_logger(self.config)
        self.model_cfg = self.config['lstm_params']['model']
        self.train_cfg = self.config['lstm_params']['training']
        self.device = torch.device(self.train_cfg['device'] if torch.cuda.is_available() else 'cpu')
        self.model_best = os.path.abspath(self.config['paths']['model_best'])
        os.makedirs(self.model_best, exist_ok=True)
        self.mode = 'auto'
        self.selected_files = []
        self.config["processed_files_log"] = os.path.abspath(
            os.path.join(self.config['paths']['log_dir'], "lstm_trained_files.log")
        )
        self.processed_files = self._load_processed_files()
        self.label_map = self.config.get("label_map", {})
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def init_logger(self, config, output_widget=None):
        from PyQt5.QtCore import QObject, pyqtSignal

        logger = logging.getLogger("LSTMTrainer")
        logger.setLevel(config['logging']['level'])

        log_file = os.path.join(config['paths']['log_dir'], "lstm_training.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        if not logger.handlers:
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(stream_handler)

        if output_widget is not None:
            already_has_gui_handler = any(
                isinstance(h, logging.Handler) and hasattr(h, 'log_signal') for h in logger.handlers
            )
            if not already_has_gui_handler:
                class QTextEditLogger(QObject, logging.Handler):
                    log_signal = pyqtSignal(str)

                    def __init__(self, widget):
                        QObject.__init__(self)
                        logging.Handler.__init__(self)
                        self.widget = widget
                        self.log_signal.connect(self.widget.appendPlainText)

                    def emit(self, record):
                        log_entry = self.format(record)
                        self.log_signal.emit(log_entry)

                gui_handler = QTextEditLogger(output_widget)
                gui_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                logger.addHandler(gui_handler)

        logger.propagate = False
        return logger

    def _load_processed_files(self) -> set:
        if os.path.exists(self.config["processed_files_log"]):
            with open(self.config["processed_files_log"], "r") as f:
                return set(os.path.abspath(line.strip()) for line in f)
        return set()

    def _save_processed_file(self, input_path: str):
        input_path = os.path.abspath(input_path)
        with open(self.config["processed_files_log"], "a") as f:
            f.write(f"{input_path}\n")
            f.flush()
        self.processed_files.add(input_path)

    def _get_all_zarr_files(self) -> list:
        """获取待训练的 .zarr 特征文件，兼容手动和自动模式"""
        matched_files = []
        skipped = 0
        input_dir = self.config['lstm_params']['data']['input_path']

        self.processed_files = self._load_processed_files()

        if getattr(self, "mode", "auto") == "manual":
            for input_path in getattr(self, "selected_files", []):
                if input_path in self.processed_files:
                    skipped += 1
                    continue
                matched_files.append(input_path)
            self.logger.info(
                f"手动模式：跳过 {skipped} 个已处理文件，{len(matched_files)} 个新文件需要处理"
            )
        else:
            for f in os.listdir(input_dir):
                if f.endswith('.zarr') and f.startswith('trainset_'):
                    input_path = os.path.join(input_dir, f)
                    if input_path in self.processed_files:
                        skipped += 1
                        continue
                    matched_files.append(input_path)
            self.logger.info(
                f"自动模式：跳过 {skipped} 个已处理文件，{len(matched_files)} 个新文件需要处理"
            )

        return matched_files

    def _compute_class_weights(self, labels):
        labels = np.array(labels)
        valid_classes = np.unique(labels)

        # 自动更新 self.num_classes 为最大标签 + 1
        self.num_classes = int(valid_classes.max()) + 1
        self.logger.info(f"⚙️ 动态设置 num_classes = {self.num_classes}")

        # 动态类别权重分配
        try:
            weights = compute_class_weight(class_weight='balanced', classes=valid_classes, y=labels)
            weight_tensor = torch.zeros(self.num_classes, device=self.device)
            for i, cls in enumerate(valid_classes):
                weight_tensor[int(cls)] = weights[i]
            self.logger.info(f"类别权重: {weight_tensor.cpu().numpy().tolist()}")
            return weight_tensor
        except Exception as e:
            self.logger.error(f"类别权重计算失败: {e}")
            return torch.ones(self.num_classes, device=self.device)


    def _get_model(self):
        return LSTMModel(
            input_size=self.model_cfg['feature_dim'],
            hidden_size=self.model_cfg['hidden_size'],
            num_layers=self.model_cfg['num_layers'],
            dropout=self.model_cfg['dropout'],
            num_classes=self.num_classes,
            bidirectional=self.model_cfg['bidirectional']
        ).to(self.device)

    def _train_one_epoch(self, model, loader, optimizer, scaler, loss_fn):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for i, (x, y) in enumerate(tqdm(loader)):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if i == 0:
                self.logger.info(f"[DEBUG] 输入 shape: {x.shape}, 是否 contiguous: {x.is_contiguous()}, 设备: {x.device}")

            with torch.amp.autocast('cuda', enabled=self.train_cfg['use_amp']):
                outputs = model(x)
                torch.cuda.synchronize()
                loss = loss_fn(outputs, y)

            scaler.scale(loss).backward()

            if (i + 1) % self.train_cfg['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            batch_count += 1
        return running_loss / max(1, batch_count)

    def _validate(self, model, loader):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                outputs = model(x)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(y.numpy())

        f1 = f1_score(targets, preds, average='weighted')
        labels = list(range(self.num_classes))
        target_names = [self.inverse_label_map.get(i, f"class_{i}") for i in labels]
        report = classification_report(targets, preds, labels=labels, target_names=target_names)
        self.logger.info("\n" + report)
        return f1

    def _train_on_pair(self, train_path, val_path):
        train_set = DoubleBufferedZarrDataset(train_path, batch_size=2048, prefetch=8)
        val_set = DoubleBufferedZarrDataset(val_path, batch_size=2048, prefetch=8)

        y_train = train_set.labels[:]
        if len(np.unique(y_train)) < 2:
            self.logger.warning(f"{train_path} 训练集类别不足，跳过。")
            return

        train_loader = DataLoader(
            train_set,
            batch_size=None,
            num_workers=1
        )
        val_loader = DataLoader(
            val_set,
            batch_size=None,
            num_workers=1
        )

        class_weights = self._compute_class_weights(y_train)

        model = self._get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scaler = torch.amp.GradScaler('cuda', enabled=self.train_cfg['use_amp'])
        loss_fn = FocalLoss(alpha=class_weights)

        best_f1 = 0.0
        best_model_path = os.path.join( self.model_best, os.path.basename(train_path).replace('trainset_', '').replace('.zarr', '_best.pt'))
        early_stopper = EarlyStopping(patience=3, path=best_model_path)
        # 在训练前打印 label 分布
        self.logger.info(f"训练标签分布: {np.unique(y_train, return_counts=True)}")

        for epoch in range(self.train_cfg['num_epochs']):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, scaler, loss_fn)
            val_f1 = self._validate(model, val_loader)
            self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val F1 = {val_f1:.4f}")
            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                self.logger.info("保存新最佳模型")

            if early_stopper(val_f1, model):
                self.logger.info("早停触发，终止训练")
                break

            ema_path = os.path.join(self.model_best, "best.pt")
            ema_fusion = EMAModelFusion(model, alpha=0.9, save_path=ema_path, device=self.device, num_classes=self.num_classes)
            if ema_fusion.has_sufficient_diversity(best_model_path, val_loader):
                ema_fusion.update(best_model_path, val_loader)
                self.logger.info("已更新滑动平均模型 best.pt")
                self._save_processed_file(train_path)
            else:
                self.logger.warning("跳过 EMA 更新：预测类别不足")

    def run(self):
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision('high')  # PyTorch >= 2.0
        except AttributeError:
            pass

        train_files = self._get_all_zarr_files()
        self.logger.info("\n====== LSTM 模型训练启动 ======")
        self.logger.info(f"训练模式: {self.mode}")
        self.logger.info(f"训练设备: {self.device}")
        self.logger.info(f"启用 AMP 混合精度: {self.train_cfg['use_amp']}")
        for train_path in train_files:
            basename = os.path.basename(train_path)
            timestamp = basename.replace("trainset_", "").replace(".zarr", "")
            val_path = os.path.join(self.config['lstm_params']['data']['input_path'], f"valset_{timestamp}.zarr")
            if not os.path.exists(val_path):
                self.logger.warning(f"未找到对应验证集文件: {val_path}，跳过")
                continue
            self._train_on_pair(train_path, val_path)

            # 更新处理记录日志
            log_path = os.path.join(self.config['paths']['log_dir'], "lstm_trained_files.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(train_path + "\n")

class EMAModelFusion:
    def __init__(self, base_model, alpha=0.9, save_path="best.pt", device='cpu', num_classes=2, diversity_threshold=2):
        self.global_model = base_model.to(device)
        self.alpha = alpha
        self.save_path = save_path
        self.device = device
        self.num_classes = num_classes
        self.diversity_threshold = diversity_threshold

        if os.path.exists(self.save_path):
            self.global_model.load_state_dict(torch.load(self.save_path, map_location=self.device))

    def _predict_class_diversity(self, model, data_loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                logits = model(x_batch)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred)
        return len(set(preds)), Counter(preds)

    def update(self, new_model_path, validation_loader=None):
        new_model = type(self.global_model)(
            input_size=self.global_model.lstm.input_size,
            hidden_size=self.global_model.lstm.hidden_size,
            num_layers=self.global_model.lstm.num_layers,
            dropout=self.global_model.lstm.dropout,
            num_classes=self.num_classes,
            bidirectional=self.global_model.lstm.bidirectional
        ).to(self.device)

        new_model.load_state_dict(torch.load(new_model_path, map_location=self.device))

        if validation_loader is not None:
            diversity, pred_counter = self._predict_class_diversity(new_model, validation_loader)
            if diversity < self.diversity_threshold:
                print(f"❌ 跳过 EMA 更新: 预测类别数太少 ({diversity}) -> {dict(pred_counter)}")
                return
            else:
                print(f"✅ 满足多样性要求，继续 EMA 更新 -> 预测类别: {dict(pred_counter)}")

        global_params = self.global_model.state_dict()
        new_params = new_model.state_dict()

        for key in global_params:
            global_params[key] = self.alpha * global_params[key] + (1 - self.alpha) * new_params[key]

        self.global_model.load_state_dict(global_params)
        torch.save(self.global_model.state_dict(), self.save_path)
        print("✅ EMA 模型已更新并保存")

    def has_sufficient_diversity(self, model_path, val_loader):
        model = type(self.global_model)(
            input_size=self.global_model.lstm.input_size,
            hidden_size=self.global_model.lstm.hidden_size,
            num_layers=self.global_model.lstm.num_layers,
            dropout=self.global_model.lstm.dropout,
            num_classes=self.num_classes,
            bidirectional=self.global_model.lstm.bidirectional
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(self.device)
                outputs = model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        return len(set(all_preds)) >= self.diversity_threshold
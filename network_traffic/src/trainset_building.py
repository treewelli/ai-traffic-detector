import os
import gc
import sys
import glob
import yaml
import h5py
import zarr
import torch
import psutil
import logging
import threading
import numpy as np
import pandas as pd
import cupy as cp
from numcodecs import Blosc
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple
from src.chunk_sizer import DynamicChunkSizer

class TrainsetBuilder:
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 设置已处理文件记录路径
        self.config["processed_files_log"] = os.path.abspath(
            os.path.join(self.config['paths']['log_dir'], "trainset_built_files.log")
        )
        self.mode = "auto"
        self.selected_files = []
        self.chunk_sizer = DynamicChunkSizer(self.config)
        self.logger = self.init_logger(self.config)
        self.processed_files = self._load_processed_files()
        self._setup_gpu()

    def init_logger(self, config, output_widget=None):
        from PyQt5.QtCore import QObject, pyqtSignal

        logger = logging.getLogger("TrainsetBuilder")
        logger.setLevel(config['logging']['level'])

        log_file = os.path.join(config['paths']['log_dir'], "trainset_building.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # ✅ 如果没有任何Handler，先添加基本的文件+终端
        if not logger.handlers:
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(stream_handler)

        # ✅ 无论如何，单独补挂 GUI日志Handler
        if output_widget is not None:
            # 检查是否已有绑定到控件的Handler
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

    def _setup_gpu(self):
        self.gpu_enabled = False
        self.gpu_stream = None

        if self.config['lstm_params']['training']['device'] == 'cuda' and cp.is_available():
            self.gpu_enabled = True
            self.gpu_stream = cp.cuda.Stream(non_blocking=True)
            self.logger.info(f"GPU设备已启用，创建CUDA流 ")
        else:
            self.logger.info("使用CPU处理")

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

    def get_feature_zarr(self) -> List[Tuple[str, str]]:
        """获取待处理的特征压缩文件（.zarr）路径列表"""
        matched_files = []
        skipped = 0
        feature_dir = os.path.join(self.config['paths']['feature_data'])
        output_dir = os.path.join(self.config['paths']['sequence_data'])

        if self.mode == "manual":
            # 手动模式：只处理用户选中的文件，同时跳过已处理
            for input_path in self.selected_files:
                if input_path in self.processed_files:
                    skipped += 1
                    continue
                filename = os.path.basename(input_path)
                output_prefix = os.path.splitext(filename.replace("feature_", "sequence_"))[0]
                output_path = os.path.join(output_dir, output_prefix)
                matched_files.append((input_path, output_path))

            self.logger.info(
                f"手动模式：跳过 {skipped} 个已处理文件，"
                f"{len(matched_files)} 个新文件需要处理"
            )
            return matched_files

        for f in os.listdir(feature_dir):
            if f.startswith("feature_") and f.endswith(".zarr"):
                input_path = os.path.abspath(os.path.join(feature_dir, f))
                if input_path in self.processed_files:
                    skipped += 1
                    continue

                output_prefix = os.path.splitext(f.replace("feature_", "sequence_"))[0]
                output_path = os.path.join(output_dir, output_prefix)
                matched_files.append((input_path, output_path))
        self.logger.info(            
            f"自动模式：跳过 {skipped} 个已处理文件，"
            f"{len(matched_files)} 个新文件需要处理"
        )        
        return matched_files

    def _process_batch_gpu(self, features, y, sequence_length):
        if not self.gpu_enabled:
            self.logger.error("GPU未启用，无法进行GPU处理。")
            return None, None

        with self.gpu_stream:
            try:
                features_gpu = cp.asarray(features, dtype=self.config['project']['dtype'])
                num_windows = len(features) - sequence_length + 1

                if num_windows <= 0:
                    self.logger.warning(f"样本数不足以生成序列窗口：{len(features)} - {sequence_length} + 1 <= 0")
                    return None, None

                windows_gpu = cp.lib.stride_tricks.as_strided(
                    features_gpu,
                    shape=(num_windows, sequence_length, features_gpu.shape[1]),
                    strides=(features_gpu.strides[0], features_gpu.strides[0], features_gpu.strides[1])
                )

                labels_gpu = cp.asarray(y[sequence_length - 1:sequence_length - 1 + num_windows]).reshape(-1, 1)
                window_ids = cp.arange(num_windows).reshape(-1, 1)
                labels_with_id = cp.concatenate([labels_gpu, window_ids], axis=1)

                self.logger.info(f"成功处理批次: {num_windows} 窗口数据, 特征形状: {windows_gpu.shape}, 标签形状: {labels_with_id.shape}")
                return cp.asnumpy(windows_gpu), cp.asnumpy(labels_with_id)

            except Exception as e:
                self.logger.error(f"GPU处理批次时发生错误: {str(e)}", exc_info=True)
                return None, None
            finally:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Device(0).synchronize()

    def save_zarr_file(self,output_path, X_pca, labels):
        chunksize = self.config["feature"].get(" zarr_chunk_size", 200000)
        compressor = Blosc(cname='zstd', clevel=5, shuffle=0)
        z = zarr.open_group(output_path, mode='w')
        z.create_dataset("X_pca", data=X_pca, chunks=(chunksize, X_pca.shape[1]), compressor=compressor)
        z.create_dataset("labels", data=labels, chunks=(chunksize,), compressor=compressor)

    def process_single_zarr(self, input_path, output_prefix, record_lock=None):
        """处理单个 .zarr 文件为多个序列批次，可在线程中安全调用"""
        sequence_length = self.config['lstm_params']['data']['sequence_length']
        dtype = self.config['project']['dtype']

        z = zarr.open_group(input_path, mode='r')
        X_pca = z["X_pca"][:]
        y = z["labels"][:]

        mask = np.isnan(X_pca)
        if np.all(mask):
                self.logger.warning(f"{input_path} 全为NaN，跳过。")

        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        X_pca[mask] = np.interp(idx[mask], idx[~mask], X_pca[~mask])

        total_samples = len(X_pca)
        i = 0
        batch_counter = 0
        last_output_info = None

        while i < total_samples:
            chunk_size = self.chunk_sizer.estimate_chunksize()
            self.logger.info(f"当前估计的块大小: {chunk_size}")

            start = i
            end = min(start + chunk_size + sequence_length - 1, total_samples)
            if end - start < sequence_length:
                self.logger.warning(f"{input_path}: 跳过窗口 start={start}, end={end}")
                i += chunk_size
                continue

            batch_features = X_pca[start:end]
            batch_y = y[start:end]
            reshaped_features, reshaped_labels = self._process_batch_gpu(batch_features, batch_y, sequence_length)

            if reshaped_features is None or reshaped_features.shape[0] == 0:
                self.logger.warning(f"{input_path}: 跳过空批次 start={start}")
                i += chunk_size

            output_path = f"{output_prefix}_batch{batch_counter + 1}.zarr"
            self.save_zarr_file(output_path, reshaped_features, reshaped_labels)

            self.logger.info(f"处理完成: 保存至 {output_path} shape={reshaped_features.shape}")

            last_output_info = (output_path, reshaped_features.shape)
            i += chunk_size
            batch_counter += 1

        if last_output_info:
                self.logger.info(f"✅ 完成: {input_path}")
                self._save_processed_file(input_path)
        else:
                self.logger.warning(f"⚠️ 未生成有效输出: {input_path}")

    def split_zarr_to_train_val(self, input_dir: str, output_dir: str, test_size: float = 0.2,
                                    buffer_size: int = 40960, seed: int = 42, mem_threshold: float = 0.8,
                                    min_class_samples: int = 100, target_min_train: int = 50000, block_len: int = 200):

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m%d%H%M")
        train_path = os.path.join(output_dir, f"trainset_{timestamp}.zarr")
        val_path = os.path.join(output_dir, f"valset_{timestamp}.zarr")

        train_z = zarr.open_group(train_path, mode='w')
        val_z = zarr.open_group(val_path, mode='w')

        X_ds = None
        y_ds = None
        train_ptr = 0
        train_buf_X, train_buf_y = [], []

        val_X_ds = None
        val_y_ds = None
        val_ptr = 0
        val_buf_X, val_buf_y = [], []

        np.random.seed(seed)
        all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".zarr") and f.startswith("sequence_")])
        compressor = Blosc(cname='zstd', clevel=5, shuffle=0)
        self.logger.info(f"📁 准备处理 {len(all_files)} 个输入文件...")
        def memory_ok():
            used = psutil.Process().memory_info().rss / (1024**3)
            total = psutil.virtual_memory().total / (1024**3)
            return used / total < mem_threshold

        def flush_train():
            nonlocal X_ds, y_ds, train_ptr
            if not train_buf_X:
                return
            X = np.stack(train_buf_X)
            y = np.array(train_buf_y)
            if X_ds is None:
                shape = X.shape[1:]
                X_ds = train_z.create_dataset("features", shape=(0, *shape), chunks=(buffer_size, *shape), dtype='float32', compressor=compressor)
                y_ds = train_z.create_dataset("labels", shape=(0,), chunks=(buffer_size,), dtype='int32', compressor=compressor)
            new_size = train_ptr + len(X)
            X_ds.resize((new_size, *X.shape[1:]))
            y_ds.resize((new_size,))
            X_ds[train_ptr:new_size] = X
            y_ds[train_ptr:new_size] = y
            self.logger.info(f"📤 flush_train: 写入 {len(X)} 条，总计 {new_size} 条")
            train_ptr = new_size
            train_buf_X.clear()
            train_buf_y.clear()

        def flush_val():
            nonlocal val_X_ds, val_y_ds, val_ptr
            if not val_buf_X:
                return
            X = np.stack(val_buf_X)
            y = np.array(val_buf_y)
            if val_X_ds is None:
                shape = X.shape[1:]
                val_X_ds = val_z.create_dataset("features", shape=(0, *shape), chunks=(buffer_size, *shape), dtype='float32', compressor=compressor)
                val_y_ds = val_z.create_dataset("labels", shape=(0,), chunks=(buffer_size,), dtype='int32', compressor=compressor)
            new_size = val_ptr + len(X)
            val_X_ds.resize((new_size, *X.shape[1:]))
            val_y_ds.resize((new_size,))
            val_X_ds[val_ptr:new_size] = X
            val_y_ds[val_ptr:new_size] = y
            self.logger.info(f"📤 flush_val: 写入 {len(X)} 条，总计 {new_size} 条")
            val_ptr = new_size
            val_buf_X.clear()
            val_buf_y.clear()

        for fname in all_files:
            path = os.path.join(input_dir, fname)
            try:
                z = zarr.open_group(path, mode='r')
                X = z['X_pca'][:]
                y = z['labels'][:]
                if y.ndim > 1:
                    y = y[:, 0]
                for i in range(len(X)):
                    train_buf_X.append(X[i])
                    train_buf_y.append(y[i])
                    if len(train_buf_X) >= buffer_size or not memory_ok():
                        flush_train()
                self.logger.info(f"✔️ 读取 {fname}：共 {len(X)} 条样本")
            except Exception as e:
                self.logger.warning(f"❌ 跳过 {fname}: {e}")

        flush_train()

        if train_ptr == 0:
            self.logger.warning("❌ 无训练数据，跳过划分")
            return

        val_size = int(train_ptr * test_size)
        val_start = train_ptr - val_size
        self.logger.info(f"📦 开始构建验证集：准备写入 {val_size} 条")
        for i in range(val_start, train_ptr, buffer_size):
            chunk_end = min(i + buffer_size, train_ptr)
            X_chunk = X_ds[i:chunk_end]
            y_chunk = y_ds[i:chunk_end]
            val_buf_X.extend(X_chunk)
            val_buf_y.extend(y_chunk)
            flush_val()
        flush_val()

        y_all = y_ds[:val_start]  # 只从训练部分看分布
        missing_labels = []
        unique_all = set(np.unique(y_all))
        unique_val = set(np.unique(val_y_ds[:])) if val_y_ds is not None else set()

        for cls in unique_all:
            if cls not in unique_val:
                self.logger.info(f"➕ 补充验证类别 {cls}")
                cls_indices = np.where(y_all == cls)[0]
                supplement_count = max(min_class_samples, int(val_size * 0.005))
                if supplement_count > 0:
                    indices = np.random.choice(cls_indices, supplement_count, replace=False)
                    val_buf_X.extend(X_ds[indices])
                    val_buf_y.extend(y_ds[indices])
                    flush_val()
        flush_val()

        y_all = y_ds[:train_ptr]
        self.logger.info("🚀 开始增强小类别样本...")
        for cls in np.unique(y_all):
            cls_indices = np.where(y_all == cls)[0]
            cls_indices = cls_indices[cls_indices < val_start]  # 避免训练增强与验证重叠
            shortfall = target_min_train - len(cls_indices)
            if shortfall > 0:
                n_blocks = shortfall // block_len
                blocks_added = 0
                for _ in range(n_blocks):
                    if len(cls_indices) < block_len:
                        break
                    start_idx = np.random.choice(cls_indices)
                    start = max(0, start_idx - block_len // 2)
                    end = min(train_ptr, start + block_len)
                    if end - start >= block_len:
                        if not memory_ok():
                            import gc; gc.collect()
                        X_block = X_ds[start:end]
                        y_block = y_ds[start:end]
                        X_ds.resize((train_ptr + block_len, *X_ds.shape[1:]))
                        y_ds.resize((train_ptr + block_len,))
                        X_ds[train_ptr:train_ptr+block_len] = X_block
                        y_ds[train_ptr:train_ptr+block_len] = y_block
                        train_ptr += block_len
                        blocks_added += 1
                if blocks_added:
                    self.logger.info(f"🔁 增强类别 {cls}：添加 {blocks_added} 个时间片段")

        train_dist = dict(Counter(y_ds[:train_ptr]))
        val_dist = dict(Counter(val_y_ds[:])) if val_y_ds else {}
        self.logger.info(f"📊 训练集类别分布: {train_dist}")
        self.logger.info(f"📊 验证集类别分布: {val_dist}")

        self.logger.info(f"✅ 完成划分：训练集 {train_ptr - val_ptr}，验证集 {val_ptr}")
        if missing_labels:
            self.logger.info(f"⚠️ 补充的验证类别: {missing_labels}")
        self.logger.info(f"输出：\n - {train_path}\n - {val_path}")


    def run_multi_thread_processing(self, input_paths,use_parallel=False):
        """多线程并发处理多个 .zarr 文件"""
        if use_parallel:
            max_workers = self.config.get("feature", {}).get("parallel_workers", 4)
            self.logger.info(f"并行处理模式：使用 {max_workers} 个工作线程")
        else:
            max_workers = 1

        record_lock = threading.Lock()
        input_paths = self.get_feature_zarr()
        def thread_safe_wrapper(input_path, output_prefix):
            try:
                self.process_single_zarr(input_path, output_prefix, record_lock)
            except Exception as e:
                with record_lock:
                    self.logger.exception(f"线程任务异常: {input_path} | {type(e).__name__}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(thread_safe_wrapper, input_path, output_prefix): input_path
                for input_path, output_prefix in input_paths
            }
            for future in as_completed(futures):
                _ = future.result()  # 所有异常都已在 wrapper 中处理
    
    def run_build_and_merge(self, use_parallel=False):
        """顺序执行构建与合并过程（线程安全）"""
        self.use_parallel = use_parallel
        self.logger.info("🧱 开始批量构建滑动窗口...")
        self.run_multi_thread_processing(self.use_parallel)

        self.logger.info("📦 开始合并构建好的 .zarr 文件为训练集...")
        self.split_zarr_to_train_val(
            input_dir=self.config['paths']['sequence_data'], 
            output_dir=self.config['paths']['train_set'],
            test_size=self.config['lstm_params']['data']['test_size']
            )
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()



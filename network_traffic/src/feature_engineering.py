import os
import sys
import yaml
import cudf
import uuid
import json
import zarr
import pynvml
import torch
import logging
import numba.cuda
import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Lock
from contextlib import contextmanager
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from imblearn.over_sampling import SMOTE
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_selection import VarianceThreshold
from cuml.preprocessing import StandardScaler as cuStandardScaler, RobustScaler, MinMaxScaler
from cuml.decomposition import PCA  # GPU加速的PCA
from typing import List, Tuple, Dict
from cudf import option_context
from src.chunk_sizer import DynamicChunkSizer

class FeatureEngineer:
    def __init__(self, config_path: str):
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        self.config = self._load_config(config_path)
        self.logger = self.init_logger(self.config)
        self._validate_paths()
        self.mode = "auto"
        self.selected_files = []
        self.chunk_sizer = DynamicChunkSizer(self.config)
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.processed_files = self._load_processed_files()
        self.sample_ratio = self.config.get("feature_engineering", {}).get("sampling", {}).get("sample_ratio", 0.02)
        self.min_sample_size = self.config.get("feature_engineering", {}).get("sampling", {}).get("min_sample_size", 10000)
        self.enable_cache_clear = self.config.get("engine_params", {}).get("enable_cache_clear", True)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件（与清洗代码保持兼容）"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["cleaned_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", config['paths']["cleaned_data"]))
        config["output_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", config['paths']["feature_data"]))
        config["processed_files_log"] = os.path.abspath(
            os.path.join(config['paths']['log_dir'], "feature_engineered_files.log"))

        config["feature"] = config.get("feature", {})
        config["feature"].setdefault("pca_components", config["feature"]["pca"]["pca_components"])
        config["feature"].setdefault("target_column", config["feature"]["pca"]["target_column"])
        
        return config

    def init_logger(self, config, output_widget=None):
        from PyQt5.QtCore import QObject, pyqtSignal

        logger = logging.getLogger("FeatureEngineer")
        logger.setLevel(config['logging']['level'])

        log_file = os.path.join(config['paths']['log_dir'], "feature_engineering.log")
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

    def _validate_paths(self):
        """验证输入输出目录"""
        if not os.path.exists(self.config["cleaned_dir"]):
            raise FileNotFoundError(f"清洗数据目录不存在: {self.config['cleaned_dir']}")
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _load_processed_files(self) -> set:
        """加载已处理文件记录"""
        if os.path.exists(self.config["processed_files_log"]):
            with open(self.config["processed_files_log"], "r") as f:
                return set(f.read().splitlines())
        return set()

    def _save_processed_file(self, input_path: str):
        """记录已处理文件"""
        with open(self.config["processed_files_log"], "a") as f:
            f.write(f"{input_path}\n")
        self.processed_files.add(input_path)

    def _get_cleaned_files(self) -> List[Tuple[str, str]]:
        """获取待处理的清洗后文件"""
        matched_files = []
        skipped = 0

        if self.mode == "manual":
            for input_path in self.selected_files:
                if input_path in self.processed_files:
                    skipped += 1
                    continue
                output_path = os.path.join(self.config["output_dir"])
                matched_files.append((input_path, output_path))

            self.logger.info(
                f"手动模式：跳过 {skipped} 个已处理文件，"
                f"{len(matched_files)} 个新文件需要处理"
            )
            return matched_files

        for f in os.listdir(self.config["cleaned_dir"]):
            if not (f.startswith("cleaned_") and f.endswith(".parquet")):
                continue
            input_path = os.path.join(self.config["cleaned_dir"], f)
            if input_path in self.processed_files:
                skipped+= 1
                continue
            output_path = os.path.join(self.config["output_dir"])
            matched_files.append((input_path, output_path))
        self.logger.info(            
            f"自动模式：跳过 {skipped} 个已处理文件，"
            f"{len(matched_files)} 个新文件需要处理"
        )
        return matched_files

    def _get_scaler(self, method: str, with_mean: bool, with_std: bool):
        """根据配置返回合适的标准化方法"""
        method = method.lower()
        if method == "standard":
            return cuStandardScaler(with_mean=with_mean, with_std=with_std)
        elif method == "robust":
            return RobustScaler(with_centering=with_mean, with_scaling=with_std)
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"未知的标准化方法: {method}，请使用 'standard'、'minmax' 或 'robust'")

    def safe_clean(self, df):
        for col in df.columns:
            df[col] = df[col].astype(cp.float32)

        df = df.replace([cp.inf, -cp.inf], 0)
        df = df.fillna(0)
        return df

    def _combine_pca_chunks(self, X_pca_list):
        """
        高效合并所有 pca 结果（从 X_pca_list）为一个 cupy 数组
        """
        # 先转换第一个 batch 并获取 dtype
        first_gpu = X_pca_list[0].to_cupy()
        total_rows = sum(x.shape[0] for x in X_pca_list)
        feature_dim = first_gpu.shape[1]

        # ✅ 用第一个 GPU chunk 推断 dtype
        combined = cp.empty((total_rows, feature_dim), dtype=first_gpu.dtype)

        offset = 0
        for x in X_pca_list:
            x_gpu = x.to_cupy() if not isinstance(x, cp.ndarray) else x
            combined[offset:offset + x_gpu.shape[0], :] = x_gpu
            offset += x_gpu.shape[0]

        return combined

    def save_zarr_file(self,output_path, X_pca, labels):
        import zarr
        from numcodecs import Blosc
        chunksize = self.config["feature"].get("zarr_chunk_size", 200000)
        compressor = Blosc(cname='zstd', clevel=5, shuffle=0)
        z = zarr.open_group(output_path, mode='w')
        z.create_dataset("X_pca", data=X_pca, chunks=(chunksize, X_pca.shape[1]), compressor=compressor)
        z.create_dataset("labels", data=labels, chunks=(chunksize,), compressor=compressor)


    def _feature_engineering_single(self, input_path: str, output_dir: str):
        try:
            # 1. 读取数据
            df = cudf.read_parquet(input_path)

            self.logger.info(f"开始处理文件: {os.path.basename(input_path)} | 原始形状: {df.shape}")
            self.safe_clean(df)

            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")

            # 2. 目标列
            target_col = self.config["feature"]["target_column"]
            if target_col not in df.columns:
                raise KeyError(f"目标列 {target_col} 不存在")

            # 3. 初始化工具
            scaler = cuStandardScaler(
                with_mean=self.config["feature"]["scaling"]["with_mean"], 
                with_std=self.config["feature"]["scaling"]["with_std"])
            pca = PCA(
                n_components=self.config["feature"]["pca"]["pca_components"],
                whiten=self.config["feature"]["pca"]["whiten"])
                
            # 4. 采样拟合 PCA
            sample_ratio = self.config["feature"]["pca"].get("sample_ratio", 0.02)
            min_sample_size = self.config["feature"]["pca"].get("min_sample_size", 100000)
            sample_size = int(min(max(len(df) * sample_ratio, min_sample_size), len(df)))
            X_sample = df.sample(n=sample_size).drop(columns=[target_col]).fillna(0)
            self.logger.info(f"采样 {sample_size} 条数据用于PCA拟合")


            scaler.fit(X_sample)
            pca.fit(scaler.transform(X_sample))

            del X_sample
            torch.cuda.empty_cache()

            # 5. 分块处理并存储 PCA 结果
            rows_per_file = min(self.config["feature"]['rows_per_file'], len(df))  
            processing_chunk_size = self.chunk_sizer.estimate_chunksize()
            self.logger.info(f"处理块大小: {processing_chunk_size}")
            file_num_chunks = (len(df) + rows_per_file - 1) // rows_per_file
            pca_components = self.config["feature"]["pca"]["pca_components"]

            for file_idx in range(file_num_chunks):
                file_chunk = df.iloc[file_idx * rows_per_file : (file_idx + 1) * rows_per_file]
                self.logger.info(f"处理中: file_chunk{file_idx+1}/{file_num_chunks} of {os.path.basename(input_path)}")
                pca_results = []
                labels = []

                for chunk_idx in range(0, len(file_chunk), processing_chunk_size):
                    processing_chunk = file_chunk.iloc[chunk_idx : chunk_idx + processing_chunk_size]

                    try:
                        X_chunk = processing_chunk.drop(columns=[target_col]).fillna(0)
                        X_scaled = scaler.transform(X_chunk)

                        # **分批 PCA 计算，减少 OOM**
                        batch_size = 100000  
                        X_pca_list = []
                        
                        for i in range(0, len(X_scaled), batch_size):
                            X_batch = X_scaled[i : i + batch_size]
                            X_pca_batch = pca.transform(X_batch)
                            X_pca_list.append(X_pca_batch)

                            del X_batch, X_pca_batch
                            if self.enable_cache_clear:
                                torch.cuda.empty_cache()

                        if X_pca_list:
                            X_pca = self._combine_pca_chunks(X_pca_list)
                        else:
                            X_pca = cp.empty((0, pca_components))

                        y = processing_chunk[target_col].to_cupy()

                        pca_results.append(X_pca)
                        labels.append(y)

                        del X_chunk, X_scaled, X_pca, y

                    except Exception as chunk_error:
                        self.logger.error(f"子块处理失败: {chunk_error}", exc_info=True)
                        raise

                # **存储 PCA 结果，供 CPU 处理滑动窗口**
                pca_results_np = cp.asnumpy(cp.vstack(pca_results))  # 转换为 NumPy
                labels_np = cp.asnumpy(cp.concatenate(labels))

                f = f"feature_{os.path.basename(input_path).replace("cleaned_", '').replace('.parquet', '')}_pca_{file_idx+1}.zarr"
                output_path = os.path.join(output_dir, f)
                os.makedirs(output_dir, exist_ok=True)
                self.save_zarr_file(output_path, pca_results_np, labels_np)

                self.logger.info(f" ✅ 保存PCA结果到: {output_path} | 形状: {pca_results_np.shape}")

                del pca_results, labels, pca_results_np, labels_np

            self._save_processed_file(input_path)
            

        except Exception as e:
            self.logger.error(f"处理失败: {input_path}, 错误: {str(e)}", exc_info=True)
            raise


    def run_parallel_engineering(self, use_parallel=False):
        """线程安全并发执行特征工程任务"""

        matched_files = self._get_cleaned_files()
        if not matched_files:
            self.logger.warning("未找到需要处理的清洗文件！")
            return

        if use_parallel:
            max_workers = self.config.get("feature", {}).get("parallel_workers", 4)
            self.logger.info(f"并行处理模式：使用 {max_workers} 个工作线程")
        else:
            max_workers = 1

        # ✅ 去重任务防止重复提交
        submitted = set()
        submitted_lock = Lock()

        def task_wrapper(input_path, output_dir):
            with submitted_lock:
                if input_path in submitted:
                    self.logger.debug(f"文件已提交过，跳过重复任务: {input_path}")
                    return
                submitted.add(input_path)
            self._feature_engineering_single(input_path, output_dir)

        self.logger.info("特征工程处理开始！")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with option_context("spill", True):
                futures = []
                for input_path, output_dir in matched_files:
                    futures.append(
                        executor.submit(task_wrapper, input_path, output_dir)
                    )

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"任务失败: {str(e)}")

        self.logger.info("所有特征工程任务已完成！")


if __name__ == "__main__":
    engineer = FeatureEngineer(os.path.join(os.path.dirname(__file__), "../config/patterns.yaml"))
    engineer.run_parallel_engineering()

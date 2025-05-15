import os
import sys
import yaml
import cudf
import torch
import logging
import numba
import traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from cuml.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
from src.chunk_sizer import DynamicChunkSizer

class DataCleaner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self.init_logger(self.config)
        self._validate_paths()
        self.mode = "auto"
        self.selected_files = []
        self.chunk_sizer = DynamicChunkSizer(self.config)
        self.processed_files = self._load_processed_files() 
        self.enable_cache_clear = self.config.get("engine_params", {}).get("enable_cache_clear", True)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["input_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", config['paths']["raw_data"]))
        config["output_dir"] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", config['paths']["cleaned_data"]))
        config["cleaned_files_log"] = os.path.abspath(
            os.path.join(config['paths']['log_dir'], "data_cleaned_files.log"))
        return config

    def init_logger(self, config, output_widget=None):
        from PyQt5.QtCore import QObject, pyqtSignal

        logger = logging.getLogger("DataCleaner")
        logger.setLevel(config['logging']['level'])
        formatter = logging.Formatter(config['logging'].get('format', "%(asctime)s - %(levelname)s - %(message)s"))
        log_file = os.path.join(config['paths']['log_dir'], "data_cleaning.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)

        # ✅ 如果没有任何Handler，先添加基本的文件+终端
        if not logger.handlers:
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
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

    def _validate_paths(self):
        """验证输入输出目录"""
        if not os.path.exists(self.config["input_dir"]):
            raise FileNotFoundError(f"输入目录不存在: {self.config['input_dir']}")
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _load_processed_files(self) -> set:
        """加载已处理文件记录"""
        if os.path.exists(self.config["cleaned_files_log"]):
            with open(self.config["cleaned_files_log"], "r") as f:
                return set(f.read().splitlines())
        return set()

    def _save_processed_file(self, input_path: str):
        """记录已处理文件"""
        with open(self.config["cleaned_files_log"], "a") as f:
            input_path = os.path.abspath(input_path)
            f.write(f"{input_path}\n")
        self.processed_files.add(input_path)

    def _get_all_csv_files(self) -> List[Tuple[str, str]]:
        """根据模式获取所有 (输入路径, 输出路径) 对，手动/自动模式均跳过已处理文件"""
        matched_files = []
        skipped = 0

        if self.mode == "manual":
            # 手动模式：只处理用户选中的文件，同时跳过已处理
            for input_path in self.selected_files:
                if input_path in self.processed_files:
                    skipped += 1
                    continue
                fname = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(
                    self.config["output_dir"],
                    f"cleaned_{fname}.parquet"
                )
                matched_files.append((input_path, output_path))

            self.logger.info(
                f"手动模式：跳过 {skipped} 个已处理文件，"
                f"{len(matched_files)} 个新文件需要处理"
            )
            return matched_files

        # 自动模式：扫描整个输入目录，跳过已处理
        for f in os.listdir(self.config["input_dir"]):
            if not f.endswith(".csv"):
                continue
            input_path = os.path.abspath(os.path.join(self.config["input_dir"], f))
            if input_path in self.processed_files:
                skipped += 1
                continue
            fname = os.path.splitext(f)[0]
            output_path = os.path.join(
                self.config["output_dir"],
                f"cleaned_{fname}.parquet"
            )
            matched_files.append((input_path, output_path))

        self.logger.info(
            f"自动模式：跳过 {skipped} 个已处理文件，"
            f"{len(matched_files)} 个新文件需要处理"
        )
        return matched_files


    def _drop_high_missing_cols(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """删除高缺失率列"""
        threshold_ratio = self.config["cleaning"].get("missing_threshold", 0.7)
        missing_thresh = int(threshold_ratio * len(df))
        return df.dropna(axis=1, thresh=missing_thresh)

    def _update_label_map(self, new_labels: list):
        """动态更新 label_map，只更新原有结构"""

        # 获取配置文件中的 label_map
        label_map = self.config.get("label_map", {})
        if label_map is None:
            label_map = {}

        for label in new_labels:
            if label not in label_map:
                next_index = len(label_map)  
                label_map[label] = next_index

        config_path = self.config['paths']['config_path']

        with open(config_path, "r", encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        config_data['label_map'] = label_map

        with open(config_path, "w", encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)

        return label_map


    def _encode_categorical(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """分类特征编码（避免空值干扰）"""
        label_map = self.config.get('label_map', {})
        if label_map is None:
            label_map = {}
        self.logger.info(f"当前标签映射: {label_map}")
        
        new_labels = []  
        for col in df.select_dtypes(include=['object']).columns:
            if col == "label":
                df[col] = df[col].astype(str).str.strip()
                unmapped_vals = df[col][~df[col].isin(label_map)]
                unmapped_vals = unmapped_vals.to_pandas()
                if not unmapped_vals.empty:
                    self.logger.info(f"[警告] 未映射标签值: {unmapped_vals.unique()}")
                    new_labels.extend(unmapped_vals.unique())

                label_map = self._update_label_map(new_labels)

                mapped = df[col].map(label_map).fillna(self.config.get("unknown_label", -1)).astype("int32")
                df[col] = mapped
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def _handle_missing_and_nonfinite(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """统一处理缺失值、非有限值、时间戳转换等问题"""
        cfg = self.config.get("cleaning", {})
        num_strategy = cfg.get("fill_numeric", "median")
        cat_strategy = cfg.get("fill_categorical", "mode")
        # 替换 Inf/-Inf 为 NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        if 'timestamp' in df.columns:
            cfg = self.config.get("cleaning", {})
            default_ts_str = cfg.get("default_timestamp", "2018-01-01")
            default_ts = pd.Timestamp(default_ts_str)

            try:
                ts_converted = cudf.to_datetime(df['timestamp'], format="%d/%m/%Y %I:%M:%S %p", dayfirst=True, errors='coerce')
                if ts_converted.isnull().all():
                    raise ValueError("全部转换失败")
                ts_converted = ts_converted.fillna(default_ts)
                df['timestamp'] = (ts_converted.astype('int64') // 10**9).astype(self.config['project']['dtype'])
            except Exception as e:
                self.logger.warning(f"[时间戳修复] 全部转换失败，将使用默认值，原因: {e}")
                df['timestamp'] = cudf.Series([default_ts.timestamp()] * len(df), dtype=self.config['project']['dtype'])


        # 数值列填充
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        if len(numeric_cols) > 0:
            if num_strategy == "median":
                fill_values = df[numeric_cols].median()
            elif num_strategy == "mean":
                fill_values = df[numeric_cols].mean()
            elif num_strategy == "zero":
                fill_values = {col: 0 for col in numeric_cols}
            else:
                fill_values = {col: 0 for col in numeric_cols}  # fallback
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)

        # 类别列填充
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                if cat_strategy == "mode":
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                elif cat_strategy == "constant":
                    mode_val = cfg.get("constant_categorical_value", "UNKNOWN")
                else:
                    mode_val = "UNKNOWN"
                df[col] = df[col].fillna(mode_val)


        # 检查仍有 NaN 的列
        if df.isnull().any().any() and cfg.get("enable_ffill", True):
            bad_cols = df.columns[df.isnull().any()].tolist()
            self.logger.warning(f"⚠️ 以下列仍有缺失值，尝试使用前向填充：{bad_cols}")
            df = df.fillna(method='ffill')

        return df

    def _clean_single_file(self, input_path: str, output_path: str):
        """单个文件清洗流程"""
        try:
            df = cudf.read_csv(input_path)
            self.logger.info(f"正在处理: {os.path.basename(input_path)}")
            self.logger.info(f"原始数据形状: {df.shape}")

            df.columns = df.columns.str.strip().str.replace('[^a-zA-Z0-9]', '_', regex=True).str.lower()
            
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            df = self._drop_high_missing_cols(df)
            df = self._handle_missing_and_nonfinite(df)
            df = self._encode_categorical(df)

            chunk_size = self.chunk_sizer.estimate_chunksize()
            self.logger.info(f"动态分块大小: {chunk_size}")
            temp_files = []
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            self.logger.info(f"处理中：{os.path.basename(input_path)}")

            for i, _ in enumerate(tqdm(range(num_chunks))):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(df))
                
                chunk = df[start:end]

                temp_file = f"{output_path}_part_{i}.parquet"
                temp_files.append(temp_file)
                chunk.to_parquet(temp_file, index=False)
                if self.enable_cache_clear:
                    torch.cuda.empty_cache()

            tables = [pq.read_table(f) for f in temp_files]
            combined_table = pa.concat_tables(tables)
            pq.write_table(combined_table, output_path)

            for temp_file in temp_files:
                os.remove(temp_file)

            self.logger.info(f" ✅ 数据清理处理成功! 文件保存至: {output_path}")
            self._save_processed_file(input_path)

        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            raise



    def run_parallel_cleaning(self, use_parallel=False):
        matched_files = self._get_all_csv_files()
        if not matched_files:
            self.logger.warning("未找到 CSV 文件")
            return

        max_workers = 1
        if use_parallel:
            max_workers = self.config.get("cleaning", {}).get("parallel_workers", 4)

        lock = Lock()
        active_files = set()

        def safe_worker(inp, out):
            with lock:
                if inp in active_files:
                    return  # 已处理
                active_files.add(inp)
            try:
                self._clean_single_file(inp, out)
            except Exception:
                self.logger.error(f"[异常] {inp}\n" + traceback.format_exc())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(safe_worker, inp, out)
                for inp, out in matched_files
            ]
            for f in as_completed(futures):
                pass  # 可加进度计数


if __name__ == "__main__":
    cleaner = DataCleaner(os.path.join(os.path.dirname(__file__), "../config/patterns.yaml"))
    cleaner.run_parallel_cleaning()

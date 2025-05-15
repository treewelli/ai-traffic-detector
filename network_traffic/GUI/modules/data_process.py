import os
import re
import sys
import yaml
import shutil
import logging
import traceback
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QDialog,
    QComboBox, QGroupBox, QCheckBox, QMessageBox, QListWidget, QPlainTextEdit,
    QListWidgetItem, QSizePolicy, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QMetaObject, Q_ARG, QTimer
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.trainset_building import TrainsetBuilder

class CustomMessageBox(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setModal(True)
        self.setMinimumWidth(420)

        self.setWindowTitle(title)

        layout = QVBoxLayout(self)

        # 显示标题
        self.title_label = QLabel(title)
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # 显示正文
        self.label = QLabel(message)
        self.label.setObjectName("message")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.label)

        # 按钮区域
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.setFixedSize(100, 32)
        ok_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

def show_custom_message(parent, title, text):
    dlg = CustomMessageBox(title, text, parent.window() if parent else None)
    dlg.exec_()

class ConsoleStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.text_written.emit(str(text))

    def flush(self):
        pass

class DataCleaningModule(QWidget):
    def __init__(self, config_path, name="数据清洗"):
        super().__init__()
        self.setObjectName(name)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        raw = self.config.get("paths", {}).get("raw_data")
        self.input_dir = os.path.abspath(raw) if raw else None
        self.file_checkboxes = []
        self.mode = "manual"

        self.init_ui()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _redirect_console(self):
        self.stream = ConsoleStream()
        self.stream.text_written.connect(self.console_output.appendPlainText)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setSpacing(0)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        self.init_run_section()
        self.init_file_section()
        self.init_console_section()
        self.init_timer_section()

        self.refresh_file_list()
        self.refresh_visibility()
        self.refresh_cleaned_list()

    def init_mode(self, layout):
        layout.addWidget(QLabel("运行模式："))

        self.manual_btn = QPushButton("手动模式", objectName="ManualToggle")
        self.auto_btn = QPushButton("自动模式", objectName="AutoToggle")

        for btn, mode in [(self.manual_btn, "manual"), (self.auto_btn, "auto")]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))
            layout.addWidget(btn)

        self.manual_btn.setChecked(True)
        layout.addStretch()

    def switch_mode(self, mode):
        self.mode = mode
        self.manual_btn.setChecked(mode == "manual")
        self.auto_btn.setChecked(mode == "auto")
        self.refresh_visibility()
        text = f"[模式切换] 当前运行模式：{'手动模式' if mode == 'manual' else '自动模式'}"
        self.console_output.appendPlainText(text)

    def refresh_visibility(self):
        self.file_group.setVisible(self.mode == "manual")

    def init_thread_mode(self, layout):
        layout.addWidget(QLabel("线程选择："))
        self.use_parallel = False
        self.single_thread_btn = QPushButton("单线程", objectName="SingleThread")
        self.multi_thread_btn = QPushButton("多线程", objectName="MultiThread")

        for btn, val in [(self.single_thread_btn, False), (self.multi_thread_btn, True)]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, v=val: self.switch_thread_mode(v))
            layout.addWidget(btn)

        self.single_thread_btn.setChecked(True)
        layout.addStretch()

    def switch_thread_mode(self, use_parallel):
        self.use_parallel = use_parallel
        self.single_thread_btn.setChecked(not use_parallel)
        self.multi_thread_btn.setChecked(use_parallel)
        text = f"[线程设置] 当前为：{'多线程模式（并行）' if use_parallel else '单线程模式（串行）'}"
        self.console_output.appendPlainText(text)

    def init_run_section(self):
        mode_row_container = QWidget()
        mode_row_container.setFixedHeight(50)
        mode_row_layout = QHBoxLayout(mode_row_container)
        mode_row_layout.setSpacing(5)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.init_mode(mode_row_layout)
        self.init_thread_mode(mode_row_layout)

        for text, handler in [
            ("添加数据集", self.add_input_file),
            ("清除记录", self.clear_processed_files),
            ("运行", self.run_data_cleaning),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(100, 32)
            if text == "运行":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #77DD77;
                        color: white;
                    }
                    QPushButton:pressed {
                        background-color: green;
                    }
                """)
            btn.clicked.connect(handler)
            mode_row_layout.addWidget(btn)

        self.layout_main.addWidget(mode_row_container)

    def init_file_section(self):
        file_row = QHBoxLayout()
        file_row.setSpacing(2)
        file_row.setContentsMargins(0, 0, 0, 0)

        self.file_group = QGroupBox("手动选择输入文件")
        self.file_group.setObjectName("FileGroup")
        file_layout = QVBoxLayout(self.file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(0)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.NoSelection)
        self.file_list.setStyleSheet("background-color:transparent;font-size:18px;")
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.file_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        file_layout.addWidget(self.file_list)

        self.cleaned_group = QGroupBox("已处理文件")
        self.cleaned_group.setObjectName("FileGroup")
        cleaned_layout = QVBoxLayout(self.cleaned_group)
        cleaned_layout.setContentsMargins(0, 0, 0, 0)
        cleaned_layout.setSpacing(0)

        self.cleaned_list = QListWidget()
        self.cleaned_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cleaned_list.setStyleSheet("background-color:transparent;font-size:18px;")
        cleaned_layout.addWidget(self.cleaned_list)

        file_row.addWidget(self.file_group, 1)
        file_row.addWidget(self.cleaned_group, 1)

        self.select_all_checkbox = None

        file_container = QWidget()
        file_container.setLayout(file_row)
        self.layout_main.addWidget(file_container)

    def init_console_section(self):
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.layout_main.addWidget(QLabel("控制台输出："))
        self.layout_main.addWidget(self.console_output)

    def init_timer_section(self):
        self.file_refresh_timer = QTimer(self)
        self.file_refresh_timer.timeout.connect(self.refresh_file_list)
        self.file_refresh_timer.start(30000)

    def toggle_all_checkboxes(self, state):
        checked = state == Qt.Checked
        for _, checkbox in self.file_checkboxes:
            checkbox.setChecked(checked)

    def load_processed_file_names(self):
        log_path = os.path.join(self.config["paths"]["log_dir"], "data_cleaned_files.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                return set(os.path.basename(line.strip()) for line in f.readlines())
        return set()

    def refresh_file_list(self):
        checked_files = {
            cb.text() for _, cb in self.file_checkboxes if cb.isChecked()
        }
        self.file_list.clear()
        self.file_checkboxes.clear()

        if self.select_all_checkbox:
            for i in range(self.file_list.count()):
                widget = self.file_list.itemWidget(self.file_list.item(i))
                if widget == self.select_all_checkbox:
                    self.file_list.takeItem(i)
                    break
            self.select_all_checkbox = None

        if not self.input_dir or not os.path.isdir(self.input_dir):
            self.select_all_checkbox = None  # 清空引用
            self.file_list.addItem("⚠️ 无法读取输入文件夹，请检查配置")
            return

        processed_files = self.load_processed_file_names()
        files = [f for f in os.listdir(self.input_dir)
                if f.endswith(".csv") and f not in processed_files]

        if not files:
            self.select_all_checkbox = None
            self.file_list.addItem("📂 未发现未处理的 csv 文件")
            return

        # === 添加全选复选框 ===
        self.select_all_checkbox = QCheckBox("全选")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_checkboxes)

        header_item = QListWidgetItem()
        header_item.setFlags(Qt.ItemIsEnabled)
        header_item.setSizeHint(self.select_all_checkbox.sizeHint())
        self.file_list.addItem(header_item)
        self.file_list.setItemWidget(header_item, self.select_all_checkbox)

        # === 添加文件复选框 ===
        for name in sorted(files):
            item = QListWidgetItem()
            checkbox = QCheckBox(name)
            checkbox.setMinimumWidth(400)
            item.setSizeHint(checkbox.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, checkbox)
            self.file_checkboxes.append((item, checkbox))
            checkbox.setChecked(name in checked_files)

    def refresh_cleaned_list(self):
        self.cleaned_list.clear()
        processed = self.load_processed_file_names()
        for fname in sorted(processed):
            self.cleaned_list.addItem(fname)

    def clear_processed_files(self):
        log_file = os.path.join(self.config["paths"]["log_dir"], "data_cleaned_files.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        out_dir = os.path.abspath(self.config["paths"]["cleaned_data"])
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith(".parquet") and f.startswith("cleaned_"):
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except:
                        pass

        self.refresh_file_list()
        self.refresh_cleaned_list()

    def add_input_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "添加CSV文件到输入目录", "", "CSV 文件 (*.csv)")
        if files and os.path.isdir(self.input_dir):
            for f in files:
                shutil.copy(f, os.path.join(self.input_dir, os.path.basename(f)))
            self.refresh_file_list()

    def run_data_cleaning(self):
        selected = [
            os.path.join(self.input_dir, cb.text())
            for (_, cb) in self.file_checkboxes if cb.isChecked() and cb.text() != "全选"
        ]
        if self.mode == "manual" and not selected:
            show_custom_message(self, "未选择文件", "请至少选择一个 CSV 文件进行手动处理。")
            return

        self.console_output.clear()

        cleaner = DataCleaner(self.config_path)
        cleaner.init_logger(self.config, output_widget=self.console_output)
        cleaner.mode = self.mode
        if cleaner.mode == "manual":
            cleaner.selected_files = selected
        use_parallel = self.use_parallel
        self.worker = self.CleaningWorker(cleaner, use_parallel)
        self.worker.finished.connect(self.on_cleaning_done)
        self.worker.error.connect(self.on_cleaning_error)
        self.worker.start()

    def on_cleaning_done(self):
        self.console_output.appendPlainText("[任务完成]")
        self.refresh_file_list()
        self.refresh_cleaned_list()

    def on_cleaning_error(self, err):
        self.console_output.appendPlainText("[错误] 清洗过程出现异常")
        self.console_output.appendPlainText(err)
        show_custom_message(self, "异常", "清洗任务失败，详情见控制台")

    class CleaningWorker(QThread):
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, cleaner, use_parallel=False):
            super().__init__()
            self.cleaner = cleaner
            self.use_parallel = use_parallel
        def run(self):
            try:
                self.cleaner.run_parallel_cleaning(self.use_parallel)
            except Exception:
                err = traceback.format_exc()
                self.error.emit(err)
            else:
                self.finished.emit()


class FeatureEngineeringModule(QWidget):
    def __init__(self, config_path, name="特征工程"):
        super().__init__()
        self.setObjectName(name)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        raw = self.config['paths']['cleaned_data']
        self.input_dir = os.path.abspath(raw) if raw else None
        self.file_checkboxes = []
        self.mode = "manual"

        self.init_ui()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setSpacing(0)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        self.init_run_section()
        self.init_file_section()
        self.init_console_section()
        self.init_timer_section()

        self.refresh_file_list()
        self.refresh_visibility()
        self.refresh_processed_list()

    def init_mode(self, layout):
        layout.addWidget(QLabel("运行模式："))

        self.manual_btn = QPushButton("手动模式", objectName="ManualToggle")
        self.auto_btn = QPushButton("自动模式", objectName="AutoToggle")

        for btn, mode in [(self.manual_btn, "manual"), (self.auto_btn, "auto")]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))
            layout.addWidget(btn)

        self.manual_btn.setChecked(True)
        layout.addStretch()

    def switch_mode(self, mode):
        self.mode = mode
        self.manual_btn.setChecked(mode == "manual")
        self.auto_btn.setChecked(mode == "auto")
        self.refresh_visibility()
        text = f"[模式切换] 当前运行模式：{'手动模式' if mode == 'manual' else '自动模式'}"
        self.console_output.appendPlainText(text)

    def refresh_visibility(self):
        self.file_group.setVisible(self.mode == "manual")

    def init_thread_mode(self, layout):
        layout.addWidget(QLabel("线程选择："))
        self.use_parallel = False
        self.single_thread_btn = QPushButton("单线程", objectName="SingleThread")
        self.multi_thread_btn = QPushButton("多线程", objectName="MultiThread")

        for btn, val in [(self.single_thread_btn, False), (self.multi_thread_btn, True)]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, v=val: self.switch_thread_mode(v))
            layout.addWidget(btn)

        self.single_thread_btn.setChecked(True)
        layout.addStretch()

    def switch_thread_mode(self, use_parallel):
        self.use_parallel = use_parallel
        self.single_thread_btn.setChecked(not use_parallel)
        self.multi_thread_btn.setChecked(use_parallel)
        text = f"[线程设置] 当前为：{'多线程模式（并行）' if use_parallel else '单线程模式（串行）'}"
        self.console_output.appendPlainText(text)

    def init_run_section(self):
        mode_row_container = QWidget()
        mode_row_container.setFixedHeight(50)
        mode_row_layout = QHBoxLayout(mode_row_container)
        mode_row_layout.setSpacing(5)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.init_mode(mode_row_layout)
        self.init_thread_mode(mode_row_layout)

        for text, handler in [
            ("清除记录", self.clear_processed_files),
            ("运行", self.run_feature_engineering),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(100, 32)
            if text == "运行":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #77DD77;
                        color: white;
                    }
                    QPushButton:pressed {
                        background-color: green;
                    }
                """)
            btn.clicked.connect(handler)
            mode_row_layout.addWidget(btn)

        self.layout_main.addWidget(mode_row_container)

    def init_file_section(self):
        file_row = QHBoxLayout()
        file_row.setSpacing(2)
        file_row.setContentsMargins(0, 0, 0, 0)

        self.file_group = QGroupBox("手动选择输入文件")
        self.file_group.setObjectName("FileGroup")
        file_layout = QVBoxLayout(self.file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(0)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.NoSelection)
        self.file_list.setStyleSheet("background-color:transparent;font-size:18px;")
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.file_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        file_layout.addWidget(self.file_list)

        self.cleaned_group = QGroupBox("已处理文件")
        self.cleaned_group.setObjectName("FileGroup")
        cleaned_layout = QVBoxLayout(self.cleaned_group)
        cleaned_layout.setContentsMargins(0, 0, 0, 0)
        cleaned_layout.setSpacing(0)

        self.cleaned_list = QListWidget()
        self.cleaned_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cleaned_list.setStyleSheet("background-color:transparent;font-size:18px;")
        cleaned_layout.addWidget(self.cleaned_list)

        file_row.addWidget(self.file_group, 1)
        file_row.addWidget(self.cleaned_group, 1)

        self.select_all_checkbox = None

        file_container = QWidget()
        file_container.setLayout(file_row)
        self.layout_main.addWidget(file_container)

    def init_console_section(self):
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.layout_main.addWidget(QLabel("控制台输出："))
        self.layout_main.addWidget(self.console_output)

    def init_timer_section(self):
        self.file_refresh_timer = QTimer(self)
        self.file_refresh_timer.timeout.connect(self.refresh_file_list)
        self.file_refresh_timer.start(30000)

    def toggle_all_checkboxes(self, state):
        checked = state == Qt.Checked
        for _, checkbox in self.file_checkboxes:
            checkbox.setChecked(checked)

    def load_processed_file_names(self):
        log_path = os.path.join(self.config["paths"]["log_dir"], "feature_engineered_files.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                return set(os.path.basename(line.strip()) for line in f.readlines())
        return set()

    def refresh_file_list(self):
        checked_files = {
            cb.text() for _, cb in self.file_checkboxes if cb.isChecked()
        }
        self.file_list.clear()
        self.file_checkboxes.clear()

        if self.select_all_checkbox:
            for i in range(self.file_list.count()):
                widget = self.file_list.itemWidget(self.file_list.item(i))
                if widget == self.select_all_checkbox:
                    self.file_list.takeItem(i)
                    break
            self.select_all_checkbox = None

        if not self.input_dir or not os.path.isdir(self.input_dir):
            self.select_all_checkbox = None  # 清空引用
            self.file_list.addItem("⚠️ 无法读取输入文件夹，请检查配置")
            return

        processed_files = self.load_processed_file_names()
        files = [f for f in os.listdir(self.input_dir)
                if f.endswith(".parquet") and f not in processed_files]

        if not files:
            self.select_all_checkbox = None
            self.file_list.addItem("📂 未发现未处理的 parquet 文件")
            return

        # === 添加全选复选框 ===
        self.select_all_checkbox = QCheckBox("全选")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_checkboxes)

        header_item = QListWidgetItem()
        header_item.setFlags(Qt.ItemIsEnabled)
        header_item.setSizeHint(self.select_all_checkbox.sizeHint())
        self.file_list.addItem(header_item)
        self.file_list.setItemWidget(header_item, self.select_all_checkbox)

        # === 添加文件复选框 ===
        for name in sorted(files):
            item = QListWidgetItem()
            checkbox = QCheckBox(name)
            checkbox.setMinimumWidth(400)
            item.setSizeHint(checkbox.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, checkbox)
            self.file_checkboxes.append((item, checkbox))
            checkbox.setChecked(name in checked_files)

    def refresh_processed_list(self):
        self.cleaned_list.clear()
        processed = self.load_processed_file_names()
        for fname in sorted(processed):
            self.cleaned_list.addItem(fname)

    def clear_processed_files(self):
        log_file = os.path.join(self.config["paths"]["log_dir"], "feature_engineered_files.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        out_dir = os.path.abspath(self.config["paths"]["feature_data"])
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith(".zarr") and f.startswith("feature_"):
                    try:
                        shutil.rmtree(os.path.join(out_dir, f))
                    except:
                        pass

        self.refresh_file_list()
        self.refresh_processed_list()

    def run_feature_engineering(self):
        selected = [
            os.path.join(self.input_dir, cb.text())
            for (_, cb) in self.file_checkboxes if cb.isChecked() and cb.text() != "全选"
        ]
        if self.mode == "manual" and not selected:
            show_custom_message(self, "未选择文件", "请至少选择一个 Parquet 文件进行手动处理。")
            return

        self.console_output.clear()
        use_parallel= self.use_parallel 
        engineer = FeatureEngineer(self.config_path)
        engineer.init_logger(self.config, output_widget=self.console_output)
        engineer.mode = self.mode
        if engineer.mode == "manual":
            engineer.selected_files = selected

        self.worker = self.FeatureEngineerWorker(engineer, use_parallel)
        self.worker.finished.connect(self.on_featureengineering_done)
        self.worker.error.connect(self.on_featureengineering_error)
        self.worker.start()

    def on_featureengineering_done(self):
        self.console_output.appendPlainText("[任务完成]")
        self.refresh_file_list()
        self.refresh_processed_list()

    def on_featureengineering_error(self, err):
        self.console_output.appendPlainText("[错误] 特征工程过程出现异常")
        self.console_output.appendPlainText(err)
        show_custom_message(self, "异常", "特征工程任务失败，详情见控制台")

    class FeatureEngineerWorker(QThread):
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, engineer, use_parallel=False):
            super().__init__()
            self.engineer = engineer
            self.use_parallel = use_parallel

        def run(self):
            try:
                self.engineer.run_parallel_engineering(self.use_parallel)
            except Exception:
                err = traceback.format_exc()
                self.error.emit(err)
            else:
                self.finished.emit()


class TrainsetBuildingModule(QWidget):
    def __init__(self, config_path, name="创建时序窗口"):
        super().__init__()
        self.setObjectName(name)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        raw = self.config['paths']['feature_data']
        self.input_dir = os.path.abspath(raw) if raw else None
        self.file_checkboxes = []
        self.mode = "manual"

        self.init_ui()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setSpacing(0)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        self.init_run_section()
        self.init_file_section()
        self.init_console_section()
        self.init_timer_section()

        self.refresh_file_list()
        self.refresh_visibility()
        self.refresh_processed_list()

    def init_mode(self, layout):
        layout.addWidget(QLabel("运行模式："))

        self.manual_btn = QPushButton("手动模式", objectName="ManualToggle")
        self.auto_btn = QPushButton("自动模式", objectName="AutoToggle")

        for btn, mode in [(self.manual_btn, "manual"), (self.auto_btn, "auto")]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))
            layout.addWidget(btn)

        self.manual_btn.setChecked(True)
        layout.addStretch()

    def switch_mode(self, mode):
        self.mode = mode
        self.manual_btn.setChecked(mode == "manual")
        self.auto_btn.setChecked(mode == "auto")
        self.refresh_visibility()
        text = f"[模式切换] 当前运行模式：{'手动模式' if mode == 'manual' else '自动模式'}"
        self.console_output.appendPlainText(text)

    def refresh_visibility(self):
        self.file_group.setVisible(self.mode == "manual")

    def init_thread_mode(self, layout):
        layout.addWidget(QLabel("线程选择："))
        self.use_parallel = False
        self.single_thread_btn = QPushButton("单线程", objectName="SingleThread")
        self.multi_thread_btn = QPushButton("多线程", objectName="MultiThread")

        for btn, val in [(self.single_thread_btn, False), (self.multi_thread_btn, True)]:
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda _, v=val: self.switch_thread_mode(v))
            layout.addWidget(btn)

        self.single_thread_btn.setChecked(True)
        layout.addStretch()

    def switch_thread_mode(self, use_parallel):
        self.use_parallel = use_parallel
        self.single_thread_btn.setChecked(not use_parallel)
        self.multi_thread_btn.setChecked(use_parallel)
        text = f"[线程设置] 当前为：{'多线程模式（并行）' if use_parallel else '单线程模式（串行）'}"
        self.console_output.appendPlainText(text)

    def init_run_section(self):
        mode_row_container = QWidget()
        mode_row_container.setFixedHeight(50)
        mode_row_layout = QHBoxLayout(mode_row_container)
        mode_row_layout.setSpacing(5)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.init_mode(mode_row_layout)
        self.init_thread_mode(mode_row_layout)

        for text, handler in [
            ("清除记录", self.clear_processed_files),
            ("运行", self.run_trainset_building),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(100, 32)
            if text == "运行":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #77DD77;
                        color: white;
                    }
                    QPushButton:pressed {
                        background-color: green;
                    }
                """)
            btn.clicked.connect(handler)
            mode_row_layout.addWidget(btn)

        self.layout_main.addWidget(mode_row_container)

    def init_file_section(self):
        file_row = QHBoxLayout()
        file_row.setSpacing(2)
        file_row.setContentsMargins(0, 0, 0, 0)

        self.file_group = QGroupBox("手动选择输入文件")
        self.file_group.setObjectName("FileGroup")
        file_layout = QVBoxLayout(self.file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(0)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.NoSelection)
        self.file_list.setStyleSheet("background-color:transparent;font-size:18px;")
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.file_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        file_layout.addWidget(self.file_list)

        self.cleaned_group = QGroupBox("已处理文件")
        self.cleaned_group.setObjectName("FileGroup")
        cleaned_layout = QVBoxLayout(self.cleaned_group)
        cleaned_layout.setContentsMargins(0, 0, 0, 0)
        cleaned_layout.setSpacing(0)

        self.cleaned_list = QListWidget()
        self.cleaned_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cleaned_list.setStyleSheet("background-color:transparent;font-size:18px;")
        cleaned_layout.addWidget(self.cleaned_list)

        file_row.addWidget(self.file_group, 1)
        file_row.addWidget(self.cleaned_group, 1)

        self.select_all_checkbox = None

        file_container = QWidget()
        file_container.setLayout(file_row)
        self.layout_main.addWidget(file_container)

    def init_console_section(self):
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.layout_main.addWidget(QLabel("控制台输出："))
        self.layout_main.addWidget(self.console_output)

    def init_timer_section(self):
        self.file_refresh_timer = QTimer(self)
        self.file_refresh_timer.timeout.connect(self.refresh_file_list)
        self.file_refresh_timer.start(30000)

    def toggle_all_checkboxes(self, state):
        checked = state == Qt.Checked
        for _, checkbox in self.file_checkboxes:
            checkbox.setChecked(checked)

    def load_processed_file_names(self):
        log_path = os.path.join(self.config["paths"]["log_dir"], "trainset_built_files.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                return set(os.path.basename(line.strip()) for line in f.readlines())
        return set()

    def refresh_file_list(self):
        checked_files = {
            cb.text() for _, cb in self.file_checkboxes if cb.isChecked()
        }
        self.file_list.clear()
        self.file_checkboxes.clear()

        if self.select_all_checkbox:
            for i in range(self.file_list.count()):
                widget = self.file_list.itemWidget(self.file_list.item(i))
                if widget == self.select_all_checkbox:
                    self.file_list.takeItem(i)
                    break
            self.select_all_checkbox = None

        if not self.input_dir or not os.path.isdir(self.input_dir):
            self.select_all_checkbox = None  # 清空引用
            self.file_list.addItem("⚠️ 无法读取输入文件夹，请检查配置")
            return

        processed_files = self.load_processed_file_names()
        files = [f for f in os.listdir(self.input_dir)
                if f.endswith(".zarr") and f not in processed_files]

        if not files:
            self.select_all_checkbox = None
            self.file_list.addItem("📂 未发现未处理的 zarr 文件")
            return

        # === 添加全选复选框 ===
        self.select_all_checkbox = QCheckBox("全选")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_checkboxes)

        header_item = QListWidgetItem()
        header_item.setFlags(Qt.ItemIsEnabled)
        header_item.setSizeHint(self.select_all_checkbox.sizeHint())
        self.file_list.addItem(header_item)
        self.file_list.setItemWidget(header_item, self.select_all_checkbox)

        # === 添加文件复选框 ===
        for name in sorted(files):
            item = QListWidgetItem()
            checkbox = QCheckBox(name)
            checkbox.setMinimumWidth(400)
            item.setSizeHint(checkbox.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, checkbox)
            self.file_checkboxes.append((item, checkbox))
            checkbox.setChecked(name in checked_files)

    def refresh_processed_list(self):
        self.cleaned_list.clear()
        processed = self.load_processed_file_names()
        for fname in sorted(processed):
            self.cleaned_list.addItem(fname)

    def clear_processed_files(self):
        log_file = os.path.join(self.config["paths"]["log_dir"], "trainset_built_files.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        out_dir = os.path.abspath(self.config["paths"]["sequence_data"])
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith(".zarr") and f.startswith("sequence_"):
                    try:
                        shutil.rmtree(os.path.join(out_dir, f))
                    except:
                        pass

        outset_dir = os.path.abspath(self.config["paths"]["train_set"])
        if os.path.isdir(outset_dir):
            for f in os.listdir(outset_dir):
                if f.endswith(".zarr"):
                    try:
                        shutil.rmtree(os.path.join(outset_dir, f))
                    except:
                        pass

        self.refresh_file_list()
        self.refresh_processed_list()

    def run_trainset_building(self):
        selected = [
            os.path.join(self.input_dir, cb.text())
            for (_, cb) in self.file_checkboxes if cb.isChecked() and cb.text() != "全选"
        ]
        if self.mode == "manual" and not selected:
            show_custom_message(self, "未选择文件", "请至少选择一个 zarr 文件进行手动处理。")
            return

        self.console_output.clear()
        
        builder = TrainsetBuilder(self.config_path)
        builder.init_logger(self.config, output_widget=self.console_output)
        builder.mode = self.mode
        if builder.mode == "manual":
            builder.selected_files = selected
        use_parallel = self.use_parallel
        self.worker = self.trainsetBuildingWorker(builder, use_parallel)
        self.worker.finished.connect(self.on_trainsetbuilding_done)
        self.worker.error.connect(self.on_trainsetbuilding_error)
        self.worker.start()

    def on_trainsetbuilding_done(self):
        self.console_output.appendPlainText("[任务完成]")
        self.refresh_file_list()
        self.refresh_processed_list()

    def on_trainsetbuilding_error(self, err):
        self.console_output.appendPlainText("[错误] 时序窗口创建过程出现异常")
        self.console_output.appendPlainText(err)
        show_custom_message(self, "异常", "时序窗口创建任务失败，详情见控制台")

    class trainsetBuildingWorker(QThread):
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, builder, use_parallel=False):
            super().__init__()
            self.builder = builder
            self.use_parallel = use_parallel
        def run(self):
            try:
                self.builder.run_build_and_merge(self.use_parallel)
            except Exception:
                err = traceback.format_exc()
                self.error.emit(err)
            else:
                self.finished.emit()
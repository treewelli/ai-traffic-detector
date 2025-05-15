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
    QListWidgetItem, QSizePolicy
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QThread, QMetaObject, Q_ARG, QTimer

#from src.lstm_train import LSTMTrainer  # 你的实际训练逻辑类

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

class LSTMTrainingModule(QWidget):
    def __init__(self, config_path, name="模型训练"):
        super().__init__()
        self.setObjectName(name)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        raw = self.config['paths']['train_set']
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

    def init_run_section(self):
        mode_row_container = QWidget()
        mode_row_container.setFixedHeight(50)
        mode_row_layout = QHBoxLayout(mode_row_container)
        mode_row_layout.setSpacing(5)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.init_mode(mode_row_layout)

        for text, handler in [
            ("清除记录", self.clear_processed_files),
            ("运行", self.run_lstm_training),
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
        self.file_refresh_timer.start(5000)

    def toggle_all_checkboxes(self, state):
        checked = state == Qt.Checked
        for _, checkbox in self.file_checkboxes:
            checkbox.setChecked(checked)

    def load_processed_file_names(self):
        log_path = os.path.join(self.config["paths"]["log_dir"], "lstm_trained_files.log")
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
                if f.endswith(".zarr") and f.startswith('trainset_') and f not in processed_files]

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
        log_file = os.path.join(self.config["paths"]["log_dir"], "lstm_trained_files.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        out_dir = os.path.abspath(self.config["paths"]["model_best"])
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith(".pt") :
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except:
                        pass

        self.refresh_file_list()
        self.refresh_processed_list()

    def run_lstm_training(self):
        selected = [
            os.path.join(self.input_dir, cb.text())
            for (_, cb) in self.file_checkboxes if cb.isChecked() and cb.text() != "全选"
        ]
        if self.mode == "manual" and not selected:
            show_custom_message(self, "未选择文件", "请至少选择一个 h5 文件进行手动处理。")
            return

        self.console_output.clear()

        # ✅ 把 trainer 构造逻辑延迟到子线程内部
        self.worker = self.LSTMTrainerWorker(
            config_path=self.config_path,
            config=self.config,
            mode=self.mode,
            selected_files=selected,
            output_widget=self.console_output
        )
        self.worker.finished.connect(self.on_lstmtraining_done)
        self.worker.error.connect(self.on_lstmtraining_error)
        self.worker.start()


    def on_lstmtraining_done(self):
        self.console_output.appendPlainText("[任务完成]")
        self.refresh_file_list()
        self.refresh_processed_list()

    def on_lstmtraining_error(self, err):
        self.console_output.appendPlainText("[错误] 模型训练过程出现异常")
        self.console_output.appendPlainText(err)
        show_custom_message(self, "异常", "模型训练任务失败，详情见控制台")

    class LSTMTrainerWorker(QThread):
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, config_path, config, mode, selected_files, output_widget=None):
            super().__init__()
            self.config_path = config_path
            self.config = config
            self.mode = mode
            self.selected_files = selected_files
            self.output_widget = output_widget

        def run(self):
            try:
                from src.lstm_train import LSTMTrainer  # 延迟 import 保证线程内加载
                trainer = LSTMTrainer(self.config_path)
                trainer.init_logger(self.config, output_widget=self.output_widget)
                trainer.mode = self.mode
                if self.mode == "manual":
                    trainer.selected_files = self.selected_files
                trainer.run()  # 真正执行训练
            except Exception:
                err = traceback.format_exc()
                self.error.emit(err)
            else:
                self.finished.emit()

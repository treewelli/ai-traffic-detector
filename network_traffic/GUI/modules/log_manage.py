import os, yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer

class LogViewerModule(QWidget):
    def __init__(self, config_path):
        super().__init__()
        self.config = self.load_config(config_path)
        self.log_dir = os.path.abspath(self.config["paths"].get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_path = None
        self.last_log_files = set()
        self.reverse_map = {}
        self.file_alias_map = {
            "data_cleaning.log": "数据清洗", "data_cleaned_files.log": "数据清洗",
            "feature_engineering.log": "特征工程", "feature_engineered_files.log": "特征工程",
            "lstm_inference.log": "LSTM推理", "sequentia_building.log": "时序窗口构建",
            "sequentia_built_files.log": "时序窗口构建", "lstm_training.log": "LSTM训练",
            "lstm_trained_files.log": "LSTM训练"
        }

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.itemClicked.connect(self.load_log_file)
        main_layout.addWidget(self.file_tree, 2)

        right_layout = QVBoxLayout()
        self.log_content = QTextEdit()
        self.log_content.setReadOnly(True)
        right_layout.addWidget(self.log_content)

        top_right_layout = QHBoxLayout()
        self.tip_label = QLabel("[提示区域]")
        top_right_layout.addWidget(self.tip_label)
        top_right_layout.addStretch()
        clear_btn = QPushButton("清空当前日志")
        clear_btn.clicked.connect(self.clear_log_file)
        top_right_layout.addWidget(clear_btn)
        right_layout.insertLayout(0, top_right_layout)

        clear_all_btn = QPushButton("清除全部日志")
        clear_all_btn.clicked.connect(self.clear_all_logs)
        top_right_layout.addWidget(clear_all_btn)

        main_layout.addLayout(right_layout, 5)
        self.refresh_file_list()

        QTimer(self, timeout=self.auto_refresh_log).start(1000)
        QTimer(self, timeout=self.check_log_file_changes).start(1000)

    def refresh_file_list(self):
        self.file_tree.clear()
        self.reverse_map.clear()

        groups = {
            "engineering": QTreeWidgetItem(self.file_tree),
            "processed": QTreeWidgetItem(self.file_tree)
        }
        groups["engineering"].setText(0, "工程日志")
        groups["processed"].setText(0, "已处理文件日志")

        for file in sorted(os.listdir(self.log_dir)):
            if not file.endswith(".log"): continue
            is_processed = "_files" in file
            group_key = "processed" if is_processed else "engineering"
            alias = self.file_alias_map.get(file, file)

            item = QTreeWidgetItem(groups[group_key])
            item.setText(0, alias)
            item.setData(0, Qt.UserRole, group_key)
            self.reverse_map[(alias, group_key)] = file

        self.file_tree.expandAll()

    def load_log_file(self, item):
        if not item.parent(): return
        alias = item.text(0)
        group_key = item.data(0, Qt.UserRole)
        file = self.reverse_map.get((alias, group_key))
        if not file:
            self.tip_label.setText(f"[错误] 无法映射日志文件: {alias}")
            return
        self.current_log_path = os.path.join(self.log_dir, file)
        try:
            with open(self.current_log_path, "r", encoding="utf-8") as f:
                self.log_content.setPlainText(f.read())
            self.tip_label.setText(f"[提示] 已加载日志文件：{file}")
        except Exception as e:
            self.log_content.clear()
            self.tip_label.setText(f"[错误] 无法读取日志文件: {e}")

    def clear_log_file(self):
        if not self.current_log_path:
            self.tip_label.setText("[提示] 请先选择一个日志文件")
            return
        try:
            with open(self.current_log_path, "w", encoding="utf-8"): pass
            self.log_content.clear()
            self.tip_label.setText("[提示] 日志文件已清空")
        except Exception as e:
            self.tip_label.setText(f"[错误] 无法清空日志文件: {e}")

    def auto_refresh_log(self):
        if not self.current_log_path or not os.path.exists(self.current_log_path): return
        try:
            with open(self.current_log_path, "r", encoding="utf-8") as f:
                content = f.read()
            if self.log_content.toPlainText() != content:
                self.log_content.setPlainText(content)
                self.tip_label.setText(f"[提示] 实时更新：{os.path.basename(self.current_log_path)}")
        except Exception as e:
            self.tip_label.setText(f"[错误] 实时读取失败: {e}")

    def check_log_file_changes(self):
        try:
            current = {f for f in os.listdir(self.log_dir) if f.endswith(".log")}
            if current != self.last_log_files:
                self.last_log_files = current
                self.refresh_file_list()
                self.tip_label.setText("[提示] 日志文件列表已更新")
        except Exception as e:
            self.tip_label.setText(f"[错误] 无法检测日志文件变动: {e}")

    def load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return {"paths": {"log_dir": "logs"}}
    
    def clear_all_logs(self):
        try:
            cleared = 0
            for file in os.listdir(self.log_dir):
                if file.endswith(".log"):
                    with open(os.path.join(self.log_dir, file), "w", encoding="utf-8") as f:
                        f.write("")
                    cleared += 1
            self.refresh_file_list()
            self.log_content.clear()
            self.tip_label.setText(f"[提示] 已清除 {cleared} 个日志文件")
        except Exception as e:
            self.tip_label.setText(f"[错误] 无法清除日志文件: {e}")


import yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QScrollArea, QPushButton, QFormLayout, QDialog,
    QGroupBox, QMessageBox, QStackedWidget, QCheckBox, QComboBox, QFrame
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont, QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QRectF

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

class UnifiedYamlConfigEditor(QWidget):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.config_data = {}
        self.fields = {}

        self.load_config()

        main_layout = QVBoxLayout(self)  # 用主垂直布局包裹
        main_layout.setContentsMargins(0, 0, 0, 0)  # 去除上下左右 padding
        main_layout.setSpacing(0)                  # 去除中间控件间距

        # ✅ 内容区域：左树 + 右表单（横向）
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self.on_item_clicked)
        content_layout.addWidget(self.tree, 2)

        self.editor_stack = QStackedWidget()
        content_layout.addWidget(self.editor_stack, 5)

        main_layout.addWidget(content_widget)   # ✅ 改成 addWidget
        main_layout.setStretchFactor(content_widget, 1)  # ✅ 让内容区域占满高度


        self.build_ui()

    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
        except Exception as e:
            show_custom_message(self,"加载失败", str(e))

    def build_ui(self):
        self.tree.clear()
        while self.editor_stack.count():
            widget = self.editor_stack.widget(0)
            self.editor_stack.removeWidget(widget)
            widget.deleteLater()

        self.fields.clear()

        for block_key, block_value in self.config_data.items():
            block_item = QTreeWidgetItem([self.translate_key(block_key)])
            self.tree.addTopLevelItem(block_item)

            if isinstance(block_value, dict):
                has_sub_dict = any(isinstance(v, dict) for v in block_value.values())
                if has_sub_dict:
                    block_item.setFlags(block_item.flags() & ~Qt.ItemIsSelectable)  # ✅ 禁止顶层节点被选中
                    for sub_key, sub_value in block_value.items():
                        sub_item = QTreeWidgetItem([self.translate_key(sub_key)])
                        block_item.addChild(sub_item)
                        form = self.create_form((block_key, sub_key), sub_value)
                        sub_item.setData(0, 1000, form)
                        self.editor_stack.addWidget(form)
                else:
                    form = self.create_form((block_key,), block_value)
                    block_item.setData(0, 1000, form)
                    self.editor_stack.addWidget(form)
            else:
                form = self.create_form((block_key,), block_value)
                block_item.setData(0, 1000, form)
                self.editor_stack.addWidget(form)

        self.tree.expandAll()

    def create_form(self, keys, values):
        form_widget = QWidget()
        form_widget.setObjectName("EditorForm")

        # 🔥 整体垂直布局（上按钮，下表单）
        overall_layout = QVBoxLayout(form_widget)
        overall_layout.setSpacing(10)
        overall_layout.setContentsMargins(0, 0, 0, 0)

        # === 按钮行 ===
        button_row = QWidget()
        button_row.setMaximumHeight(40)
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()

        save_button = QPushButton("保存修改")
        save_button.setMinimumWidth(100)
        save_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_button)

        reset_button = QPushButton("恢复修改")
        reset_button.setMinimumWidth(100)
        reset_button.clicked.connect(self.reset_current_form_fields)
        button_layout.addWidget(reset_button)

        overall_layout.addWidget(button_row)

        # === 表单区域 ===
        form_area = QWidget()
        form_layout = QFormLayout(form_area)
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        form_layout.setFormAlignment(Qt.AlignTop)
        form_widget.initial_values = {}  # 🔥 新增，保存每个小表单的初始字段

        if isinstance(values, dict):
            for key, val in values.items():
                field_widget = self.build_field_widget(keys + (key,), val)
                form_layout.addRow(self.translate_key(key), field_widget)
                form_widget.initial_values[keys + (key,)] = val  # 保存初始值
        else:
            field_widget = self.build_field_widget(keys, values)
            form_layout.addRow(self.translate_key(keys[-1]), field_widget)
            form_widget.initial_values[keys] = values

        overall_layout.addWidget(form_area)

        return form_widget

    def build_field_widget(self, key_path, value):
        last_key = key_path[-1].lower()

        if isinstance(value, bool):
            widget = self.ToggleSwitch()
            widget.setChecked(value)
            self.fields[key_path] = widget
            return widget

        elif isinstance(value, int):
            widget = QLineEdit(str(value))
            widget.setValidator(QIntValidator())
            widget.setAlignment(Qt.AlignRight)
            self.fields[key_path] = widget
            return widget
        elif isinstance(value, float):
            widget = QLineEdit(str(value))
            widget.setValidator(QDoubleValidator())
            widget.setAlignment(Qt.AlignRight)
            self.fields[key_path] = widget
            return widget

        elif any(x in last_key for x in ["path", "dir", "file"]):
            widget = QLineEdit(str(value))
            widget.setPlaceholderText("路径")
            widget.setAlignment(Qt.AlignRight)
            self.fields[key_path] = widget
            return widget

        else:
            widget = QLineEdit(str(value))
            widget.setAlignment(Qt.AlignRight)
            self.fields[key_path] = widget
            return widget

    def on_item_clicked(self, item, column):
        widget = item.data(0, 1000)
        if widget:
            self.editor_stack.setCurrentWidget(widget)

    def save_config(self):
        try:
            updated = {}
            for keys, field in self.fields.items():
                ref = updated
                for k in keys[:-1]:
                    ref = ref.setdefault(k, {})

                val = None
                if isinstance(field, QLineEdit):
                    val = self.parse_value(field.text())
                elif isinstance(field, QCheckBox):
                    val = field.isChecked()
                elif isinstance(field, QComboBox):
                    val = field.currentText()

                ref[keys[-1]] = val

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(updated, f, allow_unicode=True)
            show_custom_message(self, "成功", "配置已保存")
        except Exception as e:
            show_custom_message(self, "保存失败", str(e))

    def reset_current_form_fields(self):
        try:
            current_widget = self.editor_stack.currentWidget()
            if not hasattr(current_widget, "initial_values"):
                show_custom_message(self, "警告", "当前页面没有初始数据。")
                return

            for keys, value in current_widget.initial_values.items():
                field = self.fields.get(keys)
                if field is not None:
                    if isinstance(field, QLineEdit):
                        field.setText(str(value))
                    elif isinstance(field, QCheckBox):
                        field.setChecked(bool(value))
                    elif isinstance(field, QComboBox):
                        field.setCurrentText(str(value))

            show_custom_message(self,"重置成功", "已重置当前表单的参数。")
        except Exception as e:
            show_custom_message(self,"重置失败", str(e))

    @staticmethod
    def parse_value(value):
        try:
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            if '.' in value:
                return float(value)
            return int(value)
        except:
            return value

    @staticmethod
    def translate_key(key):
        translations = {
            "feature": "特征工程配置",
            "pca": "PCA配置",
            "scaling": "标准化配置",
            "label_map": "标签映射配置",
            "logging": "日志记录配置",
            "backup_count": "备份数",
            "level": "日志等级",
            "max_size": "最大日志文件大小",
            "lstm_params": "LSTM模型配置",
            "data": "数据加载",
            "inference": "推理配置",
            "model": "模型结构",
            "training": "训练配置",
            "valid_classes": "有效类别",
            "paths": "路径配置",
            "project": "项目信息",
            "chunk_size": "数据块大小",
            "dtype": "数据类型",
            "name": "名称",
            "parallel_workers": "并行线程",
            "version": "版本",
            "pca_components": "PCA主成分数",
            "target_column": "目标列",
            "whiten": "白化",
            "method": "标准化方法",
            "with_mean": "均值归一化",
            "with_std": "方差归一化",
            "device": "设备类型",
            "sequence_length": "序列长度",
            "test_size": "测试集比例",
            "topk": "Top-K输出",
            "chunk_overlap": "滑窗重叠比例",
            "distribution_alert_threshold": "分布警报阈值",
            "dropout": "Dropout",
            "feature_dim": "特征维度",
            "hidden_size": "隐藏层大小",
            "num_layers": "层数",
            "bidirectional": "双向",
            "accumulation_steps": "累积步数",
            "batch_size": "批大小",
            "learning_rate": "学习率",
            "num_epochs": "训练轮数",
            "use_amp": "自动混合精度",
            "use_gradient_checkpointing": "梯度检查点优化"
        }
        return translations.get(key, key)



    class ToggleSwitch(QCheckBox):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setCursor(Qt.PointingHandCursor)
            self.setFixedSize(50, 28)
            self.setChecked(False)
            self.setStyleSheet("QCheckBox { background: transparent; }")

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            bg_color = QColor("#4caf50") if self.isChecked() else QColor("#888888")
            circle_color = QColor("#ffffff")

            rect = self.rect()
            radius = rect.height() / 2
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)

            circle_diameter = rect.height() - 4
            x = rect.width() - circle_diameter - 2 if self.isChecked() else 2
            circle_rect = QRectF(x, 2, circle_diameter, circle_diameter)
            painter.setBrush(QBrush(circle_color))
            painter.drawEllipse(circle_rect)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.toggle()
                self.update()
            super().mouseReleaseEvent(event)
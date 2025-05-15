import os
import sys
import yaml
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QStackedWidget, QPushButton, QSplitter, QButtonGroup, QGridLayout, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt
from modules.sysconfig import UnifiedYamlConfigEditor
from modules.data_process import DataCleaningModule
from modules.data_process import FeatureEngineeringModule
from modules.data_process import TrainsetBuildingModule
from modules.monitor import PerformanceMonitor
from modules.model_train_test import LSTMTrainingModule
from modules.log_manage import LogViewerModule

# 空壳模块
class ModelInferenceModule(QWidget):
    def __init__(self): super().__init__()
class TrafficCaptureModule(QWidget):
    def __init__(self): super().__init__()
class AIDetectionModule(QWidget):
    def __init__(self): super().__init__()

class CustomMessageBox(QDialog):
    def __init__(self, title, message, parent=None, confirm=False):
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
        button_layout.addStretch()
                # 判断是否是确认框
        if confirm:
            self.btn_ok = QPushButton("确认")
            self.btn_ok.setMinimumWidth(80)
            self.btn_cancel = QPushButton("取消")
            self.btn_cancel.setMinimumWidth(80)
            self.btn_ok.setObjectName("btn_ok")
            self.btn_ok.clicked.connect(self.accept)
            self.btn_cancel.clicked.connect(self.reject)
            button_layout.addWidget(self.btn_cancel)
            button_layout.addWidget(self.btn_ok)
        else:
            btn = QPushButton("确定")
            btn.setMinimumWidth(80)
            btn.clicked.connect(self.accept)
            button_layout.addWidget(btn)
        layout.addLayout(button_layout)

def show_custom_message(parent, title, text):
    dlg = CustomMessageBox(title, text, parent.window() if parent else None)
    dlg.exec_()

class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(1100, 700)
        self.module_map = {
            "数据集处理": ["数据清洗", "特征工程", "训练集构建"],
            "模型训练测试": ["模型训练", "模型推理测试"],
            "流量监测": ["流量捕获", "AI检测"],
            "日志管理": ["日志管理"], 
            "系统配置": ["系统配置"]
        }
        self.module_widgets = {}
        self.current_category = None

        self.init_ui()
        # 获取屏幕中心点，并将窗口移到中间
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def init_ui(self):
        self.init_title_bar()
        self.init_left_panel()
        self.init_right_panel()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.title_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.content)  # ✅ 这里必须是 self.content
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self._drag_pos = None
        self.nav.setCurrentRow(0)

    def init_title_bar(self):
        # 自定义标题栏
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(40)
        title_layout = QGridLayout(self.title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        title_layout.setColumnStretch(0, 1)
        title_layout.setColumnStretch(1, 0)
        title_layout.setColumnStretch(2, 1)

        self.title_label = QLabel("AI流量检测系统")
        self.title_label.setStyleSheet("color: white; font-size: 21px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(self.title_label, 0, 1, alignment=Qt.AlignCenter)

        btn_min = self.create_window_button("_", self.showMinimized, "btn_minimize")
        btn_max = self.create_window_button("□", self.toggle_maximize_restore, "btn_maximize")
        btn_close = self.create_window_button("×", self.close, "btn_close")

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(3)
        btns.addWidget(btn_min)
        btns.addWidget(btn_max)
        btns.addWidget(btn_close)
        container = QWidget()
        container.setLayout(btns)
        title_layout.addWidget(container, 0, 2, alignment=Qt.AlignRight)

    def init_left_panel(self):
        self.monitor_widget = PerformanceMonitor()

        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        self.nav.setObjectName("Navigation")

        for category in self.module_map:
            item = QListWidgetItem(f"{category}")
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.nav.addItem(item)
        self.nav.currentItemChanged.connect(self.load_category_modules)

        self.reset_button = QPushButton("清除日志/输出")
        self.reset_button.setStyleSheet("background-color: #FF6347; color: white; font-size: 18px;")
        self.reset_button.setFixedHeight(36)
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self.reset_outputs)

        reset_container = QWidget()
        reset_layout = QHBoxLayout(reset_container)
        reset_layout.setContentsMargins(0, 10, 0, 10)
        reset_layout.setSpacing(0)
        reset_layout.addStretch()
        reset_layout.addWidget(self.reset_button)  
        reset_layout.addStretch()

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.monitor_widget)
        left_layout.addWidget(self.nav)
        left_layout.addWidget(reset_container)

        self.left_widget = QWidget()  # ✅ 改成 self.left_widget
        self.left_widget.setLayout(left_layout)
        self.left_widget.setMinimumWidth(200)
        self.left_widget.setMaximumWidth(300)

    def init_right_panel(self):
        self.button_bar = QWidget()
        self.button_bar.setObjectName("module_switch_bar")
        self.button_layout = QHBoxLayout(self.button_bar)
        self.button_layout.setContentsMargins(0, 10, 0, 10)
        self.button_layout.setSpacing(0)

        self.button_group = QButtonGroup()
        self.button_group.setExclusive(True)
        self.button_group.buttonClicked.connect(self.switch_module)

        self.stack = QStackedWidget()

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.button_bar)
        right_layout.addWidget(self.stack)

        self.content = QWidget()   # ✅ 改成 self.content
        self.content.setLayout(right_layout)

    def create_window_button(self, text, slot, obj_name):
        btn = QPushButton(text)
        btn.setFixedSize(40, 30)
        btn.setObjectName(obj_name)
        btn.clicked.connect(slot)
        return btn

    def toggle_maximize_restore(self):
        self._is_maximized = not getattr(self, '_is_maximized', False)
        self.showNormal() if self._is_maximized else self.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            widget = self.childAt(event.pos())
            if widget in [self.title_bar, self.title_label]:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def ensure_widget_loaded(self, widget):
        if self.stack.indexOf(widget) == -1:
            self.stack.addWidget(widget)

    def load_category_modules(self, current, _):
        if not current:
            return
        category = current.text().strip()
        if category == self.current_category:
            return
        self.current_category = category

        # 系统配置模块不显示顶部切换按钮
        if category == "系统配置":
            widget = self.load_or_get_module("系统配置")
            self.ensure_widget_loaded(widget)
            self.stack.setCurrentWidget(widget)
            self.button_bar.hide()
            return

        elif category == "日志管理":
            widget = self.load_or_get_module("日志管理")
            self.ensure_widget_loaded(widget)
            self.stack.setCurrentWidget(widget)
            self.button_bar.hide()
            return

        else:
            self.button_bar.show()

        for btn in self.button_group.buttons():
            self.button_group.removeButton(btn)
            btn.deleteLater()

        for i in reversed(range(self.button_layout.count())):
            item = self.button_layout.itemAt(i).widget()
            if item:
                self.button_layout.removeWidget(item)
                item.deleteLater()

        for i, name in enumerate(self.module_map[category]):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            self.button_group.addButton(btn)
            self.button_layout.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
                self.switch_module(btn)

    def switch_module(self, button):
        name = button.text()
        widget = self.load_or_get_module(name)
        self.ensure_widget_loaded(widget)
        self.stack.setCurrentWidget(widget)

    def load_or_get_module(self, name):
        MODULE_CLASS_MAPPING = {
            "数据清洗": lambda cfg: DataCleaningModule(cfg),
            "特征工程": lambda cfg: FeatureEngineeringModule(cfg),
            "训练集构建": lambda cfg: TrainsetBuildingModule(cfg),
            "模型训练": lambda cfg: LSTMTrainingModule(cfg),
            "模型推理测试": lambda cfg: ModelInferenceModule(),
            "流量捕获": lambda cfg: TrafficCaptureModule(),
            "AI检测": lambda cfg: AIDetectionModule(),
            "日志管理": lambda cfg: LogViewerModule(cfg),
            "系统配置": lambda cfg: UnifiedYamlConfigEditor(cfg),
        }
        if name in self.module_widgets:
            return self.module_widgets[name]
        config = os.path.join(os.path.dirname(__file__), "../config/patterns.yaml")
        widget = MODULE_CLASS_MAPPING.get(name, lambda cfg: QLabel(f"[{name}] 未实现"))(config)
        self.module_widgets[name] = widget
        return widget

    def reset_outputs(self):
        dlg = CustomMessageBox("确认操作", "确定要清除所有日志和输出文件吗？", self, confirm=True)
        if dlg.exec_() == QDialog.Accepted:
            # TODO: 在这里执行清理动作，比如删除logs目录、output目录
            # 示例：
            try:
                config_path = os.path.join(os.path.dirname(__file__), "../config/patterns.yaml")
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                paths = [
                    config['paths']['cleaned_data'],
                    config['paths']['feature_data'],
                    config['paths']['log_dir'],
                    config['paths']['sequence_data'],
                    config['paths']['model_best'],
                    config['paths']['model_ckpt']
                    ]
                for path in paths:
                    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
                    if os.path.exists(abs_path):
                        for root, dirs, files in os.walk(abs_path):
                            for file in files:
                                os.remove(os.path.join(root, file))

                show_custom_message(self, "完成", "日志和输出文件已清除。")
            except Exception as e:
                show_custom_message(self, "错误", f"清除失败: {str(e)}")


def main():
    app = QApplication(sys.argv)
    with open(os.path.join(os.path.dirname(__file__), "UI.qss"), "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
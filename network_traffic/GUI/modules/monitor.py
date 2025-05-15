import psutil
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class PerformanceMonitor(QWidget):
    status_updated = pyqtSignal(dict)

    def __init__(self, interval=500, parent=None):
        """
        作为独立的 QWidget 小部件，带有自动刷新和样式
        :param interval: 刷新间隔（毫秒）
        """
        super().__init__(parent)
        self.interval = interval / 1000.0  # 转为秒
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = None
        self.gpu_memory_usage = None
        self._lock = threading.Lock()
        self._thread = None
        self.running = False

        # ===== 界面元素 =====
        self.cpu_mem_label = QLabel("CPU: -- %    内存: -- %")
        self.gpu_label = QLabel("GPU: -- %    显存: -- %")

        layout = QVBoxLayout()
        layout.addWidget(self.cpu_mem_label)
        layout.addWidget(self.gpu_label)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        self.setLayout(layout)

        self.setFixedSize(200, 80)
        self.apply_styles()

        # ===== 定时器刷新 =====
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(int(interval))

        # ===== 后台性能监控线程 =====
        self.start_monitor()

    def apply_styles(self):
        """应用暗色卡片样式"""
        self.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
            }
        """)
        self.cpu_mem_label.setAlignment(Qt.AlignCenter)
        self.gpu_label.setAlignment(Qt.AlignCenter)

    def start_monitor(self):
        """启动后台监控"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop_monitor(self):
        """停止后台监控"""
        self.running = False
        if self._thread is not None:
            self._thread.join()

    def _monitor_loop(self):
        while self.running:
            with self._lock:
                self.cpu_usage = psutil.cpu_percent(interval=None)
                self.memory_usage = psutil.virtual_memory().percent
                if GPU_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.gpu_usage = gpu.load * 100
                        self.gpu_memory_usage = (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else None
                    else:
                        self.gpu_usage = None
                        self.gpu_memory_usage = None
            time.sleep(self.interval)

    def update_display(self):
        """定时刷新界面内容"""
        with self._lock:
            cpu_text = f"CPU: {self.cpu_usage:.0f}%    内存: {self.memory_usage:.0f}%"
            if self.gpu_usage is not None:
                gpu_text = f"GPU: {self.gpu_usage:.0f}%    显存: {self.gpu_memory_usage:.0f}%"
            else:
                gpu_text = "GPU 信息不可用"

            self.cpu_mem_label.setText(cpu_text)
            self.gpu_label.setText(gpu_text)

    def closeEvent(self, event):
        self.stop_monitor()
        super().closeEvent(event)
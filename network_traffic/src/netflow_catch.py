import scapy.all as scapy
import torch
import csv
import dash
from dash import dcc, html
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import threading

# 1. 实时流量捕获模块
def packet_callback(packet):
    # 分析每一个捕获的数据包
    if packet.haslayer(scapy.IP):
        analyze_packet(packet)

# 2. 检测分析模块
def analyze_packet(packet):
    # 加载并推理模型
    model = load_model()  # 加载模型
    features = extract_features(packet)  # 提取特征
    prediction = infer(model, features)  # 使用推理函数预测
    
    # 如果是正常流量，写入CSV文件
    if prediction == 0:  # 假设0表示正常流量
        save_to_csv(packet)

# 假设从数据包中提取简单的特征
def extract_features(packet):
    if packet.haslayer(scapy.IP):
        return [packet[scapy.IP].src, packet[scapy.IP].dst, packet[scapy.IP].proto]
    return [0, 0, 0]

# 加载模型的函数
def load_model():
    model = torch.load('traffic_model.pt')  # 假设模型保存在 traffic_model.pt 文件中
    model.eval()  # 设置模型为评估模式
    return model

# 使用模型进行推理
def infer(model, features):
    # 假设模型输入为一个Tensor类型的特征向量
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        prediction = model(features_tensor)
    predicted_class = prediction.argmax(dim=1).item()  # 获取预测的类别
    return predicted_class

# 将正常流量写入CSV文件
def save_to_csv(packet):
    fieldnames = ['Source IP', 'Destination IP', 'Protocol', 'Timestamp']
    
    # 打开CSV文件进行写入
    with open('normal_traffic.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 如果文件为空，写入列标题
        if file.tell() == 0:
            writer.writeheader()
        
        # 写入数据包信息
        writer.writerow({
            'Source IP': packet[scapy.IP].src,
            'Destination IP': packet[scapy.IP].dst,
            'Protocol': packet[scapy.IP].proto,
            'Timestamp': packet.time
        })

# 3. 可视化界面模块
def create_dashboard():
    # 创建Dash应用
    app = dash.Dash(__name__)

    # 模拟一些流量数据，示例为每秒的流量
    time_series = [1, 2, 1, 3, 2, 4, 5, 3, 2, 1]  # 模拟数据

    # 创建图表
    fig = go.Figure(data=[go.Scatter(x=list(range(10)), y=time_series)])
    fig.update_layout(title='Network Traffic Trend', xaxis_title='Time (s)', yaxis_title='Packet Count')

    # 设计布局
    app.layout = html.Div([
        html.H1('流量检测系统'),
        dcc.Graph(figure=fig)
    ])

    return app

# 启动流量捕获线程
def start_packet_sniffer():
    scapy.sniff(prn=packet_callback, store=0)

# 启动可视化界面的后台线程
def start_dashboard():
    app = create_dashboard()
    app.run_server(debug=True, use_reloader=False)  # use_reloader=False避免重复启动

# 主函数
if __name__ == '__main__':
    # 使用多线程并行执行流量捕获与可视化界面
    sniffer_thread = threading.Thread(target=start_packet_sniffer)
    dashboard_thread = threading.Thread(target=start_dashboard)

    sniffer_thread.start()
    dashboard_thread.start()

    # 让主线程等待直到所有线程完成
    sniffer_thread.join()
    dashboard_thread.join()

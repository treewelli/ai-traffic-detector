import torch
import torch.nn as nn
import time

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=4, batch_first=True)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM().to(device)
x = torch.randn(8192, 10,128 ).to(device)
y = torch.randint(0, 10, (8192,)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(">>> 开始前向与反向传播")
start = time.time()
for i in range(10):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"Batch {i+1}: Loss={loss.item():.4f}")
end = time.time()

print(f"✅ 完成训练循环，总耗时 {end - start:.2f}s")

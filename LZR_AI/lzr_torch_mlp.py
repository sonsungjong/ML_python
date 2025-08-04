# pip3 install torch torchvision torchaudio
# 예측 시간: 0.006830초

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

# CUDA / TORCH 사용 가능 여부
print(torch.cuda.is_available())  # True여야 정상
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# 데이터 로드
df = pd.read_csv("mdi.csv", sep=";")
target_col = 'HUMAN'
X = df.drop(columns=[target_col]).values.astype('float32')
y = df[target_col].values.astype('int64')

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 정의
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 2)  # 이진 분류

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # CrossEntropyLoss 사용시 소프트맥스 불필요

# GPU 우선 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = SimpleMLP(input_dim).to(device)

# PyTorch Dataset, DataLoader (전체를 한 번에 처리, 배치X)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long, device=device)

# 단일 예측용 함수
def predict_single(model, input_arr, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_arr, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        return pred  # 0 or 1

# 옵티마이저, 손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 간단한 학습 (10에폭)
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# 예측 및 시간 측정
model.eval()
start_time = time.time()
with torch.no_grad():
    logits = model(X_test_tensor)
    preds = logits.argmax(dim=1).cpu().numpy()
end_time = time.time()

# 평가
print("정확도:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print(f"예측 시간: {end_time - start_time:.6f}초")

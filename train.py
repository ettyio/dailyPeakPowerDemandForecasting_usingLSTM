import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from config import Config
from data_collector import fetch_weather_data, load_power_data_from_files
from utils import get_processed_data, create_sequences
from model import PowerDemandLSTM

def main():
    print(">>> 1. 데이터 수집 중...")
    weather_df = fetch_weather_data()
    power_files = [
        "2021년 1_12월 수요관리후 발전단 수요실적.csv",
        "한국전력거래소_시간별 전국 전력수요량_20221231.csv",
        "시간별 전국 전력수요량_20231231.csv",
        "한국전력거래소_시간별 전국 전력수요량_20241231.csv"
    ]
    power_df = load_power_data_from_files(power_files)
    
    if weather_df.empty or power_df.empty:
        print("[에러] 데이터 수집 실패")
        return

    print(">>> 2. 데이터 전처리 및 파생변수 생성...")
    scaled_data, scaler, num_features, dates = get_processed_data(weather_df, power_df)
    
    X, y = create_sequences(scaled_data, Config.SEQ_LENGTH)
    
    train_size = int(len(X) * Config.TRAIN_SPLIT)
    
    X_train = torch.FloatTensor(X[:train_size]).to(Config.DEVICE)
    y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(Config.DEVICE)
    X_test = torch.FloatTensor(X[train_size:]).to(Config.DEVICE)
    y_test = torch.FloatTensor(y[train_size:]).unsqueeze(1).to(Config.DEVICE)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print(f">>> 3. 학습 시작 (Features: {num_features}개, Loss: Huber)")
    model = PowerDemandLSTM(
        input_size=num_features,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # HuberLoss 사용
    criterion = nn.HuberLoss(delta=1.0)
    
    # Weight Decay 추가 (과적합 방지)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 학습률 스케줄러 (Loss가 안 줄어들면 LR 감소)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss) # 스케줄러 업데이트
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 모델 저장 로직 추가 가능
            
        if (epoch+1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   [Epoch {epoch+1}] Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    print(">>> 4. 예측 수행...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    # 역변환
    target_min = scaler.data_min_[-1]
    target_range = scaler.data_range_[-1]
    
    actual_load = y_test_np * target_range + target_min
    predicted_load = test_pred * target_range + target_min

    model_dates = dates[Config.SEQ_LENGTH:].reset_index(drop=True)
    test_dates = model_dates[train_size:].reset_index(drop=True)

    result_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_load.flatten(),
        'Pred': predicted_load.flatten()
    })
    
    # MAPE 계산
    result_df['MAPE'] = np.abs((result_df['Actual'] - result_df['Pred']) / result_df['Actual']) * 100
    mape = result_df['MAPE'].mean()
    
    print(f"\n>>> 최종 결과 (MAPE): {mape:.2f}%")
    
    result_df.to_csv("train_result_huber.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(result_df['Date']), result_df['Actual'], label='Actual', color='black', alpha=0.7)
    plt.plot(pd.to_datetime(result_df['Date']), result_df['Pred'], label='Prediction', color='red', alpha=0.8)
    plt.title(f'Prediction Result (MAPE: {mape:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("train_plot_huber.png")
    print(">>> 그래프 저장 완료.")

if __name__ == "__main__":
    main()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta

from config import Config
from data_collector import fetch_weather_data, load_power_data_from_files
from utils import get_processed_data, create_sequences, preprocess_and_merge
from model import PowerDemandLSTM
from workalendar.asia import SouthKorea
import urllib3
from save_results import save_and_plot

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_reference_data():
    """비교용 데이터(KPX 예측, 실적) 로드 - 이상치 제거 기능 추가"""
    print("\n>>> [참조 데이터] 비교용 파일 로드 중...")
    
    # 1. KPX 5분 예측 데이터
    forecast_files = glob.glob("5분수요예측MW_2025년*.csv")
    kpx_forecast_df = pd.DataFrame()
    
    if forecast_files:
        dfs = []
        for f in forecast_files:
            try:
                try: df = pd.read_csv(f, skiprows=2, encoding='utf-8')
                except: df = pd.read_csv(f, skiprows=2, encoding='euc-kr')
                
                df['tm'] = pd.to_datetime(df['시간'], format='mixed', errors='coerce')
                df = df.dropna(subset=['tm'])
                df['KPX_Forecast'] = pd.to_numeric(df['수요예측MW'], errors='coerce')
                
                # 150,000 MW 이상의 비정상 데이터 제거
                df = df[df['KPX_Forecast'] < 150000] 
                
                df['Date'] = df['tm'].dt.date
                daily_max = df.groupby('Date')['KPX_Forecast'].max().reset_index()
                dfs.append(daily_max)
            except: pass
        if dfs:
            kpx_forecast_df = pd.concat(dfs).drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
            kpx_forecast_df['Date'] = pd.to_datetime(kpx_forecast_df['Date'])

    # 2. 실제 실적 데이터
    actual_file = "25년 일별 최대전력수급.csv"
    actual_df = pd.DataFrame()
    if os.path.exists(actual_file):
        try:
            try: df = pd.read_csv(actual_file, encoding='utf-8')
            except: df = pd.read_csv(actual_file, encoding='euc-kr')
            df['Date'] = pd.to_datetime(df[['년', '월', '일']].astype(str).agg('-'.join, axis=1))
            
            target_col = [c for c in df.columns if '최대전력' in c and '일시' not in c][0]
            if df[target_col].dtype == object:
                df['Actual_Load'] = df[target_col].str.replace(',', '').astype(float)
            else: df['Actual_Load'] = df[target_col]
            
            # 실제 실적도 혹시 모를 오타 방지
            df = df[df['Actual_Load'] < 150000]
            
            actual_df = df[['Date', 'Actual_Load']].sort_values('Date').reset_index(drop=True)
        except: pass

    if kpx_forecast_df.empty: kpx_forecast_df = pd.DataFrame(columns=['Date', 'KPX_Forecast'])
    if actual_df.empty: actual_df = pd.DataFrame(columns=['Date', 'Actual_Load'])
    return kpx_forecast_df, actual_df

def main():
    # ------------------------------------------------------------------
    # 1. 학습 데이터 로드 및 모델 학습
    # ------------------------------------------------------------------
    print(">>> 1. 학습 데이터 로드 및 모델 학습...")
    Config.START_DT = "20210101"
    Config.END_DT = "20241231" # 학습용 종료일
    
    past_weather = fetch_weather_data()
    power_files = [
        "2021년 1_12월 수요관리후 발전단 수요실적.csv",
        "한국전력거래소_시간별 전국 전력수요량_20221231.csv",
        "시간별 전국 전력수요량_20231231.csv",
        "한국전력거래소_시간별 전국 전력수요량_20241231.csv"
    ]
    past_power = load_power_data_from_files(power_files)
    
    # 학습 데이터 생성
    scaled_data, scaler, num_features, _ = get_processed_data(past_weather, past_power)
    X, y = create_sequences(scaled_data, Config.SEQ_LENGTH)
    
    X_train = torch.FloatTensor(X).to(Config.DEVICE)
    y_train = torch.FloatTensor(y).unsqueeze(1).to(Config.DEVICE)
    
    model = PowerDemandLSTM(num_features, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.OUTPUT_SIZE, Config.DROPOUT).to(Config.DEVICE)
    
    # Huber Loss 사용
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    model.train()
    print(">>> 모델 학습 시작 (Huber Loss)...")
    for epoch in range(Config.EPOCHS):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item()) # 스케줄러 갱신
        
        if (epoch+1) % 50 == 0: 
            print(f"    Epoch {epoch+1} Loss: {loss.item():.5f}")

    # ------------------------------------------------------------------
    # 2. 2025년 예측을 위한 데이터 준비 (Sliding Window 방식)
    # ------------------------------------------------------------------
    print("\n>>> 2. 2025년 예측 데이터 준비 (데이터 병합 중)...")
    
    # (1) 2025년 날씨 데이터 가져오기
    today_str = datetime.now().strftime("%Y%m%d")
    Config.START_DT = "20250101"
    Config.END_DT = today_str
    future_weather = fetch_weather_data()
    
    if future_weather.empty:
        print("[에러] 2025년 날씨 데이터 없음.")
        return

    # (2) 과거 데이터와 미래 데이터 병합 (이동평균, 시퀀스 연결을 위해 필수)
    # 2024년 12월 데이터(최소 SEQ_LENGTH 만큼)가 2025년 1월 1일 예측의 '과거'로 쓰여야 함
    # 편의상 전체 과거 날씨와 2025년 날씨를 합침
    full_weather = pd.concat([past_weather, future_weather]).drop_duplicates(subset=['tm']).sort_values('tm').reset_index(drop=True)
    
    # 전력 데이터는 2025년 부분이 없으므로 Dummy로 채워서 병합
    dummy_future_power = pd.DataFrame({'tm': future_weather['tm'], 'Load': 0})
    full_power = pd.concat([past_power, dummy_future_power]).reset_index(drop=True)
    
    # (3) 전처리 (여기서 T_avg_Roll3 등이 끊김 없이 계산됨)
    full_df = preprocess_and_merge(full_weather, full_power)
    
    # (4) Feature Scaling
    # 학습 때 쓴 scaler를 그대로 사용해야 함.
    # feature_cols 순서: utils.py와 동일
    feature_cols = [
        'T_max', 'T_avg', 'WetBulb', 'CDD', 'HDD', 
        'T_avg_Roll3', 'T_max_Roll3', 'WetBulb_Roll3', 'Discomfort_Idx',
        'Load_Last_Year', 
        'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 
        'IsWeekend', 'IsLawHoliday'
    ]
    target_col = ['Load_Max']
    
    # 전체 데이터를 Scale 변환
    raw_data_full = full_df[feature_cols + target_col].values
    scaled_full = scaler.transform(raw_data_full) # fit_transform 아님! transform만!
    
    # ------------------------------------------------------------------
    # 3. 예측 수행 (Sliding Window)
    # ------------------------------------------------------------------
    print(">>> 3. 예측 수행 (Sliding Window 적용)...")
    
    # 2025년 데이터가 시작되는 인덱스 찾기
    start_date_2025 = pd.to_datetime("2025-01-01")
    start_idx = full_df[full_df['Date'] == start_date_2025].index[0]
    
    # 예측할 날짜들
    predict_dates = full_df.loc[start_idx:, 'Date'].reset_index(drop=True)
    X_pred_list = []
    
    # SEQ_LENGTH 데이터를 슬라이딩하며 가져오기
    # i는 2025년 날짜의 인덱스
    for i in range(start_idx, len(full_df)):
        # i 시점의 데이터를 예측하기 위해 [i-SEQ_LENGTH : i] 구간의 데이터를 입력으로 사용
        # 입력 데이터에는 Target(Load) 정보는 제외해야 하므로 [:, :-1] 슬라이싱
        seq = scaled_full[i - Config.SEQ_LENGTH : i, :-1]
        X_pred_list.append(seq)
        
    X_pred = torch.FloatTensor(np.array(X_pred_list)).to(Config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        preds = model(X_pred).cpu().numpy()
    
    # 역변환
    t_min = scaler.data_min_[-1]
    t_range = scaler.data_range_[-1]
    pred_mw = preds.flatten() * t_range + t_min
    
    result_df = pd.DataFrame({'Date': predict_dates, 'LSTM_Pred': pred_mw})

    # 참조 데이터 로드 및 저장
    kpx_df, actual_df = load_reference_data()
    save_and_plot(result_df, kpx_df, actual_df)

if __name__ == "__main__":
    main()
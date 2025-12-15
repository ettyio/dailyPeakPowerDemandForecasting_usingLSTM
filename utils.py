import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import Config
from workalendar.asia import SouthKorea

def preprocess_and_merge(weather_hourly, power_hourly):
    """기상, 전력 데이터를 일별 기준으로 집계하고 병합"""
    
    # 1. 기상 데이터 일별 집계
    weather_hourly['Date'] = weather_hourly['tm'].dt.date
    daily_weather = weather_hourly.groupby('Date').agg({
        'T': ['max', 'mean', 'min'], # min 추가
        'WetBulb': 'mean'
    })
    # 컬럼 레벨 정리
    daily_weather.columns = ['T_max', 'T_avg', 'T_min', 'WetBulb']
    daily_weather = daily_weather.reset_index()

    # 2. 전력 데이터 일별 집계
    power_hourly['Date'] = power_hourly['tm'].dt.date
    daily_power = power_hourly.groupby('Date').agg({
        'Load': 'max'
    }).reset_index()
    daily_power.columns = ['Date', 'Load_Max']

    # 3. 데이터 병합
    df = pd.merge(daily_weather, daily_power, on='Date', how='inner')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 날짜순 정렬 (for 이동평균 계산)
    df = df.sort_values('Date').reset_index(drop=True)

    # 4. 파생변수 생성
    df['CDD'] = df['T_avg'].apply(lambda x: max(0, x - Config.T_BASE_CDD))
    df['HDD'] = df['T_avg'].apply(lambda x: max(0, Config.T_BASE_HDD - x))
    
    # 열 축적 효과 (이동평균)
    # 3일, 7일 이동평균 (결측치는 뒤의 값으로 채움)
    df['T_avg_Roll3'] = df['T_avg'].rolling(window=3).mean().bfill()
    df['T_max_Roll3'] = df['T_max'].rolling(window=3).mean().bfill()
    df['WetBulb_Roll3'] = df['WetBulb'].rolling(window=3).mean().bfill()
    
    # 불쾌지수 (WetBulb와 T_avg 이용한 근사값)
    # 0.72 * (T + WetBulb) + 40.6
    df['Discomfort_Idx'] = 0.72 * (df['T_avg'] + df['WetBulb']) + 40.6

    # 작년 동요일 부하
    df['Load_Last_Year'] = df['Load_Max'].shift(364)
    
    # 순환 변수
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # 공휴일
    cal = SouthKorea()
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    def check_law_holiday(dt):
        is_holiday = cal.is_holiday(dt)
        is_weekend = dt.weekday() >= 5
        return 1 if is_holiday and not is_weekend else 0
        
    df['IsLawHoliday'] = df['Date'].apply(check_law_holiday)

    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i+seq_length, :-1] 
        y = data[i+seq_length, -1]      
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_processed_data(weather_df, power_df):
    """학습용 데이터 생성"""
    merged_df = preprocess_and_merge(weather_df, power_df)
    
    # 결측치 제거
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    # 추가된 파생변수 포함
    feature_cols = [
        'T_max', 'T_avg', 'WetBulb', 'CDD', 'HDD', 
        'T_avg_Roll3', 'T_max_Roll3', 'WetBulb_Roll3', 'Discomfort_Idx', # 추가된 변수들
        'Load_Last_Year', 
        'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 
        'IsWeekend', 'IsLawHoliday'
    ]
    target_col = ['Load_Max']
    
    # 컬럼 존재 여부 확인 (예외처리)
    existing_features = [c for c in feature_cols if c in merged_df.columns]
    
    raw_data = merged_df[existing_features + target_col].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)
    
    return scaled_data, scaler, len(existing_features), merged_df['Date']
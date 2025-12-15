import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def save_and_plot(result_df, kpx_df, actual_df):
    print("\n>>> [Save Results] 결과 저장 및 시각화 시작...")

    # 1. 데이터 병합
    final_compare = result_df.merge(kpx_df, on='Date', how='left') \
                             .merge(actual_df, on='Date', how='left')
    
    # 2. 성능 평가
    eval_df = final_compare.dropna(subset=['Actual_Load', 'LSTM_Pred'])
    
    title_text = "2025 Power Demand Forecast"
    
    if not eval_df.empty:
        mape = np.mean(np.abs((eval_df['Actual_Load'] - eval_df['LSTM_Pred']) / eval_df['Actual_Load'])) * 100
        rmse = np.sqrt(mean_squared_error(eval_df['Actual_Load'], eval_df['LSTM_Pred']))
        mae = mean_absolute_error(eval_df['Actual_Load'], eval_df['LSTM_Pred'])
        
        print(f"    - 데이터 개수: {len(eval_df)}일")
        print(f"    - MAPE: {mape:.2f}%")
        print(f"    - RMSE: {rmse:.2f} MW")
        print(f"    - MAE:  {mae:.2f} MW")
        
        title_text += f" (MAPE: {mape:.2f}%)"
    else:
        print("    [알림] 비교할 실제 실적 데이터가 없습니다.")

    # 3. CSV 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"forecast_comparison_{current_time}.csv"
    final_compare.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"    -> CSV 저장 완료: {csv_filename}")
    
    # 4. 그래프 그리기
    plt.figure(figsize=(14, 7))
    
    # 데이터 플로팅
    # 내 모델 (빨간 실선)
    plt.plot(final_compare['Date'], final_compare['LSTM_Pred'], 
             color='red', label='My LSTM Model', linewidth=2)
    
    # 실제 실적 (검정 실선)
    if 'Actual_Load' in final_compare.columns:
        plt.plot(final_compare['Date'], final_compare['Actual_Load'], 
                 color='black', label='Actual Load', linewidth=1.5, alpha=0.8)
    
    # KPX 예측 (초록 점선) - 이상치(15만 이상)는 NaN 처리하여 그래프에서 숨김
    if 'KPX_Forecast' in final_compare.columns:
        clean_kpx = final_compare['KPX_Forecast'].copy()
        clean_kpx[clean_kpx > 150000] = np.nan # 그래프 그릴 때만 제거
        plt.plot(final_compare['Date'], clean_kpx, 
                 color='green', label='KPX Forecast', linestyle='--', alpha=0.7)
        
    # # KPX 예측 (파란 점선) - 예측값에 9%를 더해 발전단 기준으로 보정 (x 1.09)
    # if 'KPX_Forecast' in final_compare.columns:
    #     clean_kpx = final_compare['KPX_Forecast'] * 1.09
    #     clean_kpx[clean_kpx > 150000] = np.nan
    #     plt.plot(final_compare['Date'], clean_kpx, 
    #          color='blue', label='KPX Forecast (+9% Adjusted)', linestyle='--', alpha=0.7)

    plt.title(title_text)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    
    # Y축 범위 강제 설정 (대한민국 전력수요 범위)
    # 4만 MW ~ 11만 MW 사이로 고정하되, 데이터에 따라 유동적으로 약간 조정
    plt.ylim(30000, 110000) 

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = f"forecast_plot_{current_time}.png"
    plt.savefig(plot_filename)
    print(f"    -> Plot 저장 완료: {plot_filename}")
    print(">>> 모든 저장 작업 완료.")
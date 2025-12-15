import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ---------------------------------------------------------
# [설정] 한글 폰트
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def robust_read_csv(filepath):
    """
    파일을 읽기 위해 온갖 방법을 다 동원하는 함수
    """
    # 1. 인코딩 시도 목록
    encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']
    
    for enc in encodings:
        try:
            # 2. 헤더 위치 찾기 ('시간' 컬럼 찾기)
            for h in range(5):
                try:
                    df = pd.read_csv(filepath, encoding=enc, header=h)
                    # 컬럼 공백 제거
                    df.columns = df.columns.str.strip()
                    
                    if '시간' in df.columns and '수요예측MW' in df.columns:
                        return df # 성공하면 바로 반환
                except:
                    continue
        except:
            continue
            
    return None

def save_and_plot_comparison(kpx_df, actual_df):
    print("\n>>> [분석] 데이터 병합 및 시각화 시작...")
    
    # 병합
    final_compare = pd.merge(actual_df, kpx_df, on='Date', how='inner')
    final_compare.sort_values('Date', inplace=True)
    
    if final_compare.empty:
        print("   [오류] 병합된 데이터가 없습니다. 날짜 형식을 확인하세요.")
        return

    # KPX 보정 (+9%)
    if 'KPX_Forecast' in final_compare.columns:
        temp = final_compare['KPX_Forecast'].copy()
        temp[temp > 150000] = np.nan # 이상치 제거
        final_compare['KPX_Adjusted'] = temp * 1.09

    # 성능 평가
    eval_df = final_compare.dropna(subset=['Actual_Load', 'KPX_Adjusted'])
    if not eval_df.empty:
        mape = np.mean(np.abs((eval_df['Actual_Load'] - eval_df['KPX_Adjusted']) / eval_df['Actual_Load'])) * 100
        print(f"   -> 분석 기간: {eval_df['Date'].min()} ~ {eval_df['Date'].max()} (총 {len(eval_df)}일)")
        print(f"   -> 전체 평균 MAPE: {mape:.2f}%")
    
    # CSV 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_compare.to_csv(f"Comparison_Result_{current_time}.csv", index=False, encoding='utf-8-sig')

    # 그래프
    plt.figure(figsize=(16, 8))
    
    # 실적 (검정)
    plt.plot(final_compare['Date'], final_compare['Actual_Load'], 
             color='black', label='실제 전력수요 (Actual)', linewidth=2, alpha=0.8)
    
    # KPX 보정 (빨강)
    if 'KPX_Adjusted' in final_compare.columns:
        plt.plot(final_compare['Date'], final_compare['KPX_Adjusted'], 
                 color='red', label='KPX 예측 보정 (+9%, 발전단)', linewidth=1.5, alpha=0.9)
        
    plt.title(f"2025년 1월~10월 전력수요 예측 비교 (MAPE: {mape:.2f}%)")
    plt.xlabel('날짜')
    plt.ylabel('전력수요 (MW)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_file = f"Comparison_Plot_{current_time}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"   -> 그래프 저장 완료: {plot_file}")


if __name__ == "__main__":
    print(">>> 1. KPX 예측 파일(1월~10월) 정밀 로딩 시작...")
    kpx_daily_list = []
    
    for m in range(1, 11):
        patterns = [
            f"5분수요예측MW_2025년{m}월.csv", 
            f"5분수요예측MW_2025년{m:02d}월.csv",
            f"5분수요예측MW_2025년 {m}월.csv" # 띄어쓰기 혹시 몰라 추가
        ]
        
        found_path = None
        for p in patterns:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            print(f"   [{m}월] 파일 발견: {found_path}", end=" -> ")
            try:
                # 강력한 읽기 함수 호출
                df = robust_read_csv(found_path)
                
                if df is not None:
                    # 날짜 파싱
                    df['시간'] = pd.to_datetime(df['시간'], format='mixed', errors='coerce')
                    df['Date'] = df['시간'].dt.date
                    
                    # 데이터 존재 여부 확인
                    if df['Date'].isnull().all():
                         print("실패 (날짜 파싱 불가)")
                    else:
                        daily_max = df.groupby('Date')['수요예측MW'].max().reset_index()
                        kpx_daily_list.append(daily_max)
                        print(f"성공! ({len(daily_max)}일 데이터)")
                else:
                    print("실패 (헤더/인코딩 인식 불가)")
            except Exception as e:
                print(f"에러 ({e})")
        else:
            print(f"   [{m}월] 파일을 찾을 수 없음 (파일명 확인 필요)")

    # 통합
    if kpx_daily_list:
        kpx_total_df = pd.concat(kpx_daily_list, ignore_index=True)
        kpx_total_df.columns = ['Date', 'KPX_Forecast']
        print(f"\n   -> KPX 데이터 총 {len(kpx_total_df)}일 확보됨.")
    else:
        print("\n   -> [치명적 오류] 읽어온 예측 데이터가 하나도 없습니다.")
        exit()

    print("\n>>> 2. 실적 데이터 로딩...")
    actual_file = '25년 일별 최대전력수급.csv'
    
    try:
        # 실적 파일 읽기 시도
        df_act = robust_read_csv(actual_file)
        if df_act is None: # 실패 시 기존 방식 시도
             df_act = pd.read_csv(actual_file, encoding='cp949')
             
        # 날짜 컬럼 생성
        if 'Date' not in df_act.columns:
            df_act['Date_Str'] = (df_act['년'].astype(str) + '-' + 
                                  df_act['월'].astype(str).str.zfill(2) + '-' + 
                                  df_act['일'].astype(str).str.zfill(2))
            df_act['Date'] = pd.to_datetime(df_act['Date_Str']).dt.date
            
        actual_total_df = df_act[['Date', '최대전력(MW)']].copy()
        actual_total_df.columns = ['Date', 'Actual_Load']
        print(f"   -> 실적 데이터 {len(actual_total_df)}일 로드 완료.")
        
        save_and_plot_comparison(kpx_total_df, actual_total_df)
        
    except Exception as e:
        print(f"   -> [오류] 실적 파일 로드 중 에러: {e}")
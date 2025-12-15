import requests
import pandas as pd
import math
import os
import time
from datetime import datetime, timedelta
from config import Config

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def calculate_wetbulb(temp, rh):
    """습구온도 계산 (Stull 근사식)"""
    try:
        tw = (temp * math.atan(0.151977 * (rh + 8.313659)**0.5) + 
              math.atan(temp + rh) - math.atan(rh - 1.676331) + 
              0.00391838 * (rh)**1.5 * math.atan(0.023101 * rh) - 4.686035)
        return tw
    except:
        return temp

def fetch_weather_data():
    """
    기상청 API 데이터 조회
    """
    base_url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    # 수집 종료일이 '오늘' 또는 '미래'라면, 무조건 '어제'로 변경
    today = datetime.now().strftime("%Y%m%d")
    # yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    yesterday = '20251204' # ppt 제작을 위한 조정
    
    req_start = Config.START_DT
    req_end = Config.END_DT
    
    # 종료일 강제 설정
    if req_end >= today:
        print(f"    [알림] ASOS 관측 데이터는 과거 데이터만 제공하므로,")
        print(f"           종료일을 '{req_end}'에서 '{yesterday}'로 자동 변경합니다.")
        req_end = yesterday
        
    if req_start > req_end:
        print("    [경고] 시작일이 종료일보다 미래입니다. 데이터를 수집할 수 없습니다.")
        return pd.DataFrame()

    start_year = int(req_start[:4])
    end_year = int(req_end[:4])
    
    print(f">>> [기상청] 데이터 수집 시작 ({req_start} ~ {req_end})")
    
    final_df = pd.DataFrame()
    
    for year in range(start_year, end_year + 1):
        curr_start = f"{year}0101"
        curr_end = f"{year}1231"
        
        # 범위 조정
        if curr_start < req_start: curr_start = req_start
        if curr_end > req_end: curr_end = req_end
        
        print(f"    - {year}년 데이터 요청 중... ({curr_start} ~ {curr_end})")
        
        # 타임아웃 방지
        params = {
            "serviceKey": Config.WEATHER_API_KEY,
            "pageNo": 1, 
            "numOfRows": 500, 
            "dataType": "JSON",
            "dataCd": "ASOS", 
            "dateCd": "HR", 
            "stnIds": Config.STN_ID,
            "startDt": curr_start, 
            "startHh": "00",
            "endDt": curr_end, 
            "endHh": "23"
        }
        
        year_items = []
        
        try:
            # 타임아웃 60초 설정
            resp = requests.get(base_url, params=params, verify=False, timeout=120)
            
            try:
                data = resp.json()
            except:
                print(f"      [에러] {year}년 응답이 JSON이 아닙니다. (건너뜀)")
                continue

            if 'response' not in data or 'body' not in data['response']:
                 print(f"      [에러] {year}년 본문 없음.")
                 continue
                 
            total_count = data['response']['body']['totalCount']
            if total_count == 0:
                print(f"      [알림] {year}년 조회된 데이터가 0건입니다.")
                continue

            items = data['response']['body']['items']['item']
            year_items.extend(items)
            
            total_pages = math.ceil(total_count / 500)
            if total_pages > 1:
                for page in range(2, total_pages + 1):
                    params['pageNo'] = page
                    time.sleep(0.2) # 서버 부하 방지
                    resp = requests.get(base_url, params=params, verify=False, timeout=30)
                    items = resp.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
                    year_items.extend(items)
            
            df_year = pd.DataFrame(year_items)
            df_year['tm'] = pd.to_datetime(df_year['tm'])
            df_year['T'] = pd.to_numeric(df_year['ta'], errors='coerce')
            df_year['RH'] = pd.to_numeric(df_year['hm'], errors='coerce')
            df_year = df_year.dropna(subset=['T', 'RH'])
            df_year['WetBulb'] = df_year.apply(lambda row: calculate_wetbulb(row['T'], row['RH']), axis=1)
            
            final_df = pd.concat([final_df, df_year[['tm', 'T', 'WetBulb']]])
            print(f"      -> 성공: {len(df_year)}행")

        except Exception as e:
            print(f"      [API 통신 오류] {e}")
            
    if final_df.empty:
        return pd.DataFrame()
        
    final_df = final_df.drop_duplicates(subset=['tm']).sort_values('tm').reset_index(drop=True)
    print(f">>> [기상청] 최종 수집 완료: 총 {len(final_df)}행")
    return final_df

def load_power_data_from_files(file_list):
    """ (기존과 동일) """
    all_dfs = []
    print(f">>> [전력 데이터] 파일 {len(file_list)}개 로드 시작...")
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            continue
            
        try:
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except: 
                try: df = pd.read_csv(file_path, encoding='euc-kr')
                except: df = pd.read_csv(file_path, encoding='cp949')
            
            date_col = next((c for c in df.columns if '날짜' in c or 'Date' in c), None)
            if not date_col: continue
                
            df = df.rename(columns={date_col: 'Date'})
            value_vars = [c for c in df.columns if c != 'Date']
            df_melted = df.melt(id_vars=['Date'], value_vars=value_vars, var_name='Hour_Str', value_name='Load')
            df_melted['Hour'] = df_melted['Hour_Str'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
            df_melted['Date'] = pd.to_datetime(df_melted['Date'])
            df_melted['tm'] = df_melted['Date'] + pd.to_timedelta(df_melted['Hour'], unit='h')
            
            if df_melted['Load'].dtype == object:
                df_melted['Load'] = df_melted['Load'].astype(str).str.replace(',', '').astype(float)
            
            all_dfs.append(df_melted[['tm', 'Load']])
            print(f"    - 로드 성공: {file_path}") # 성공 로그 추가
            
        except: pass

    if not all_dfs: return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['tm']).sort_values('tm').reset_index(drop=True)
    
    start_dt = pd.to_datetime(Config.START_DT)
    end_dt = pd.to_datetime(Config.END_DT) + pd.Timedelta(days=1)
    final_df = final_df[(final_df['tm'] >= start_dt) & (final_df['tm'] < end_dt)]
    
    print(f">>> [전력 데이터] 병합 완료: {len(final_df)}행")
    return final_df
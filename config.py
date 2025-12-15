import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ==========================================
    # [1] API 설정
    # ==========================================
    # 기상청: 기상관측 지상(종관, ASOS) 시간자료 조회
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    STN_ID = "108"  # 108: 서울 (전국 대표성을 위해 서울 사용)

    # 전력거래소: 시간별 전국 전력수요량
    POWER_API_KEY = os.getenv("SERVICE_KEY")

    # 데이터 수집 기간 (YYYYMMDD)
    START_DT = "20210101"
    END_DT = "20241231"

    # ==========================================
    # [2] 데이터 전처리 기준
    # ==========================================
    T_BASE_CDD = 24.0  # 냉방도일 기준온도
    T_BASE_HDD = 18.0  # 난방도일 기준온도

    # ==========================================
    # [3] 모델 하이퍼파라미터
    # ==========================================
    SEQ_LENGTH = 14       # 과거 14일 데이터를 보고 내일 예측
    HIDDEN_SIZE = 128     # LSTM 은닉층 크기
    NUM_LAYERS = 2       # LSTM 레이어 개수
    OUTPUT_SIZE = 1      # 출력 크기 (일일 최대 수요 1개)
    DROPOUT = 0.2
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 300
    TRAIN_SPLIT = 0.8    # 학습 데이터 비율 (80%)

    # GPU 사용 가능 여부
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
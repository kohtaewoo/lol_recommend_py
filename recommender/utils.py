import pandas as pd
import numpy as np
import os

# ✅ Riot API 지역 설정
REGION_KR = "https://kr.api.riotgames.com"
REGION_ACCOUNT = "https://americas.api.riotgames.com"

# ✅ API 헤더 (recommend.py에서 동적으로 세팅)
HEADERS = {}

# ✅ Z-score 기반 패널티 가중치 계산 함수
def load_zscore_penalty(excel_path, scale=1.0):
    """
    챔피언 출현 빈도를 기반으로 Z-score를 계산하고,
    이를 통해 흔한 챔피언일수록 패널티를 부여하는 가중치 반환.

    Parameters:
        excel_path (str): 엑셀 파일 경로
        scale (float): 패널티 스케일 (기본값: 1.0)

    Returns:
        dict: {champion_name: weight}
    """
    df = pd.read_excel(excel_path)
    
    if not {"Champion", "Count"}.issubset(df.columns):
        raise ValueError("엑셀 파일에는 'Champion', 'Count' 컬럼이 포함되어야 합니다.")

    counts = df.set_index("Champion")["Count"]
    std = counts.std()

    if std == 0:
        penalty_weights = pd.Series(1.0, index=counts.index)
    else:
        z_scores = (counts - counts.mean()) / std
        penalty_weights = 1 / (1 + scale * z_scores.clip(lower=0))
        
    return penalty_weights.to_dict()

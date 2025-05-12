import os
import json
import joblib
import pandas as pd
import requests

from recommender.utils import HEADERS, REGION_KR, REGION_ACCOUNT, load_zscore_penalty

# 환경 설정
REGION_KR = "https://kr.api.riotgames.com"
REGION_ACCOUNT = "https://americas.api.riotgames.com"

# 현재 파일 기준 base 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(DATA_DIR, "r_model")

# 파일 경로
master_path = os.path.join(DATA_DIR, "lol_champion_frequency_data", "master_user_champion_frequency_log1p.json")
user_path = os.path.join(DATA_DIR, "lol_champion_frequency_data", "user_champion_frequency_log1p.json")
role_path = os.path.join(DATA_DIR, "champion_roles.json")
top_path = os.path.join(DATA_DIR, "lol_line_list", "lol_top_champions.json")
mid_path = os.path.join(DATA_DIR, "lol_line_list", "lol_mid_champions.json")
jungle_path = os.path.join(DATA_DIR, "lol_line_list", "lol_jungle_champions.json")
bot_path = os.path.join(DATA_DIR, "lol_line_list", "lol_bot_champions.json")
support_path = os.path.join(DATA_DIR, "lol_line_list", "lol_support_champions.json")
user_count_path = os.path.join(DATA_DIR, "lol_champion_counts", "champion_counts.xlsx")
master_count_path = os.path.join(DATA_DIR, "lol_champion_counts", "master_champion_counts.xlsx")

# 챔피언 ID → 이름 매핑
def get_champion_mapping():
    version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    latest_version = requests.get(version_url).json()[0]
    champ_url = f"http://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    champ_data = requests.get(champ_url).json()["data"]
    return {int(v["key"]): k for k, v in champ_data.items()}

champion_id_to_name = get_champion_mapping()

# 역할군 정의
with open(role_path, "r", encoding="utf-8") as f:
    role_data = json.load(f)

with open(top_path, "r") as f: top_data = json.load(f)
with open(mid_path, "r") as f: mid_data = json.load(f)
with open(jungle_path, "r") as f: jungle_data = json.load(f)
with open(bot_path, "r") as f: bot_data = json.load(f)
with open(support_path, "r") as f: support_data = json.load(f)

champion_positions = {}
for data, role in [(top_data, "Top"), (mid_data, "Mid"), (jungle_data, "Jungle"), (bot_data, "Bottom"), (support_data, "Support")]:
    for champ in data:
        champion_positions.setdefault(champ, []).append(role)

# 데이터셋 로딩
with open(master_path, "r", encoding="utf-8") as f: master_data = json.load(f)
with open(user_path, "r", encoding="utf-8") as f: user_data = json.load(f)

df_master = pd.DataFrame.from_dict(master_data, orient="index").fillna(0)
df_user = pd.DataFrame.from_dict(user_data, orient="index").fillna(0)
common_champs = list(set(df_master.columns) & set(df_user.columns))

# 모델 로딩
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
df_master_pca = pd.read_json(os.path.join(MODEL_DIR, "df_master_pca.json"), orient="index")
df_user_scaled = pd.read_json(os.path.join(MODEL_DIR, "df_user_scaled.json"), orient="index")

# 클러스터 설명
cluster_descriptions = {
    0: ("예측불가 플레이형", "정형화된 틀보다, 나만의 방식으로 흐름을 바꾸는 걸 좋아하는 타입입니다."),
    1: ("손에 익은 몰입형", "익숙한 챔프만 잡으면 몰입 끝. 감각에 몸을 맡기는 스타일입니다."),
    2: ("직진 전투형", "복잡한 거 싫고, 정면에서 붙어서 이기는 맛으로 플레이합니다."),
    3: ("감각 조율형", "상황에 따라 자연스럽게 조율하며, 흐름을 부드럽게 이어갑니다."),
    4: ("기본기 중심형", "무리하지 않고 안정적으로, 팀의 중심을 잡아주는 플레이를 선호합니다."),
    5: ("순간 포착형", "‘지금이야!’ 싶은 타이밍을 놓치지 않고 빠르게 전투를 열어버립니다."),
    6: ("그림 설계형", "머릿속에 시나리오를 그리고, 흐름을 계획하며 이끄는 걸 좋아합니다."),
    7: ("지속 퍼포먼스형", "조용히, 묵묵히, 꾸준히. 언뜻 안 보여도 결과로 말하는 타입입니다."),
    8: ("틈새 연결형", "눈에 띄진 않아도 빈틈을 메우고 팀을 자연스럽게 이어주는 연결고리입니다."),
    9: ("전장 전체형", "눈앞의 싸움보다 전체 맵을 보며, 흐름을 설계하고 움직입니다.")
}

# 패널티 가중치 로딩
penalty_user = load_zscore_penalty(user_count_path)
penalty_master = load_zscore_penalty(master_count_path)

import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity

# 🔧 utils에서 설정만 가져옴
from recommender.utils import REGION_ACCOUNT, REGION_KR

# 🔧 나머지 모델/데이터는 __init__.py에서 가져옴
from recommender import (
    scaler, pca, df_master_pca, df_user_scaled,
    df_master, df_user, common_champs,
    role_data, champion_id_to_name,
    champion_positions, penalty_user, penalty_master
)


def recommend_by_riot_id(riot_id, headers):
    name, tag = map(quote, riot_id.split("#", 1))
    input_vector = pd.Series(0.0, index=common_champs)

    # ✅ PUUID 가져오기
    res1 = requests.get(
        f"{REGION_ACCOUNT}/riot/account/v1/accounts/by-riot-id/{name}/{tag}",
        headers=headers
    )
    if res1.status_code != 200:
        raise ValueError(f"Account API failed: {res1.status_code} - {res1.text}")

    data1 = res1.json()
    puuid = data1.get("puuid")
    if not puuid:
        raise ValueError("Missing 'puuid' in Riot API response")

    # ✅ 챔피언 숙련도 상위 5개
    res2 = requests.get(
        f"{REGION_KR}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}",
        headers=headers
    )
    if res2.status_code != 200:
        raise ValueError(f"Mastery API failed: {res2.status_code} - {res2.text}")

    mastery_data = res2.json()[:5]

    input_champions = [
        champion_id_to_name.get(ch["championId"])
        for ch in mastery_data
        if champion_id_to_name.get(ch["championId"])
    ]

    log_scores = [np.log1p(ch["championPoints"]) for ch in mastery_data]
    total_score = sum(log_scores)

    for ch, score in zip(mastery_data, log_scores):
        champ = champion_id_to_name.get(ch["championId"])
        if champ in input_vector:
            input_vector[champ] = score / total_score

    # ✅ 역할군 추출
    input_roles = set()
    for champ in input_champions:
        roles = role_data.get(champ, {})
        if roles.get("main_role"):
            input_roles.add(roles["main_role"])
        if roles.get("sub_role"):
            input_roles.add(roles["sub_role"])

    # ✅ 유저 벡터 스케일링 및 PCA
    input_df = pd.DataFrame([input_vector])[scaler.feature_names_in_]
    input_scaled = pd.DataFrame(
        scaler.transform(input_df), columns=scaler.feature_names_in_
    )
    input_pca = pca.transform(input_scaled)

    # ✅ 유사도 계산
    sim_master = cosine_similarity(input_pca, df_master_pca.values).flatten()
    sim_user = cosine_similarity(input_scaled, df_user_scaled.values).flatten()

    top_master = df_master.loc[
        df_master_pca.index[np.argsort(sim_master)[::-1][:50]]
    ].mean()
    top_user = df_user.iloc[np.argsort(sim_user)[::-1][:50]].mean()

    combined = {}
    for champ in set(top_master.index) | set(top_user.index):
        m_score = top_master.get(champ, 0) * penalty_master.get(champ, 1.0)
        u_score = top_user.get(champ, 0) * penalty_user.get(champ, 1.0)
        base = 0.6 * m_score + 0.4 * u_score

        bonus = 0.0
        roles = role_data.get(champ, {})
        if roles.get("main_role") in input_roles:
            bonus += 0.2
        if roles.get("sub_role") in input_roles:
            bonus += 0.1

        combined[champ] = base * (1 + bonus) + 7

    # ✅ 입력 챔피언 제외
    for champ in input_champions:
        combined.pop(champ, None)

    final = pd.Series(combined).sort_values(ascending=False).head(100)
    max_score = final.max()
    final_percentage = {
        champ: max(int(score / max_score * 100), 1)
        for champ, score in final.items()
    }

    # ✅ 라인별 추천 정리
    result = {"Top": [], "Jungle": [], "Mid": [], "Bottom": [], "Support": []}
    for champ, _ in final_percentage.items():
        for role in champion_positions.get(champ, []):
            result[role].append(champ)

    for role in result:
        result[role] = result[role][:5]

    return result, input_champions

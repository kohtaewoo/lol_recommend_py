import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from recommender import (
    REGION_KR, REGION_ACCOUNT,
    df_master, champion_id_to_name,
    cluster_descriptions
)

# ✅ 전역에서 클러스터링 모델 1회 학습
champ_counts = (df_master > 0).sum()
selected_champs = champ_counts[champ_counts >= 50].index
penalty_weights = 1 / (1 + 0.4 * ((champ_counts - champ_counts.mean()) / champ_counts.std()).clip(lower=0))

X = normalize(df_master[selected_champs], norm='l2')
scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X)

pca_c = PCA(n_components=0.9)
X_pca = pca_c.fit_transform(X_scaled)

kmeans_c = KMeans(n_clusters=10, n_init=20, random_state=42)
kmeans_c.fit(X_pca)


def predict_user_cluster(riot_id, headers):
    name, tag = map(quote, riot_id.split("#", 1))

    # ✅ Riot API 호출로 puuid 획득
    res1 = requests.get(
        f"{REGION_ACCOUNT}/riot/account/v1/accounts/by-riot-id/{name}/{tag}",
        headers=headers
    )
    res1.raise_for_status()
    puuid = res1.json().get("puuid")
    if not puuid:
        raise ValueError("Missing 'puuid' in Riot API response")

    # ✅ 숙련도 정보 요청
    res2 = requests.get(
        f"{REGION_KR}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}",
        headers=headers
    )
    res2.raise_for_status()
    mastery_data = res2.json()[:5]

    # ✅ 입력 벡터 생성
    input_vector = pd.Series(0.0, index=selected_champs)
    total = 0
    for ch in mastery_data:
        champ_name = champion_id_to_name.get(ch["championId"])
        if champ_name in input_vector:
            score = np.log1p(ch["championPoints"])
            input_vector[champ_name] = score
            total += score

    if total == 0:
        return 0, "정체불명", "정보 부족"

    input_vector /= total
    input_vector *= penalty_weights

    # ✅ 모델 추론만 수행
    input_normalized = normalize([input_vector], norm='l2')
    input_scaled = scaler_c.transform(input_normalized)
    input_pca = pca_c.transform(input_scaled)
    cluster_id = kmeans_c.predict(input_pca)[0]

    title, desc = cluster_descriptions[cluster_id]
    return cluster_id, title, desc

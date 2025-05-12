import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from recommender.utils import HEADERS
from recommender import (
    REGION_KR, REGION_ACCOUNT,
    df_master, champion_id_to_name,
    cluster_descriptions
)


def predict_user_cluster(riot_id):
    name, tag = map(quote, riot_id.split("#", 1))

    # Riot API 호출
    res1 = requests.get(f"{REGION_ACCOUNT}/riot/account/v1/accounts/by-riot-id/{name}/{tag}", headers=HEADERS)
    puuid = res1.json()["puuid"]

    res2 = requests.get(f"{REGION_KR}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}", headers=HEADERS)
    mastery_data = res2.json()[:5]

    # 상위 5챔프 기반 사용자 벡터 생성
    champ_counts = (df_master > 0).sum()
    selected_champs = champ_counts[champ_counts >= 50].index
    penalty_weights = 1 / (1 + 0.4 * ((champ_counts - champ_counts.mean()) / champ_counts.std()).clip(lower=0))

    input_vector = pd.Series(0.0, index=selected_champs)
    total = 0
    for ch in mastery_data:
        champ_name = champion_id_to_name.get(ch["championId"])
        if champ_name in input_vector:
            score = np.log1p(ch["championPoints"])
            input_vector[champ_name] = score
            total += score

    if total == 0:
        return (0, "정체불명", "정보 부족")

    input_vector /= total
    input_vector *= penalty_weights

    # 정규화 후 클러스터링
    input_normalized = normalize([input_vector], norm='l2')
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(normalize(df_master[selected_champs], norm='l2'))

    pca_c = PCA(n_components=0.9)
    X_pca = pca_c.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=10, n_init=20, random_state=42)
    kmeans.fit(X_pca)

    input_scaled = scaler_c.transform(input_normalized)
    input_pca = pca_c.transform(input_scaled)

    cluster_id = kmeans.predict(input_pca)[0]
    title, desc = cluster_descriptions[cluster_id]
    return cluster_id, title, desc

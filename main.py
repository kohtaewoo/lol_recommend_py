from flask import Flask, request, jsonify
from recommender.recommend import recommend_by_riot_id
from recommender.cluster import predict_user_cluster, load_cluster_model
import requests
from urllib.parse import quote, unquote
import os
from dotenv import load_dotenv

# ✅ .env 파일 읽기
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
BACKEND_AUTH_KEY = os.getenv("BACKEND_AUTH_KEY")

# ✅ 클러스터링 모델 전역 로딩 (recommend는 내부에서 전역 사용 중)
cluster_model = load_cluster_model()

app = Flask(__name__)

# ✅ 공통 인증 함수
def is_authorized():
    return request.headers.get("X-Backend-Key") == BACKEND_AUTH_KEY


@app.route("/recommend", methods=["GET"])
def recommend():
    if not is_authorized():
        return jsonify({"error": "Unauthorized"}), 401

    riot_id = request.args.get("riotId")
    if not riot_id or not RIOT_API_KEY:
        return jsonify({"error": "riotId and Riot API key are required"}), 400

    try:
        riot_id = unquote(riot_id)
        headers = {"X-Riot-Token": RIOT_API_KEY}

        # ✅ 모델 전달 생략
        recommendations, input_champions = recommend_by_riot_id(riot_id, headers)
        cluster_id, cluster_title, cluster_desc = predict_user_cluster(riot_id, headers, cluster_model)

        result = {
            "riotId": riot_id,
            "clusterTitle": cluster_title,
            "clusterDesc": cluster_desc,
            "inputChampions": input_champions,
            "recommendations": {
                role: [str(champ) for champ in champs]
                for role, champs in recommendations.items()
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/check", methods=["GET"])
def check_user():
    if not is_authorized():
        return jsonify({"error": "Unauthorized"}), 401

    riot_id = request.args.get("riotId")
    if not riot_id or not RIOT_API_KEY:
        return jsonify({"error": "riotId and Riot API key are required"}), 400

    try:
        riot_id = unquote(riot_id)
        name, tag = map(quote, riot_id.split("#", 1))
        url = f"{REGION_ACCOUNT}/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
        headers = {"X-Riot-Token": RIOT_API_KEY}
        res = requests.get(url, headers=headers)

        return jsonify({"exists": res.status_code == 200})
    except:
        return jsonify({"exists": False})

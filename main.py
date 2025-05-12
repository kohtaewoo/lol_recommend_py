from flask import Flask, request, jsonify
from recommender.recommend import recommend_by_riot_id
from recommender.cluster import predict_user_cluster
import requests
from urllib.parse import quote, unquote
import os
from dotenv import load_dotenv

# ✅ .env 파일 읽기
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    riot_id = request.args.get("riotId")

    if not riot_id or not RIOT_API_KEY:
        return jsonify({"error": "riotId and Riot API key are required"}), 400

    try:
        riot_id = unquote(riot_id)

        # ✅ 요청마다 headers 생성
        headers = {"X-Riot-Token": RIOT_API_KEY}

        # ✅ headers 인자 모두 전달
        recommendations, input_champions = recommend_by_riot_id(riot_id, headers)
        cluster_id, cluster_title, cluster_desc = predict_user_cluster(riot_id, headers)

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
    riot_id = request.args.get("riotId")

    if not riot_id or not RIOT_API_KEY:
        return jsonify({"error": "riotId and Riot API key are required"}), 400

    try:
        riot_id = unquote(riot_id)
        name, tag = map(quote, riot_id.split("#", 1))
        url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
        headers = {"X-Riot-Token": RIOT_API_KEY}
        res = requests.get(url, headers=headers)

        return jsonify({"exists": res.status_code == 200})
    except:
        return jsonify({"exists": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render 호환
    app.run(host="0.0.0.0", port=port)

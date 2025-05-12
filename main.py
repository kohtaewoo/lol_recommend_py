from flask import Flask, request, jsonify
import requests
from urllib.parse import quote, unquote
import os
from dotenv import load_dotenv
from recommender.recommend import recommend_by_riot_id
from recommender.cluster import predict_user_cluster

# ✅ Load .env for local, ignored on Render
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
        recommendations, input_champions = recommend_by_riot_id(riot_id)
        cluster_id, cluster_title, cluster_desc = predict_user_cluster(riot_id)

        result = {
            "riotId": riot_id,
            "clusterTitle": cluster_title,
            "clusterDesc": cluster_desc,
            "inputChampions": input_champions,
            "recommendations": {
                role: [str(champ) for champ in champs] for role, champs in recommendations.items()
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

import requests
from urllib.parse import quote

from recommender.utils import REGION_ACCOUNT, REGION_KR


def get_puuid(riot_id: str, headers: dict) -> str:
    """
    Riot ID를 통해 PUUID를 가져온다.
    """
    name, tag = map(quote, riot_id.split("#", 1))
    url = f"{REGION_ACCOUNT}/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()["puuid"]


def get_top_mastery(puuid: str, headers: dict, count: int = 5) -> list:
    """
    PUUID를 통해 상위 챔피언 숙련도 데이터를 가져온다.
    """
    url = f"{REGION_KR}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()[:count]

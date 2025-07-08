# 🧠 LOL 챔피언 추천 시스템 (`lol_recommend_py`)

## 📌 프로젝트 개요

본 프로젝트는 **롤(League of Legends)** 유저 데이터를 기반으로 PCA와 클러스터링을 활용하여  
유저 성향에 맞는 챔피언을 추천하는 시스템입니다.

> 사용자의 챔피언 플레이 로그 데이터를 기반으로 클러스터링 후,  
> 각 클러스터의 선호 챔피언을 분석하여 개인화된 추천을 수행합니다.

---

## 📂 폴더 구조

```
lol_recommend_py/
├── data/                             # 원시 데이터 폴더
│   ├── cluster_profiles.json         # 클러스터별 대표 챔피언
│   ├── user_cluster_map.json         # 유저별 클러스터 매핑
│   ├── lol_champion_counts/          # 챔피언 사용 횟수 엑셀
│   │   ├── champion_counts.xlsx
│   │   └── master_champion_counts.xlsx
│   ├── lol_champion_frequency_data/  # log1p 변환된 챔피언 사용 비율
│   │   ├── master_user_champion_frequency_log1p.json
│   │   └── user_champion_frequency_log1p.json
│   ├── lol_line_list/                # 포지션별 챔피언 목록
│   │   ├── lol_top_champions.json
│   │   ├── lol_jungle_champions.json
│   │   ├── lol_mid_champions.json
│   │   ├── lol_bot_champions.json
│   │   └── lol_support_champions.json
│
├── r_model/                          # 모델 및 전처리된 데이터
│   ├── df_master_pca.json
│   ├── df_user_scaled.json
│   ├── pca.pkl
│   ├── scaler.pkl
│   └── champion_roles.json
│
├── recommender/                      # 추천 시스템 코드
│   ├── __init__.py
│   ├── cluster.py                    # 클러스터링 로직
│   ├── fetch_data.py                # 데이터 로딩 함수
│   ├── recommend.py                 # 추천 알고리즘 메인 로직
│   ├── utils.py                     # 유틸 함수
│   └── main.py                      # 실행 진입점
│
├── requirements.txt                 # Python 의존성 명세
└── README.md
```

---

## 🛠️ 주요 기능

### ✅ 유저-챔피언 사용 분석
- 사용 빈도를 기반으로 log1p 스케일링 및 표준화 적용
- PCA를 통해 차원 축소 후 클러스터링 수행

### ✅ 클러스터 기반 추천
- 유저를 클러스터에 할당
- 해당 클러스터 내 인기 챔피언 추천

### ✅ 역할(포지션) 필터링
- 유저가 선호하는 라인 (탑, 정글 등) 기반 챔피언만 추천 가능

---

## 🚀 실행 방법

```bash
# 가상환경 권장
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 의존성 설치
pip install -r requirements.txt

# 추천 실행
python recommender/main.py
```

---

## 🔍 requirements.txt 주요 패키지

```
pandas
numpy
scikit-learn
```

---

## 🧩 향후 개선 방향

- 웹 기반 인터페이스 추가
- 실제 유저 전적 API 연동
- 실시간 클러스터링 갱신 기능 추가

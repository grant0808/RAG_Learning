# LangChain & LangGraph 실습 튜토리얼 🚀

이 프로젝트는 **LangChain**과 **LangGraph**를 처음 시작하는 분들을 위한 단계별 학습 환경 및 소스 코드 예제를 포함하고 있습니다.

## 📁 폴더 구조

```text
├── .env                         # 환경 변수 설정 (API Token)
├── README.md                    # 본 파일
├── pyproject.toml               # 의존성 및 패키지 설정
└── tutorials/
    ├── 01_langchain_basics.py   # LCEL 및 기본적인 LangChain 사용법
    ├── 02_rag_basics.py         # 임베딩 데이터베이스를 활용한 기본 RAG 파이프라인
    ├── 03_langgraph_basics.py   # LangGraph의 핵심 개념(State, Node, Edge) 실습
    └── 04_advanced_rag_graph.py # Advanced RAG (Corrective RAG) 구현
```

---

## 🛠️ 사전 설정 및 실행 방법

### 1. 가상환경 및 의존성 활성화
프로젝트 루트 폴더에서 아래 명령어를 실행하여 가상환경이 정상적으로 활성화되었는지 확인합니다. (의존성 패키지는 이미 가상환경에 추가되었습니다.)
```bash
# 가상환경 활성화 (맥OS/Linux 기준)
source .venv/bin/activate
```

### 2. 환경 변수 설정
`.env` 파일에 Hugging Face 토큰 및 LangSmith 설정이 정상적으로 들어있는지 확인해 주세요.
* `HUGGINGFACEHUB_API_TOKEN`: Hugging Face API 서비스를 호출하기 위한 인증 토큰
* `LANGSMITH_API_KEY` (선택): 실행 과정을 시각화하여 디버깅할 수 있게 해 주는 LangSmith 토큰

---

## 🏃 튜토리얼 실행하기

각 튜토리얼 스크립트를 터미널에서 순서대로 실행하며 코드를 공부해 보세요.

### Step 1: LangChain 기초
LCEL(LangChain Expression Language)을 이용하여 LLM을 호출하고 응답을 파싱하는 기본 구조를 배웁니다.
```bash
python tutorials/01_langchain_basics.py
```

### Step 2: RAG 파이프라인 기초
Chroma DB를 구축하고 텍스트 문서를 임베딩해 저장한 다음, 사용자 질문에 매칭되는 문서를 검색해 대답하게 만듭니다.
```bash
python tutorials/02_rag_basics.py
```

### Step 3: LangGraph 기초
노드(Node), 조건부 에지(Conditional Edge), 상태(State)의 흐름을 갖는 순환 그래프 모델의 기초 구조를 파악합니다.
```bash
python tutorials/03_langgraph_basics.py
```

### Step 4: Corrective RAG (CRAG) 구현
벡터 DB에 찾고자 하는 정보가 없거나 부족할 때, LangGraph 워크플로우 내에서 스스로 이를 판별하고 DuckDuckGo 외부 검색을 통해 보완하여 완벽한 답변을 만드는 고급 RAG 구현법을 배웁니다.
```bash
python tutorials/04_advanced_rag_graph.py
```

---

## 📚 추가 학습 가이드 안내
자세한 로드맵 설명 및 이론적 바탕은 생성된 [학습 가이드](file:///Users/hwangbyeonghyeon/.gemini/antigravity-cli/brain/c001a400-fb41-4ae5-944e-4b558280d3e8/learning_guide.md) 문서를 참고해 주시기 바랍니다.
# RAG_Learning

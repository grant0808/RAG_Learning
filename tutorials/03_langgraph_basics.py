import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# 1. 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# 2. Graph State (상태) 정의
# 그래프 내의 노드들이 서로 데이터를 주고받기 위한 공유 객체 타입 정의
class GraphState(TypedDict):
    question: str       # 사용자의 원래 질문
    is_safe: bool       # 질문이 부적절한지 여부 (필터링 검사 결과)
    generation: str     # 최종 생성된 답변

# 3. Nodes (노드) 정의
# 각 노드는 현재 상태(State)를 입력으로 받고, 변경된 상태(State dict)를 반환하는 '함수'입니다.

def check_safety(state: GraphState) -> dict:
    """사용자 질문에 유해하거나 테스트 불가능한 단어가 있는지 단순 검사하는 노드"""
    print("🧹 [노드 1] check_safety: 질문의 적합성을 확인 중입니다...")
    question = state["question"].lower()
    
    # 예시: '폭탄', '해킹', '비속어' 단어가 포함되면 안전하지 않다고 판정
    bad_words = ["폭탄", "해킹", "비속어", "공격"]
    is_safe = True
    for word in bad_words:
        if word in question:
            is_safe = False
            break
            
    return {"is_safe": is_safe}

def generate_answer(state: GraphState) -> dict:
    """일반적인 답변을 생성하는 노드"""
    print("🧠 [노드 2] generate_answer: 답변을 생성하는 중입니다...")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문에 정확하고 친절하게 한국어로 대답해 주세요."),
        ("user", "{question}")
    ])
    
    chain = prompt | model | StrOutputParser()
    generation = chain.invoke({"question": question})
    
    return {"generation": generation}

def reject_request(state: GraphState) -> dict:
    """부적절한 질문에 대해 거절 메시지를 생성하는 노드"""
    print("🚫 [노드 3] reject_request: 요청을 거절하는 중입니다...")
    return {"generation": "죄송합니다. 제공해주신 질문에는 부적절하거나 처리할 수 없는 단어가 포함되어 있습니다. 다른 질문을 해주세요."}

# 4. Conditional Edge (조건부 에지) 결정 함수 정의
# 이전 노드의 실행 결과(State)를 보고 다음으로 이동할 노드의 이름을 문자열로 리턴합니다.
def route_question(state: GraphState) -> str:
    print("🛤️ [라우터] route_question: 상태를 기반으로 다음 단계 결정 중...")
    if state["is_safe"]:
        return "generate_answer"
    else:
        return "reject_request"

# 5. Graph 빌드
# State 타입을 기반으로 그래프 선언
workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("check_safety", check_safety)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("reject_request", reject_request)

# 흐름 및 연결(Edge) 설정
# 진입점 설정 (그래프가 시작할 때 첫 실행할 노드)
workflow.set_entry_point("check_safety")

# 조건부 에지 추가
# check_safety 노드가 종료된 후, route_question 조건함수를 실행하여 "generate_answer" 또는 "reject_request" 노드로 이동하게 합니다.
workflow.add_conditional_edges(
    "check_safety",
    route_question,
    {
        "generate_answer": "generate_answer",
        "reject_request": "reject_request"
    }
)

# 각각의 끝 노드에서 종료 상태(END)로 이동하도록 에지 설정
workflow.add_edge("generate_answer", END)
workflow.add_edge("reject_request", END)

# 그래프 컴파일 (동작 가능한 실행 프로그램으로 빌드)
app = workflow.compile()

# 6. 실행 및 테스트
if __name__ == "__main__":
    # 케이스 1: 안전한 질문
    safe_input = {"question": "대한민국의 수도는 어디야?"}
    print(f"\n--- 🧪 테스트 1: 안전한 질문 ({safe_input['question']}) ---")
    result = app.invoke(safe_input)
    print(f"✨ 최종 결과:\n{result['generation']}")

    # 케이스 2: 부적절한 단어가 포함된 질문
    unsafe_input = {"question": "해킹 공격 기법 중 SQL Injection에 대해 알려줘."}
    print(f"\n--- 🧪 테스트 2: 부적절한 단어가 포함된 질문 ({unsafe_input['question']}) ---")
    result = app.invoke(unsafe_input)
    print(f"✨ 최종 결과:\n{result['generation']}")

import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# 1. 환경 변수 및 설정 로드
load_dotenv()

print("🔌 모델 및 인프라 구축 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# DuckDuckGo 웹 검색기 객체 생성
web_search_tool = DuckDuckGoSearchRun()

# Chroma DB 초기화 (기존 02번 튜토리얼 데이터가 있으면 읽어옴)
db_path = "./chroma_db_tutorial"
vector_store = Chroma(
    collection_name="tutorial_collection",
    embedding_function=embeddings,
    persist_directory=db_path
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 2. Graph State 정의
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_search: bool  # 웹 검색이 필요한지 여부 플래그
    generation: str

# 3. Nodes (노드) 함수 구현

def retrieve(state: GraphState) -> dict:
    """벡터 DB에서 사용자 질문과 유사한 문서를 검색하여 상태를 업데이트합니다."""
    print("🔍 [Node] retrieve: 벡터 데이터베이스에서 관련 문서 검색 중...")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "web_search": False}


def grade_documents(state: GraphState) -> dict:
    """검색된 문서들의 관련성을 판별합니다. 관련 없는 문서는 걸러내고, 유효 문서가 부족할 시 웹 검색 플래그를 설정합니다."""
    print("📝 [Node] grade_documents: 검색된 문서들의 연관성 평가 중...")
    question = state["question"]
    documents = state["documents"]
    
    # 평가를 위한 프롬프트 구성
    # LLM이 'yes' 혹은 'no'로만 대답하도록 엄격하게 지시
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 문서 평가자입니다. 주어진 문서가 질문과 핵심적인 연관성이 있는지 확인해 주세요. 관련이 있다면 'yes', 전혀 연관성이 없다면 'no'라고 답하세요. 다른 설명 없이 오직 'yes' 또는 'no'로만 답변하세요."),
        ("user", "질문: {question}\n\n문서:\n{doc_content}")
    ])
    
    grader_chain = prompt | model | StrOutputParser()
    
    filtered_docs = []
    need_web_search = False
    
    for doc in documents:
        # LLM을 호출하여 연관성 판단
        score = grader_chain.invoke({"question": question, "doc_content": doc.page_content}).strip().lower()
        print(f"   - 문서 평가 결과: {score}")
        if "yes" in score:
            filtered_docs.append(doc)
        else:
            continue
            
    # 유효한 문서가 하나도 없는 경우 웹 검색 필요성을 True로 세팅
    if len(filtered_docs) == 0:
        need_web_search = True
        
    return {"documents": filtered_docs, "web_search": need_web_search}


def web_search(state: GraphState) -> dict:
    """로컬 데이터가 부족한 경우 DuckDuckGo를 통해 외부 검색을 시도하여 문서에 병합합니다."""
    print("🌐 [Node] web_search: 부족한 정보 보완을 위해 DuckDuckGo 웹 검색 수행 중...")
    question = state["question"]
    documents = state["documents"]
    
    # 웹 검색 수행
    try:
        search_result = web_search_tool.invoke(question)
        web_doc = Document(page_content=search_result, metadata={"source": "duckduckgo_search"})
        documents.append(web_doc)
        print("   - 웹 검색 성공 및 결과 통합 완료.")
    except Exception as e:
        print(f"   - 웹 검색 중 에러 발생: {e}")
        
    return {"documents": documents}


def generate(state: GraphState) -> dict:
    """최종 문서를 바인딩하여 최종 답변을 생성합니다."""
    print("🧠 [Node] generate: 최종 취합된 정보를 기반으로 답변 생성 중...")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(doc.page_content for doc in documents)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "제공된 정보(Context)를 바탕으로 사용자의 질문에 한국어로 친절하게 답변해 주세요. 정보가 충분하지 않거나 누락된 상태라면 솔직히 모른다고 이야기해 주세요."),
        ("user", "Context:\n{context}\n\n질문: {question}")
    ])
    
    chain = prompt | model | StrOutputParser()
    generation = chain.invoke({"question": question, "context": context})
    
    return {"generation": generation}


# 4. Conditional Edge 결정 함수
def decide_to_generate(state: GraphState) -> str:
    """웹 검색 노드로 분기할지, 바로 최종 생성 노드로 이동할지를 결정합니다."""
    print("🛤️ [Edge] decide_to_generate: 분기 노선 결정 중...")
    if state["web_search"]:
        print("   -> 웹 검색 필요 판정. web_search 노드로 이동합니다.")
        return "web_search"
    else:
        print("   -> 충분한 정보가 검색됨. generate 노드로 이동합니다.")
        return "generate"


# 5. Graph 빌드
workflow = StateGraph(GraphState)

# 노드 정의 등록
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# 흐름 및 엣지 구성
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 조건부 분기 추가 (grade_documents 실행 후 결정)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

# 웹 검색 완료 후 생성으로 연결
workflow.add_edge("web_search", "generate")
# 생성 완료 후 그래프 종료
workflow.add_edge("generate", END)

# 컴파일
app = workflow.compile()

# 6. 실행 및 테스트
if __name__ == "__main__":
    # 질문 1: 로컬 DB에 등록해 둔 내용에 대한 질문 (RAG Basics 문서 참고)
    # 02번을 실행한 적이 있다면 'LangGraph의 특징'에 대해 로컬 DB에 지식이 있어 바로 Generate로 가게 됩니다.
    query_local = "LangGraph가 무엇이고 어떤 특징을 가지고 있는지 설명해 줘."
    print(f"\n================ 🧪 테스트 1: 로컬 지식 기반 쿼리 ================")
    print(f"질문: {query_local}")
    result_local = app.invoke({"question": query_local})
    print(f"\n✨ 최종 답변:\n{result_local['generation']}")
    
    # 질문 2: 로컬 DB에 없는 새로운 최신 상식 질문 (웹 검색 필수 트리거)
    # '2026년 동계 올림픽 개최지'와 같이 로컬 DB에 없는 데이터는 웹 검색을 시도하여 답변합니다.
    query_web = "2026년 동계 올림픽 개최지는 어디인가요?"
    print(f"\n================ 🧪 테스트 2: 외부 웹 검색 트리거 쿼리 ================")
    print(f"질문: {query_web}")
    result_web = app.invoke({"question": query_web})
    print(f"\n✨ 최종 답변:\n{result_web['generation']}")

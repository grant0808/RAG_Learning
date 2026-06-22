import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드
load_dotenv()

# 2. 임베딩 모델 및 LLM 설정
print("🔌 임베딩 모델(sentence-transformers) 및 LLM 초기화 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.2, # RAG의 경우 답변의 일관성을 위해 온도를 낮춤
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# 3. 로컬 벡터스토어 구성
# 튜토리얼용 로컬 Chroma DB 디렉토리 지정
db_path = "./chroma_db_tutorial"
vector_store = Chroma(
    collection_name="tutorial_collection",
    embedding_function=embeddings,
    persist_directory=db_path
)

# 4. 임시 데이터 인덱싱 (Indexing)
# 임의의 도메인 지식 문서를 정의합니다.
sample_docs = [
    Document(
        page_content="LangChain은 LLM(대형 언어 모델) 애플리케이션 개발을 모듈식으로 지원하는 프레임워크입니다. 체인(Chain), 에이전트(Agent), 프롬프트(Prompt) 등의 컴포넌트를 제공합니다.",
        metadata={"source": "langchain_doc"}
    ),
    Document(
        page_content="LangGraph는 다중 에이전트 워크플로우를 그래프 기반 상태 머신으로 구현할 수 있도록 돕는 LangChain의 확장 라이브러리입니다. 순환(Cyclic) 그래프를 지원하는 것이 특징입니다.",
        metadata={"source": "langgraph_doc"}
    ),
    Document(
        page_content="RAG(검색 증강 생성)는 모델이 학습하지 않은 외부 데이터를 검색(Retrieve)하여, 그 정보를 기반으로 정확한 답변을 생성(Generate)하는 기술입니다.",
        metadata={"source": "rag_doc"}
    )
]

print("📦 문서 청킹 및 임베딩 벡터 저장 중...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(sample_docs)

# 벡터스토어에 문서 추가
vector_store.add_documents(documents=splits)
print(f"✅ {len(splits)}개의 텍스트 조각이 벡터 데이터베이스에 저장되었습니다.")

# 5. Retriever 및 RAG 체인 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "아래 제공된 컨텍스트(Context)만을 사용하여 질문에 답변해 주세요. 답을 모른다면 솔직하게 모른다고 대답하세요. 한국어로 답변해 주세요.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# 문서 직렬화 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 파이프라인 구성
# 1) 입력 질문에 매칭되는 문서를 검색하여 'context'에 주입
# 2) 질문을 그대로 'question'에 주입
# 3) 프롬프트 템플릿에 데이터 포매팅
# 4) LLM 호출 및 결과 파싱
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    query = "LangGraph의 주요 특징은 무엇이며 LangChain과 어떤 차이가 있나요?"
    print(f"\n❓ 질문: {query}")
    
    print("\n🔍 RAG 시스템 검색 및 답변 생성 시작...")
    response = rag_chain.invoke(query)
    
    print(f"\n✨ 생성된 답변:\n{response}")

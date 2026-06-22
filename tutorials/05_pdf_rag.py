import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드 (.env 파일에서 API 키 등을 가져옴)
load_dotenv()

# 2. PDF 로드 (PyPDFLoader 사용)
print("📄 PDF 파일 로드 중...")
pdf_path = "2412.15605v2.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF 파일을 찾을 수 없습니다: '{pdf_path}'.\n"
        f"프로젝트 루트 디렉토리에 해당 PDF 파일이 있는지 확인해 주세요."
    )

# PyPDFLoader는 PDF의 각 페이지를 하나의 Document 객체로 로드합니다.
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"✅ PDF 로드 완료. 총 페이지 수: {len(docs)}페이지")

# 3. 텍스트 분할 (Chunking)
print("✂️ 문서 텍스트 분할(Chunking) 중...")
# RecursiveCharacterTextSplitter를 사용하여 문맥이 끊기지 않도록 적절한 크기로 자릅니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # 각 청크의 최대 문자 수
    chunk_overlap=200,    # 청크 간 겹치는 문자 수 (맥락 유지용)
    add_start_index=True  # 각 청크가 원본 문서에서 시작하는 인덱스 저장
)
splits = text_splitter.split_documents(docs)
print(f"✅ 문서를 {len(splits)}개의 텍스트 조각(Chunk)으로 분할했습니다.")

# 4. 임베딩 모델 및 로컬 벡터 DB 설정
print("🔌 임베딩 모델(sentence-transformers) 및 LLM 초기화 중...")
# 허깅페이스 임베딩 모델 설정 (로컬에서 벡터 변환 수행)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# 벡터 DB가 저장될 로컬 디렉토리 경로
db_path = "./chroma_pdf_db"

# 기존에 빌드된 DB가 있다면 삭제하여 중복 방지 (실습 편의용)
if os.path.exists(db_path):
    print("🧹 이전 벡터 DB 디렉토리 초기화 중...")
    shutil.rmtree(db_path)

# Chroma 벡터 DB 구성 및 문서 벡터 저장
vector_store = Chroma(
    collection_name="pdf_rag_collection",
    embedding_function=embeddings,
    persist_directory=db_path
)

print("📦 분할된 문서를 벡터 DB에 임베딩하여 저장 중...")
vector_store.add_documents(documents=splits)
print(f"✅ 벡터 데이터베이스 저장 완료 (경로: {db_path})")

# 5. LLM 및 검색기(Retriever) 설정
# HuggingFace Hub 기반 LLMEndpoint 설정
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.3,  # 온도를 미세하게 조정하여 유연한 텍스트 생성 유도
    max_new_tokens=512,
    repetition_penalty=1.15,  # 무한 반복 출력 방지를 위한 패널티 설정
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# 검색기(Retriever) 설정: 유사한 청크 3개 추출
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# [⚠️ Llama-3-8B 모델 사용 시 참고사항]
# 현재 사용하고 있는 meta-llama/Meta-Llama-3-8B-Instruct 모델은 영어 위주로 학습되어 있어,
# 영어 문서 컨텍스트(Context)가 대량으로 주어지면 한국어 답변 지시사항을 무시하고 영어로 답변을 작성하는 성향이 있습니다.
# 
# 해결 방법:
# 1) GPT-4, Gemini, 또는 Cohere Command-R과 같은 다국어 지원 대규모 LLM API를 사용합니다.
# 2) RAG 대상 문서를 한글 PDF 문서로 교체하여 실행하면 정상적으로 한국어로 답변이 생성됩니다.
# 3) 2단계(영문 답변 생성 -> 번역 모델로 한글 번역) 파이프라인을 구축합니다.

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "제공된 컨텍스트(Context)를 바탕으로 사용자의 질문에 한국어로 성실하게 답변해 주세요. "
               "논문의 사실을 바탕으로 작성하고, 한국어로 번역해서 명확히 설명해 주세요. "
               "정보가 부족하다면 모른다고 답변해 주세요.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# 문서 포맷팅 함수 (검색된 문서들을 가독성 있게 연결)
def format_docs(docs):
    formatted = []
    for doc in docs:
        # PyPDFLoader는 metadata에 'page' 번호를 0-indexed로 추가해 줍니다.
        page_num = doc.metadata.get("page", 0) + 1
        formatted.append(f"[Page {page_num}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# 6. RAG 체인 파이프라인 구성 (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 7. 실행 예제
if __name__ == "__main__":
    # 논문의 핵심 주제 중 하나인 CAG에 대해 한국어로 질문
    query = "Cache-Augmented Generation (CAG)이 무엇이며 RAG와 어떻게 다른가요?"
    print(f"\n❓ 사용자 질문: {query}")
    
    print("\n🔍 RAG 시스템 검색 및 답변 생성 시작...")
    response = rag_chain.invoke(query)
    
    print(f"\n✨ 생성된 답변:\n{response}")

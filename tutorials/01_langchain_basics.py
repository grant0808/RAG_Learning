import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드 (.env)
load_dotenv()

# 2. LLM 초기화 (Hugging Face Endpoint API 활용)
# HuggingFaceEndpoint를 사용하여 서버리스 API로 Llama-3-8B-Instruct 모델을 호출합니다.
print("🤖 Hugging Face LLM 모델을 로드 중입니다...")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# ChatHuggingFace로 감싸 대화형(Chat) 인터페이스 인터페이스로 포맷합니다.
model = ChatHuggingFace(llm=llm)

# 3. Prompt Template 정의
# 대화형 프롬프트를 구성합니다.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 인공지능 분야의 전문 조수입니다. 질문에 쉽고 명확하게 한국어로 답변해 주세요."),
    ("user", "{question}")
])

# 4. LCEL (LangChain Expression Language) 체인 구성
# prompt -> model -> output_parser의 흐름을 '|' 연산자로 연결합니다.
chain = prompt_template | model | StrOutputParser()

# 5. 실행
if __name__ == "__main__":
    question = "LangChain의 LCEL(LangChain Expression Language)이 무엇이고 왜 사용하는지 한 문장으로 설명해 줘."
    print(f"\n❓ 질문: {question}")
    
    print("\n⏳ 답변 생성 중...")
    try:
        response = chain.invoke({"question": question})
        print(f"\n✨ 답변:\n{response}")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        print("💡 .env 파일에 올바른 HUGGINGFACEHUB_API_TOKEN이 설정되었는지 확인해 주세요.")

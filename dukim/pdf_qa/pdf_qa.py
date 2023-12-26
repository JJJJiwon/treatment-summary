from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# 1. pdf load
print('1. pdf load')
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
loader = PyPDFLoader('pdf/transformer_1706.03762.pdf')
pages = loader.load_and_split()

# https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500, # 500 글자
    chunk_overlap=20, # 20자 중복
    length_function=len, # 길이 함수
    is_separator_regex=False, # 정규표현식 사용 여부
)

texts = text_splitter.split_documents(pages)

# 2. embedding vector로 바꾸기
print('2. embedding vector로 바꾸기')
# https://huggingface.co/intfloat/multilingual-e5-large
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# 3. vector db에 저장하기
print('3. vector db에 저장하기')
# https://python.langchain.com/docs/integrations/vectorstores/chroma
db = Chroma.from_documents(texts, embeddings_model)

# 4. pc에 있는 llama2 모델 인스턴스 생성하기
print('4. llm 인스턴스 생성')
# https://python.langchain.com/docs/integrations/providers/ctransformers
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
config = {'context_length': 1024}
llm = CTransformers(model="./llm/llama-2-7b-chat.ggmlv3.q2_K.bin", model_type="llama", config=config)

# 5. 질의 응답
print('5. 질의 응답')
# https://js.langchain.com/docs/modules/chains/popular/vector_db_qa
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

question = 'What is a transformer?'
print(f'Q : {question}')
result = qa_chain({'query': question})
print(f"A : {result['result']}")

question = 'What is a self-attention?'
print(f'Q : {question}')
result = qa_chain({'query': question})
print(f"A : {result['result']}")

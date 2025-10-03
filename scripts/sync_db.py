import os
import sys
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
# 공식 문서에 따라 VectorSearchVectorStore를 가져옵니다.
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import VectorSearchVectorStore
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document

# --- 1. 환경 변수 확인 ---
# ENDPOINT_ID를 다시 추가합니다.
required_env_vars = [
    "POSTGRES_CONNECTION_STRING", "GCP_PROJECT_ID", "VERTEX_AI_INDEX_ID",
    "VERTEX_AI_ENDPOINT_ID", "VERTEX_AI_REGION"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# --- 2. 설정 및 초기화 ---
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INDEX_ID = os.getenv("VERTEX_AI_INDEX_ID")
ENDPOINT_ID = os.getenv("VERTEX_AI_ENDPOINT_ID")
REGION = os.getenv("VERTEX_AI_REGION")

assert POSTGRES_CONNECTION_STRING and PROJECT_ID and INDEX_ID and ENDPOINT_ID and REGION

RECORD_MANAGER_NAMESPACE = f"vertexai/{INDEX_ID}"

print("Initializing clients and managers...")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

# Vertex AI Vector Search 초기화 (from_components 메서드를 사용하는 올바른 방식)
vectorstore = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    location=REGION,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    embedding=embeddings,
)

# SQL Record Manager 초기화
record_manager = SQLRecordManager(
    RECORD_MANAGER_NAMESPACE, db_url=POSTGRES_CONNECTION_STRING
)
record_manager.create_schema()
print("Initialization complete.")

# --- 3. 함수 정의 ---
def get_changed_files():
    """Git diff를 통해 변경된 파일 목록을 가져옵니다."""
    print("Checking for changed files using git diff...")
    result = subprocess.run(
        ['git', 'diff', '--name-status', 'HEAD~1', 'HEAD'],
        capture_output=True, text=True, check=True
    ).stdout
    
    files = {'added_modified': [], 'deleted': []}
    for line in result.strip().split('\n'):
        if not line: continue
        try:
            status, file_path = line.split('\t')
            if file_path.endswith(('.cc', '.h', '.cpp')):
                if status in ['A', 'M']:
                    files['added_modified'].append(file_path)
                elif status == 'D':
                    files['deleted'].append(file_path)
        except ValueError:
            print(f"Warning: Could not parse git diff line: {line}")
    print(f"Found files: {files}")
    return files

def load_documents_from_files(file_paths):
    """파일 경로 리스트로부터 LangChain Document 객체들을 로드합니다."""
    print(f"Loading {len(file_paths)} documents from files...")
    docs = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": file_path}))
        except FileNotFoundError:
            print(f"Warning: File not found during loading, skipping: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    return docs

# --- 4. 메인 로직 실행 ---
if __name__ == "__main__":
    changed_files = get_changed_files()
    
    docs_to_process = load_documents_from_files(changed_files['added_modified'])
    if docs_to_process:
        cpp_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=1500,
            chunk_overlap=150
        )
        split_docs = cpp_splitter.split_documents(docs_to_process)

        if split_docs:
            print(f"Indexing {len(split_docs)} chunks for {len(docs_to_process)} files...")
            index(
                split_docs,
                record_manager,
                vectorstore,
                cleanup="full",
                source_id_key="source",
            )
            print("Indexing for added/modified files finished.")
        else:
            print("No new chunks to index after splitting.")
    else:
        print("No added or modified files to process.")

    if changed_files['deleted']:
        print(f"Deleting records for {len(changed_files['deleted'])} files...")
        record_manager.delete_keys(changed_files['deleted'])
        print("Deletion of records finished.")
    else:
        print("No deleted files to process.")
        
    print("Synchronization script finished successfully.")


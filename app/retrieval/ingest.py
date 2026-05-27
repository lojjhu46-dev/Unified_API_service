"""文档摄取模块"""

import os
import uuid
from typing import List
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


def load_document(file_path: str) -> list:
    """加载文档"""
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        return loader.load()
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        raise


def split_documents(documents: list) -> list:
    """切分文档"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )
        return splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"切分文档失败: {e}")
        raise


def ingest_file(file_path: str) -> dict:
    """摄取文件到向量存储"""
    from app.retrieval.vector_store import add_documents

    docs = load_document(file_path)
    chunks = split_documents(docs)
    chunk_count = add_documents(chunks)

    document_id = str(uuid.uuid4())[:12]
    filename = os.path.basename(file_path)

    logger.info(f"文档摄取完成: {filename}, {chunk_count}个切块")

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": chunk_count,
    }


def validate_file_extension(filename: str) -> bool:
    """验证文件扩展名"""
    allowed_extensions = {".pdf", ".txt"}
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


def save_uploaded_file(content: bytes, filename: str) -> str:
    """保存上传的文件"""
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}_{filename}")
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

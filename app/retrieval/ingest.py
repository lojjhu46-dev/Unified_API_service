"""文档摄取模块"""

import os
import re
import uuid
from pathlib import Path
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_ALLOWED_EXTENSIONS = {".pdf", ".txt"}


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


def ingest_file(file_path: str, original_filename: str | None = None) -> dict:
    """摄取文件到向量存储"""
    from app.retrieval.vector_store import add_documents

    document_id = str(uuid.uuid4())[:12]
    stored_filename = os.path.basename(file_path)
    safe_original_filename = sanitize_filename(original_filename or stored_filename)

    docs = load_document(file_path)
    chunks = split_documents(docs)
    for index, chunk in enumerate(chunks):
        # 这些元数据会写入 Chroma，后续可用于来源追踪、权限过滤和文档删除。
        metadata = chunk.metadata if isinstance(getattr(chunk, "metadata", None), dict) else {}
        chunk.metadata = {
            **metadata,
            "document_id": document_id,
            "original_filename": safe_original_filename,
            "stored_filename": stored_filename,
            "chunk_index": index,
        }
    chunk_count = add_documents(chunks)

    logger.info(f"文档摄取完成: {safe_original_filename}, {chunk_count}个切块")

    return {
        "document_id": document_id,
        "filename": safe_original_filename,
        "chunks": chunk_count,
    }


def validate_file_extension(filename: str) -> bool:
    """验证文件扩展名"""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in _ALLOWED_EXTENSIONS


def sanitize_filename(filename: str | None) -> str:
    """清理上传文件名，防止路径穿越和非法字符进入保存路径。"""
    raw_name = filename or ""
    # UploadFile.filename 可能携带 Windows 或 Unix 路径分隔符，这里统一只保留 basename。
    base_name = raw_name.replace("\\", "/").split("/")[-1]
    ext = os.path.splitext(base_name)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        ext = ".txt"

    stem = os.path.splitext(base_name)[0]
    stem = _INVALID_FILENAME_CHARS.sub("_", stem).strip(" .")
    if not stem:
        stem = "upload"
    return f"{stem}{ext}"


def save_uploaded_file(content: bytes, filename: str) -> str:
    """保存上传的文件"""
    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_filename(filename)
    file_path = (upload_dir / f"{uuid.uuid4()}_{safe_filename}").resolve()
    if upload_dir not in file_path.parents:
        raise ValueError("上传文件路径非法")

    with open(file_path, "wb") as f:
        f.write(content)
    return str(file_path)

"""检索模块测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.retrieval.retriever import Retriever
from app.config import settings
from app.retrieval.ingest import (
    ingest_file,
    load_document,
    load_xlsx_document,
    sanitize_filename,
    save_uploaded_file,
    validate_file_extension,
)
from app.schemas import SourceItem


@pytest.fixture
def mock_retriever():
    retriever = Retriever()
    retriever._use_chroma = False
    retriever.refresh = MagicMock(return_value=False)
    return retriever


@pytest.fixture(autouse=True)
def mock_keyword_search_by_default():
    with patch("app.retrieval.vector_store.async_keyword_search", new=AsyncMock(return_value=[])):
        yield


@pytest.mark.asyncio
async def test_mock_search(mock_retriever):
    results = await mock_retriever.search("测试问题", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, SourceItem) for r in results)
    assert results[0].score > results[1].score


def test_validate_file_extension():
    assert validate_file_extension("test.pdf") is True
    assert validate_file_extension("test.txt") is True
    assert validate_file_extension("test.docx") is True
    assert validate_file_extension("test.xlsx") is True
    assert validate_file_extension("test.doc") is False
    assert validate_file_extension("test.docm") is False
    assert validate_file_extension("test.xls") is False


def test_sanitize_filename_removes_paths_and_invalid_chars():
    assert sanitize_filename("../evil.txt") == "evil.txt"
    assert sanitize_filename(r"C:\tmp\evil.txt") == "evil.txt"
    assert sanitize_filename("../evil.docx") == "evil.docx"
    assert sanitize_filename("../bad.xlsx") == "bad.xlsx"
    assert sanitize_filename('bad:name?.pdf') == "bad_name_.pdf"


def test_load_document_uses_docx_loader():
    with patch("langchain_community.document_loaders.Docx2txtLoader") as mock_loader:
        instance = mock_loader.return_value
        instance.load.return_value = ["doc"]

        result = load_document("test.docx")

    mock_loader.assert_called_once_with("test.docx")
    assert result == ["doc"]


def test_load_document_uses_xlsx_loader():
    with patch("app.retrieval.ingest.load_xlsx_document", return_value=["sheet doc"]) as mock_loader:
        result = load_document("test.xlsx")

    mock_loader.assert_called_once_with("test.xlsx")
    assert result == ["sheet doc"]


def test_load_xlsx_document_extracts_sheet_text(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "SheetA"
    sheet.append(["名称", "数量"])
    sheet.append(["苹果", 3])
    workbook.save(path)

    docs = load_xlsx_document(str(path))

    assert len(docs) == 1
    assert "名称\t数量" in docs[0].page_content
    assert "苹果\t3" in docs[0].page_content
    assert docs[0].metadata["sheet"] == "SheetA"


def test_extract_query_terms_keeps_weather_focus_term():
    retriever = Retriever()

    terms = retriever._extract_query_terms("查找关于天气的历年毕业选题")

    assert "天气" in terms
    assert "历年" in terms
    assert "毕业选题" in terms


def test_extract_query_terms_expands_code_targets():
    retriever = Retriever()

    sort_terms = retriever._extract_query_terms("排序算法的python代码片段")
    sine_terms = retriever._extract_query_terms("绘制一条正弦曲线，曲线的样式是默认的")

    assert "快速排序" in sort_terms
    assert "quick_sort" in sort_terms
    assert "quicksort" in sort_terms
    assert "np.sin" in sine_terms
    assert "ax.plot" in sine_terms


def test_merge_search_results_keeps_better_duplicate_score():
    retriever = Retriever()
    doc = MagicMock(
        page_content="以下是Python实现快速排序算法的示例代码：\ndef quick_sort(arr):\n    return arr",
        metadata={"source": "quick.txt", "document_id": "doc1", "chunk_index": 0},
    )

    results = retriever._merge_search_results([(doc, 1.5)], [(doc, -20)])

    assert results == [(doc, -20)]


def test_save_uploaded_file_stays_inside_upload_dir(tmp_path):
    old_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path)
    try:
        for filename in ["../evil.txt", r"C:\tmp\evil.txt", "bad:name?.txt"]:
            saved_path = save_uploaded_file(b"hello", filename)
            resolved = __import__("pathlib").Path(saved_path).resolve()
            assert tmp_path.resolve() in resolved.parents
            assert ".." not in resolved.name
            assert ":" not in resolved.name
    finally:
        settings.upload_dir = old_upload_dir


@patch("app.retrieval.vector_store.add_documents")
@patch("app.retrieval.ingest.split_documents")
@patch("app.retrieval.ingest.load_document")
def test_ingest_file(mock_load, mock_split, mock_add):
    mock_load.return_value = [MagicMock()]
    chunk1 = MagicMock(metadata={"page": 1})
    chunk2 = MagicMock(metadata={"page": 2})
    mock_split.return_value = [chunk1, chunk2]
    mock_add.return_value = 2

    result = ingest_file("/tmp/stored.pdf", original_filename="../test.pdf")
    assert result["chunks"] == 2
    assert "document_id" in result
    assert "filename" in result
    assert result["filename"] == "test.pdf"
    assert chunk1.metadata["document_id"] == result["document_id"]
    assert chunk2.metadata["document_id"] == result["document_id"]
    assert chunk1.metadata["chunk_index"] == 0
    assert chunk2.metadata["chunk_index"] == 1
    assert chunk1.metadata["original_filename"] == "test.pdf"
    assert chunk1.metadata["stored_filename"] == "stored.pdf"
    assert chunk1.metadata["knowledge_base_type"] == "enterprise"
    assert chunk1.metadata["owner_open_id"] == ""
    assert chunk1.metadata["chat_id"] == ""
    assert chunk1.metadata["channel"] == "api"


@patch("app.retrieval.vector_store.add_documents")
@patch("app.retrieval.ingest.split_documents")
@patch("app.retrieval.ingest.load_document")
def test_ingest_file_personal_metadata(mock_load, mock_split, mock_add):
    mock_load.return_value = [MagicMock()]
    chunk = MagicMock(metadata={})
    mock_split.return_value = [chunk]
    mock_add.return_value = 1

    result = ingest_file(
        "/tmp/stored.txt",
        original_filename="我的资料.txt",
        knowledge_base_type="personal",
        owner_open_id="ou_test123",
        chat_id="oc_test789",
        channel="feishu",
    )

    assert result["chunks"] == 1
    assert chunk.metadata["knowledge_base_type"] == "personal"
    assert chunk.metadata["owner_open_id"] == "ou_test123"
    assert chunk.metadata["chat_id"] == "oc_test789"
    assert chunk.metadata["channel"] == "feishu"


@pytest.mark.asyncio
async def test_chroma_search_fallback_to_mock():
    retriever = Retriever()
    retriever._use_chroma = False
    retriever.refresh = MagicMock(return_value=False)
    results = await retriever.search("测试", top_k=2)
    assert len(results) > 0


def test_normalize_distance_score():
    retriever = Retriever()
    assert retriever._normalize_distance_score(0) == 1.0
    assert 0.0 < retriever._normalize_distance_score(1.5) < 1.0
    assert retriever._normalize_distance_score(None) == 0.0
    assert 0.0 <= retriever._normalize_distance_score(-1) <= 1.0


@pytest.mark.asyncio
async def test_chroma_exception_falls_back_to_mock():
    retriever = Retriever()
    retriever._use_chroma = True
    with patch.object(retriever, "_chroma_search", new=AsyncMock(side_effect=RuntimeError("boom"))):
        results = await retriever.search("测试", top_k=2)

    assert len(results) == 2
    assert all(item.source_type == "knowledge_base" for item in results)


@pytest.mark.asyncio
async def test_chroma_search_keeps_full_content_for_llm_and_short_snippet():
    retriever = Retriever()
    long_content = "开头" * 120 + "先小康后大同 阶级特征 高度的生产力"
    doc = MagicMock(
        page_content=long_content,
        metadata={"source": "4.pdf", "document_id": "doc1", "chunk_index": 0},
    )

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(doc, 0)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search("历史影响", top_k=1)

    assert len(results) == 1
    assert results[0].snippet == long_content[:settings.rag_display_snippet_chars]
    assert "先小康后大同" in results[0].content
    assert "content" not in results[0].model_dump()


@pytest.mark.asyncio
async def test_chroma_search_expands_neighbor_chunks():
    retriever = Retriever()
    doc = MagicMock(
        page_content="5. 儒家大同思想的历史影响",
        metadata={"source": "4.pdf", "document_id": "doc1", "chunk_index": 1},
    )
    chunks = [
        {"content": "上一段", "metadata": {"chunk_index": 0}},
        {"content": "5. 儒家大同思想的历史影响", "metadata": {"chunk_index": 1}},
        {"content": "进步性、阶级特征、缺乏可实现途径", "metadata": {"chunk_index": 2}},
    ]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(doc, 0)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=chunks),
    ):
        results = await retriever._chroma_search("历史影响", top_k=1)

    assert "上一段" in results[0].content
    assert "进步性、阶级特征、缺乏可实现途径" in results[0].content


@pytest.mark.asyncio
async def test_chroma_search_expands_more_chunks_for_global_questions():
    retriever = Retriever()
    doc = MagicMock(
        page_content="儒家大同思想的基本内容主要包括以下五个方面。",
        metadata={"source": "4.pdf", "document_id": "doc1", "chunk_index": 5},
    )
    chunks = [
        {"content": "社会制度：全民公有", "metadata": {"chunk_index": 5}},
        {"content": "管理制度：选贤与能", "metadata": {"chunk_index": 6}},
        {"content": "人际关系：讲信修睦", "metadata": {"chunk_index": 7}},
        {"content": "社会保障：人人得其所", "metadata": {"chunk_index": 8}},
        {"content": "劳动态度：各尽其力", "metadata": {"chunk_index": 9}},
    ]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(doc, 0)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=chunks),
    ):
        results = await retriever._chroma_search("儒家大同思想的主要内容", top_k=1)

    assert "全民公有" in results[0].content
    assert "选贤与能" in results[0].content
    assert "讲信修睦" in results[0].content
    assert "人人得其所" in results[0].content
    assert "各尽其力" in results[0].content


@pytest.mark.asyncio
async def test_chroma_search_neighbor_failure_uses_original_chunk():
    retriever = Retriever()
    doc = MagicMock(
        page_content="原始命中切块",
        metadata={"source": "4.pdf", "document_id": "doc1", "chunk_index": 1},
    )

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(doc, 0)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        results = await retriever._chroma_search("历史影响", top_k=1)

    assert results[0].content == "原始命中切块"


@pytest.mark.asyncio
async def test_enterprise_scope_excludes_personal_sources():
    retriever = Retriever()
    enterprise_doc = MagicMock(
        page_content="企业知识",
        metadata={"source": "enterprise.txt", "knowledge_base_type": "enterprise"},
    )
    legacy_doc = MagicMock(
        page_content="旧企业知识",
        metadata={"source": "legacy.txt"},
    )
    personal_doc = MagicMock(
        page_content="个人知识",
        metadata={"source": "personal.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(enterprise_doc, 0), (legacy_doc, 0.1), (personal_doc, 0.2)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search("测试", top_k=5, knowledge_scope=["enterprise"])

    titles = [item.title for item in results]
    assert "enterprise.txt" in titles
    assert "legacy.txt" in titles
    assert "personal.txt" not in titles


@pytest.mark.asyncio
async def test_personal_scope_only_returns_matching_owner():
    retriever = Retriever()
    owner_doc = MagicMock(
        page_content="当前用户个人知识",
        metadata={"source": "mine.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )
    other_doc = MagicMock(
        page_content="其他用户个人知识",
        metadata={"source": "other.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_other"},
    )
    enterprise_doc = MagicMock(
        page_content="企业知识",
        metadata={"source": "enterprise.txt", "knowledge_base_type": "enterprise"},
    )

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(return_value=[(owner_doc, 0), (other_doc, 0.1), (enterprise_doc, 0.2)]),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "测试",
            top_k=5,
            knowledge_scope=["personal"],
            owner_open_id="ou_test123",
        )

    assert [item.title for item in results] == ["mine.txt"]


@pytest.mark.asyncio
async def test_personal_scope_falls_back_when_owner_filter_misses():
    retriever = Retriever()
    personal_doc = MagicMock(
        page_content="Python实现快速排序算法\ndef quick_sort(arr):\n    return arr",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_ingest_owner"},
    )

    async def fake_search(_query, k=5, metadata_filter=None):
        if metadata_filter == {"owner_open_id": "ou_request_user"}:
            return []
        if metadata_filter == {"knowledge_base_type": "personal"}:
            return [(personal_doc, 0)]
        return []

    with patch.object(settings, "personal_kb_strict_owner_filter", False), patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "Python实现快速排序算法",
            top_k=5,
            knowledge_scope=["personal"],
            owner_open_id="ou_request_user",
        )

    assert [item.title for item in results] == ["personal_quick_sort.txt"]


@pytest.mark.asyncio
async def test_personal_scope_reranks_exact_code_hit_before_vector_noise():
    retriever = Retriever()
    noise_docs = [
        MagicMock(
            page_content=f"个人知识库无关内容 {index}",
            metadata={"source": f"noise_{index}.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_ingest_owner"},
        )
        for index in range(3)
    ]
    quick_sort_doc = MagicMock(
        page_content="以下是Python实现快速排序算法的示例代码：\ndef quick_sort(arr):\n    return arr",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_ingest_owner"},
    )

    async def fake_search(_query, k=5, metadata_filter=None):
        if metadata_filter == {"owner_open_id": "ou_request_user"}:
            return []
        if metadata_filter == {"knowledge_base_type": "personal"}:
            return [(doc, index * 0.01) for index, doc in enumerate(noise_docs)] + [(quick_sort_doc, 1.5)]
        return []

    with patch.object(settings, "personal_kb_strict_owner_filter", False), patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "个人知识库检索，找出“Python实现快速排序算法”的python代码片段",
            top_k=3,
            knowledge_scope=["personal"],
            owner_open_id="ou_request_user",
        )

    assert results[0].title == "personal_quick_sort.txt"


@pytest.mark.asyncio
async def test_personal_sort_algorithm_returns_quick_sort_before_unrelated_chunks():
    retriever = Retriever()
    noise_doc = MagicMock(
        page_content="用户：安装复杂吗？需要专业人员吗？\n客服：基础安装非常简单。",
        metadata={"source": "personal_noise.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )
    quick_sort_doc = MagicMock(
        page_content="以下是Python实现快速排序算法的示例代码：\n\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )

    async def fake_similarity_search(_query, k=5, metadata_filter=None):
        return [(noise_doc, 0), (quick_sort_doc, 1.5)]

    async def fake_keyword_search(terms, k=5, metadata_filter=None):
        assert "quick_sort" in terms
        return [(quick_sort_doc, -20)]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_similarity_search),
    ), patch(
        "app.retrieval.vector_store.async_keyword_search",
        new=AsyncMock(side_effect=fake_keyword_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "“排序算法”的python代码片段",
            top_k=3,
            knowledge_scope=["personal"],
            owner_open_id="ou_test123",
        )

    assert results[0].title == "personal_quick_sort.txt"
    assert "def quick_sort" in results[0].snippet


@pytest.mark.asyncio
async def test_personal_scope_strict_owner_filter_disables_fallback():
    retriever = Retriever()
    personal_doc = MagicMock(
        page_content="Python实现快速排序算法",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_ingest_owner"},
    )

    async def fake_search(_query, k=5, metadata_filter=None):
        if metadata_filter == {"owner_open_id": "ou_request_user"}:
            return []
        if metadata_filter == {"knowledge_base_type": "personal"}:
            return [(personal_doc, 0)]
        return []

    with patch.object(settings, "personal_kb_strict_owner_filter", True), patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "Python实现快速排序算法",
            top_k=5,
            knowledge_scope=["personal"],
            owner_open_id="ou_request_user",
        )

    assert results == []


@pytest.mark.asyncio
async def test_combined_scope_merges_enterprise_and_current_owner_personal():
    retriever = Retriever()
    enterprise_doc = MagicMock(
        page_content="企业知识",
        metadata={"source": "enterprise.txt", "knowledge_base_type": "enterprise"},
    )
    owner_doc = MagicMock(
        page_content="当前用户个人知识",
        metadata={"source": "mine.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )
    other_doc = MagicMock(
        page_content="其他用户个人知识",
        metadata={"source": "other.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_other"},
    )

    async def fake_search(_query, k=5, metadata_filter=None):
        if metadata_filter:
            return [(owner_doc, 0), (other_doc, 0.1)]
        return [(enterprise_doc, 0.2), (other_doc, 0.3)]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "测试",
            top_k=5,
            knowledge_scope=["enterprise", "personal"],
            owner_open_id="ou_test123",
        )

    titles = [item.title for item in results]
    assert "enterprise.txt" in titles
    assert "mine.txt" in titles
    assert "other.txt" not in titles


@pytest.mark.asyncio
async def test_combined_scope_keeps_personal_result_when_enterprise_scores_dominate():
    retriever = Retriever()
    enterprise_docs = [
        MagicMock(
            page_content=f"绘制一条正弦曲线 企业片段 {index}",
            metadata={"source": f"enterprise_{index}.txt", "knowledge_base_type": "enterprise"},
        )
        for index in range(6)
    ]
    personal_doc = MagicMock(
        page_content="Python实现快速排序算法\ndef quick_sort(arr):\n    return arr",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )

    async def fake_search(_query, k=5, metadata_filter=None):
        if metadata_filter:
            return [(personal_doc, 0.9)]
        return [(doc, index * 0.01) for index, doc in enumerate(enterprise_docs)]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "绘制一条正弦曲线 Python实现快速排序算法",
            top_k=5,
            knowledge_scope=["enterprise", "personal"],
            owner_open_id="ou_test123",
        )

    titles = [item.title for item in results]
    assert "personal_quick_sort.txt" in titles
    assert any(title.startswith("enterprise_") for title in titles)


@pytest.mark.asyncio
async def test_combined_scope_searches_quoted_targets_separately():
    retriever = Retriever()
    animation_doc = MagicMock(
        page_content="制作一个圆点沿曲线运动的动画，并时刻显示圆点的坐标位置\n\ndef update(frame):\n    return point, coord_text",
        metadata={"source": "enterprise_animation.txt", "knowledge_base_type": "enterprise"},
    )
    quick_sort_doc = MagicMock(
        page_content="Python实现快速排序算法\ndef quick_sort(arr):\n    return quick_sort(left) + middle + quick_sort(right)",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )
    irrelevant_personal_doc = MagicMock(
        page_content="个人库无关内容",
        metadata={"source": "personal_other.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_test123"},
    )

    async def fake_search(query, k=5, metadata_filter=None):
        if "快速排序" in query and metadata_filter:
            return [(quick_sort_doc, 0)]
        if "圆点沿曲线运动" in query and metadata_filter is None:
            return [(animation_doc, 0)]
        if metadata_filter:
            return [(irrelevant_personal_doc, 1.5)]
        return []

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "企业知识库和个人知识库结合检索，找出“制作一个圆点沿曲线运动的动画，并时刻显示圆点的坐标位置”和“Python实现快速排序算法”的python代码片段",
            top_k=5,
            knowledge_scope=["enterprise", "personal"],
            owner_open_id="ou_test123",
        )

    titles = [item.title for item in results]
    assert "enterprise_animation.txt" in titles
    assert "personal_quick_sort.txt" in titles


@pytest.mark.asyncio
async def test_combined_scope_preserves_one_result_per_quoted_target_before_top_k_cutoff():
    retriever = Retriever()
    animation_docs = [
        MagicMock(
            page_content=f"制作一个圆点沿曲线运动的动画 片段 {index}",
            metadata={"source": f"animation_{index}.txt", "knowledge_base_type": "enterprise"},
        )
        for index in range(5)
    ]
    quick_sort_doc = MagicMock(
        page_content="Python实现快速排序算法\ndef quick_sort(arr):\n    return arr",
        metadata={"source": "personal_quick_sort.txt", "knowledge_base_type": "personal", "owner_open_id": "ou_request_user"},
    )

    async def fake_search(query, k=5, metadata_filter=None):
        if "圆点沿曲线运动" in query and metadata_filter is None:
            return [(doc, index * 0.01) for index, doc in enumerate(animation_docs)]
        if "快速排序" in query and metadata_filter:
            return [(quick_sort_doc, 0.8)]
        return []

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "找出“制作一个圆点沿曲线运动的动画”和“Python实现快速排序算法”的python代码片段",
            top_k=2,
            knowledge_scope=["enterprise", "personal"],
            owner_open_id="ou_request_user",
        )

    titles = [item.title for item in results]
    assert len(results) == 2
    assert any(title.startswith("animation_") for title in titles)
    assert "personal_quick_sort.txt" in titles


@pytest.mark.asyncio
async def test_enterprise_keyword_search_recalls_weather_xlsx_chunk():
    retriever = Retriever()
    noise_doc = MagicMock(
        page_content="Matplotlib绘图代码和人工智能伦理问题",
        metadata={"source": "noise.txt", "knowledge_base_type": "enterprise"},
    )
    weather_doc = MagicMock(
        page_content="郑丽华\t20125081042\t基于推荐算法的电子商城设计与实现\t基于BP神经网络的碳排放数据分析系统的设计与实现\t基于机器学习的天气数据分析与预测系统的实现",
        metadata={
            "source": "weather.xlsx",
            "knowledge_base_type": "enterprise",
            "document_id": "xlsx1",
            "chunk_index": 0,
        },
    )

    async def fake_similarity_search(_query, k=5, metadata_filter=None):
        return [(noise_doc, 0)]

    async def fake_keyword_search(terms, k=5, metadata_filter=None):
        assert "天气" in terms
        return [(weather_doc, -10)]

    with patch(
        "app.retrieval.vector_store.async_similarity_search",
        new=AsyncMock(side_effect=fake_similarity_search),
    ), patch(
        "app.retrieval.vector_store.async_keyword_search",
        new=AsyncMock(side_effect=fake_keyword_search),
    ), patch(
        "app.retrieval.vector_store.async_get_document_chunks",
        new=AsyncMock(return_value=[]),
    ):
        results = await retriever._chroma_search(
            "查找关于天气的历年毕业选题",
            top_k=3,
            knowledge_scope=["enterprise"],
        )

    assert results[0].title == "weather.xlsx"
    assert "天气数据分析" in results[0].snippet

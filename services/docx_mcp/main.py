"""DOCX MCP HTTP 服务

提供 DOCX 文档操作的 HTTP API 端点。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.mcp_common.auth import verify_api_key
from services.mcp_common.health import add_health_endpoint
from services.mcp_common.logging import setup_logging
from services.mcp_common.models import fail, ok
from services.mcp_common.path_security import validate_file_path, validate_output_path
from services.docx_mcp import operations

logger = setup_logging("docx_mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DOCX MCP 服务启动")
    yield
    logger.info("DOCX MCP 服务停止")


app = FastAPI(title="DOCX MCP Service", version="0.1.0", lifespan=lifespan)
add_health_endpoint(app, "docx_mcp")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理：确保所有响应都是 HTTP 200"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(content=fail("内部服务错误"))


async def _run_in_executor(func, *args) -> Any:
    """在线程池中运行同步函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


@app.post("/docx/read_structure")
async def read_structure(request: Request, body: dict) -> dict:
    """读取 DOCX 文档结构"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    try:
        result = await _run_in_executor(operations.read_structure, str(resolved))
        return ok(result)
    except Exception as e:
        logger.error(f"read_structure 失败: {e}")
        return fail(str(e))


@app.post("/docx/read_paragraph")
async def read_paragraph(request: Request, body: dict) -> dict:
    """读取指定段落的文本"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    index = body.get("index")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if index is None:
        return fail("缺少 index 参数")

    try:
        text = await _run_in_executor(operations.read_paragraph, str(resolved), int(index))
        return ok({"text": text})
    except IndexError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"read_paragraph 失败: {e}")
        return fail(str(e))


@app.post("/docx/replace_paragraph")
async def replace_paragraph(request: Request, body: dict) -> dict:
    """替换指定段落的文本"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    index = body.get("index")
    new_text = body.get("new_text", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if index is None:
        return fail("缺少 index 参数")

    try:
        await _run_in_executor(operations.replace_paragraph, str(resolved), int(index), new_text)
        return ok()
    except IndexError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"replace_paragraph 失败: {e}")
        return fail(str(e))


@app.post("/docx/delete_paragraph")
async def delete_paragraph(request: Request, body: dict) -> dict:
    """删除指定段落"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    index = body.get("index")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if index is None:
        return fail("缺少 index 参数")

    try:
        await _run_in_executor(operations.delete_paragraph, str(resolved), int(index))
        return ok()
    except IndexError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"delete_paragraph 失败: {e}")
        return fail(str(e))


@app.post("/docx/append_paragraph")
async def append_paragraph(request: Request, body: dict) -> dict:
    """在文档末尾追加段落"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    text = body.get("text", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    try:
        await _run_in_executor(operations.append_paragraph, str(resolved), text)
        return ok()
    except Exception as e:
        logger.error(f"append_paragraph 失败: {e}")
        return fail(str(e))


@app.post("/docx/read_table_cell")
async def read_table_cell(request: Request, body: dict) -> dict:
    """读取指定表格单元格文本"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    table_index = body.get("table_index")
    row = body.get("row")
    col = body.get("col")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if table_index is None or row is None or col is None:
        return fail("缺少 table_index、row 或 col 参数")

    try:
        text = await _run_in_executor(
            operations.read_table_cell,
            str(resolved),
            int(table_index),
            int(row),
            int(col),
        )
        return ok({"text": text})
    except (IndexError, ValueError) as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"read_table_cell 失败: {e}")
        return fail(str(e))


@app.post("/docx/replace_table_cell")
async def replace_table_cell(request: Request, body: dict) -> dict:
    """替换指定表格单元格文本"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    table_index = body.get("table_index")
    row = body.get("row")
    col = body.get("col")
    new_text = body.get("new_text", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if table_index is None or row is None or col is None:
        return fail("缺少 table_index、row 或 col 参数")

    try:
        await _run_in_executor(
            operations.replace_table_cell,
            str(resolved),
            int(table_index),
            int(row),
            int(col),
            new_text,
        )
        return ok()
    except (IndexError, ValueError) as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"replace_table_cell 失败: {e}")
        return fail(str(e))


@app.post("/docx/clear_table_cell")
async def clear_table_cell(request: Request, body: dict) -> dict:
    """清空指定表格单元格文本"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    table_index = body.get("table_index")
    row = body.get("row")
    col = body.get("col")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if table_index is None or row is None or col is None:
        return fail("缺少 table_index、row 或 col 参数")

    try:
        await _run_in_executor(
            operations.clear_table_cell,
            str(resolved),
            int(table_index),
            int(row),
            int(col),
        )
        return ok()
    except (IndexError, ValueError) as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"clear_table_cell 失败: {e}")
        return fail(str(e))


@app.post("/docx/save_copy")
async def save_copy(request: Request, body: dict) -> dict:
    """保存文档副本到指定路径"""
    verify_api_key(request)

    file_path = body.get("file_path", "")
    output_path = body.get("output_path", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    output_resolved, err = validate_output_path(output_path)
    if err:
        return fail(err)

    try:
        result = await _run_in_executor(operations.save_copy, str(resolved), str(output_resolved))
        return ok({"output_path": result})
    except Exception as e:
        logger.error(f"save_copy 失败: {e}")
        return fail(str(e))

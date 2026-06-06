"""XLSX MCP HTTP 服务

提供 XLSX 文档操作的 HTTP API 端点。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.mcp_common.health import add_health_endpoint
from services.mcp_common.logging import setup_logging
from services.mcp_common.models import fail, ok
from services.mcp_common.path_security import validate_file_path, validate_output_path
from services.xlsx_mcp import operations

logger = setup_logging("xlsx_mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("XLSX MCP 服务启动")
    yield
    logger.info("XLSX MCP 服务停止")


app = FastAPI(title="XLSX MCP Service", version="0.1.0", lifespan=lifespan)
add_health_endpoint(app, "xlsx_mcp")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局异常处理：确保所有响应都是 HTTP 200"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(content=fail("内部服务错误"))


async def _run_in_executor(func, *args) -> Any:
    """在线程池中运行同步函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


@app.post("/xlsx/read_structure")
async def read_structure(body: dict) -> dict:
    """读取 XLSX 工作簿结构"""
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


@app.post("/xlsx/read_cell")
async def read_cell(body: dict) -> dict:
    """读取指定单元格的值"""
    file_path = body.get("file_path", "")
    sheet = body.get("sheet", "")
    cell = body.get("cell", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if not sheet:
        return fail("缺少 sheet 参数")
    if not cell:
        return fail("缺少 cell 参数")

    try:
        value = await _run_in_executor(operations.read_cell, str(resolved), sheet, cell)
        return ok({"value": value})
    except KeyError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"read_cell 失败: {e}")
        return fail(str(e))


@app.post("/xlsx/modify_cell")
async def modify_cell(body: dict) -> dict:
    """修改指定单元格的值"""
    file_path = body.get("file_path", "")
    sheet = body.get("sheet", "")
    cell = body.get("cell", "")
    value = body.get("value", "")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if not sheet:
        return fail("缺少 sheet 参数")
    if not cell:
        return fail("缺少 cell 参数")

    try:
        await _run_in_executor(operations.modify_cell, str(resolved), sheet, cell, value)
        return ok()
    except KeyError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"modify_cell 失败: {e}")
        return fail(str(e))


@app.post("/xlsx/append_row")
async def append_row(body: dict) -> dict:
    """在指定工作表末尾追加一行"""
    file_path = body.get("file_path", "")
    sheet = body.get("sheet", "")
    values = body.get("values", [])

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if not sheet:
        return fail("缺少 sheet 参数")

    try:
        await _run_in_executor(operations.append_row, str(resolved), sheet, values)
        return ok()
    except KeyError as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"append_row 失败: {e}")
        return fail(str(e))


@app.post("/xlsx/delete_row")
async def delete_row(body: dict) -> dict:
    """删除指定行（1-based）"""
    file_path = body.get("file_path", "")
    sheet = body.get("sheet", "")
    row = body.get("row")

    resolved, err = validate_file_path(file_path)
    if err:
        return fail(err)

    if not sheet:
        return fail("缺少 sheet 参数")
    if row is None:
        return fail("缺少 row 参数")

    try:
        await _run_in_executor(operations.delete_row, str(resolved), sheet, int(row))
        return ok()
    except (KeyError, ValueError) as e:
        return fail(str(e))
    except Exception as e:
        logger.error(f"delete_row 失败: {e}")
        return fail(str(e))


@app.post("/xlsx/save_copy")
async def save_copy(body: dict) -> dict:
    """保存文档副本到指定路径"""
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

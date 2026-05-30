"""飞书 Channel Adapter 测试"""

import asyncio
import base64
import hashlib
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from app.channels.feishu import FeishuAdapter
from app.config import settings
from app.main import (
    app,
    decide_feishu_knowledge_scope,
    parse_feishu_card_action,
    pending_feishu_files,
    process_feishu_card_action,
    process_feishu_file_event,
    process_feishu_message,
    send_feishu_processing_heartbeat,
)


def _encrypted_body(body: dict, encrypt_key: str) -> dict:
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    iv = b"0123456789abcdef"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return {"encrypt": base64.b64encode(iv + encrypted).decode("utf-8")}


def _message_event(
    event_id: str = "event_1",
    message_id: str = "om_test456",
    text: str = "你好",
) -> dict:
    return {
        "header": {
            "event_type": "im.message.receive_v1",
            "token": "expected_token",
            "event_id": event_id,
            "uuid": "uuid_1",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_test123"},
                "sender_type": "user",
            },
            "message": {
                "message_id": message_id,
                "chat_id": "oc_test789",
                "chat_type": "p2p",
                "message_type": "text",
                "content": f'{{"text": "{text}"}}',
            },
        },
    }


def _file_event(
    event_id: str = "file_event_1",
    message_id: str = "om_file456",
    file_name: str = "资料.pdf",
    file_key: str = "file_key_1",
) -> dict:
    body = _message_event(event_id=event_id, message_id=message_id)
    body["event"]["message"]["message_type"] = "file"
    body["event"]["message"]["content"] = json.dumps(
        {"file_key": file_key, "file_name": file_name},
        ensure_ascii=False,
    )
    return body


def _card_action(
    pending_id: str,
    action: str = "confirm_save_personal_file",
    open_id: str = "ou_test123",
) -> dict:
    return {
        "header": {
            "event_type": "card.action.trigger",
            "token": "expected_token",
            "event_id": f"card_{pending_id}",
        },
        "event": {
            "operator": {"operator_id": {"open_id": open_id}},
            "context": {"open_chat_id": "oc_test789"},
            "action": {"value": {"action": action, "pending_id": pending_id}},
        },
    }


@pytest.fixture
def adapter():
    return FeishuAdapter()


@pytest.fixture
def client():
    return TestClient(app)


class TestChallengeVerification:
    """Challenge 验证测试"""

    def test_verify_challenge_schema_v2(self, adapter):
        body = {
            "type": "url_verification",
            "challenge": "test_challenge_123",
            "token": "test_token",
        }
        result = adapter.verify_challenge(body)
        assert result == {"challenge": "test_challenge_123"}

    def test_verify_challenge_schema_v1(self, adapter):
        body = {
            "challenge": "test_challenge_456",
            "token": "test_token",
        }
        result = adapter.verify_challenge(body)
        assert result == {"challenge": "test_challenge_456"}

    def test_verify_challenge_no_challenge(self, adapter):
        body = {"event": {"type": "message"}}
        result = adapter.verify_challenge(body)
        assert result is None


class TestTokenVerification:
    """Token 验证测试"""

    def test_verify_token_valid(self, adapter):
        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = "expected_token"
            body = {"token": "expected_token"}
            assert adapter.verify_token(body) is True

    def test_verify_token_invalid(self, adapter):
        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = "expected_token"
            body = {"token": "wrong_token"}
            assert adapter.verify_token(body) is False

    def test_verify_token_no_config(self, adapter):
        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = None
            body = {"token": "any_token"}
            assert adapter.verify_token(body) is False


class TestEventParsing:
    """事件解析测试"""

    def test_parse_message_event_v2(self, adapter):
        body = {
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "test_token",
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_test123"},
                    "sender_type": "user",
                },
                "message": {
                    "message_id": "om_test456",
                    "chat_id": "oc_test789",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": '{"text": "你好"}',
                },
            },
        }
        result = adapter.parse_event(body)
        assert result is not None
        assert result["open_id"] == "ou_test123"
        assert result["chat_id"] == "oc_test789"
        assert result["message_id"] == "om_test456"
        assert result["text"] == "你好"
        assert result["dedupe_key"] == "om_test456"

    def test_parse_message_event_v2_dedupe_prefers_event_id(self, adapter):
        body = _message_event(event_id="event_123", message_id="om_456")
        result = adapter.parse_event(body)
        assert result["event_id"] == "event_123"
        assert result["uuid"] == "uuid_1"
        assert result["dedupe_key"] == "event_123"

    def test_parse_message_event_v2_group(self, adapter):
        body = {
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "test_token",
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_user123"},
                    "sender_type": "user",
                },
                "message": {
                    "message_id": "om_msg456",
                    "chat_id": "oc_group789",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text": "@_user_1 今天天气怎么样"}',
                    "mentions": [
                        {
                            "key": "@_user_1",
                            "id": {"open_id": "ou_bot"},
                            "name": "测试机器人",
                            "mentioned_type": "bot",
                        },
                    ],
                },
            },
        }
        result = adapter.parse_event(body)
        assert result is not None
        assert result["text"] == "今天天气怎么样"

    def test_parse_non_text_message(self, adapter):
        body = {
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "test_token",
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_test123"},
                    "sender_type": "user",
                },
                "message": {
                    "message_id": "om_test456",
                    "chat_id": "oc_test789",
                    "chat_type": "p2p",
                    "message_type": "image",
                    "content": '{"image_key": "img_xxx"}',
                },
            },
        }
        result = adapter.parse_event(body)
        assert result is None

    def test_parse_missing_required_fields(self, adapter):
        body = _message_event()
        body["event"]["sender"]["sender_id"].pop("open_id")
        assert adapter.parse_event(body) is None

    def test_parse_unknown_event(self, adapter):
        body = {
            "header": {
                "event_type": "unknown.event",
                "token": "test_token",
            },
            "event": {},
        }
        result = adapter.parse_event(body)
        assert result is None

    def test_parse_file_message_event_v2(self, adapter):
        result = adapter.parse_event(_file_event())
        assert result is not None
        assert result["event_kind"] == "file"
        assert result["open_id"] == "ou_test123"
        assert result["chat_id"] == "oc_test789"
        assert result["message_id"] == "om_file456"
        assert result["file_key"] == "file_key_1"
        assert result["file_name"] == "资料.pdf"
        assert result["dedupe_key"] == "file_event_1"


class TestSessionId:
    """Session ID 生成测试"""

    def test_generate_session_id_stable(self, adapter):
        session_id1 = adapter.generate_session_id("ou_123", "oc_456")
        session_id2 = adapter.generate_session_id("ou_123", "oc_456")
        assert session_id1 == session_id2

    def test_generate_session_id_different(self, adapter):
        session_id1 = adapter.generate_session_id("ou_123", "oc_456")
        session_id2 = adapter.generate_session_id("ou_789", "oc_456")
        assert session_id1 != session_id2

    def test_generate_session_id_length(self, adapter):
        session_id = adapter.generate_session_id("ou_123", "oc_456")
        assert len(session_id) == 12


class TestDedupe:
    """事件去重测试"""

    def test_dedupe_first_seen_then_duplicate(self, adapter):
        assert adapter.mark_event_seen("event_1") is True
        assert adapter.mark_event_seen("event_1") is False

    def test_dedupe_expired_key_can_be_seen_again(self, adapter):
        adapter._dedupe_ttl_seconds = -1
        assert adapter.mark_event_seen("event_1") is True
        assert adapter.mark_event_seen("event_1") is True

    def test_dedupe_key_priority(self, adapter):
        assert adapter.get_dedupe_key("event", "uuid", "message") == "event"
        assert adapter.get_dedupe_key(None, "uuid", "message") == "uuid"
        assert adapter.get_dedupe_key(None, None, "message") == "message"


class TestTenantAccessToken:
    """tenant_access_token 缓存测试"""

    @pytest.mark.asyncio
    async def test_get_tenant_access_token_reuses_valid_cache(self, adapter):
        adapter._tenant_access_token = "cached_token"
        adapter._tenant_access_token_expires_at = 9999999999
        token = await adapter.get_tenant_access_token()
        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_get_tenant_access_token_refreshes_expired_cache(self, adapter):
        adapter._tenant_access_token = "expired_token"
        adapter._tenant_access_token_expires_at = 0

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 0,
            "tenant_access_token": "new_token",
            "expire": 7200,
        }

        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            mock_settings.feishu_app_id = "app_id"
            mock_settings.feishu_app_secret = "app_secret"
            token = await adapter.get_tenant_access_token()

        assert token == "new_token"
        assert adapter._tenant_access_token == "new_token"
        assert adapter._tenant_access_token_expires_at > 0

    @pytest.mark.asyncio
    async def test_reply_auth_error_clears_cached_token(self, adapter):
        adapter._tenant_access_token = "cached_token"
        adapter._tenant_access_token_expires_at = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 99991663, "msg": "invalid token"}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await adapter.reply_message("om_test123", "测试回复")

        assert result is False
        assert adapter._tenant_access_token is None


class TestReplyMessage:
    """回复消息测试"""

    @pytest.mark.asyncio
    async def test_reply_message_success(self, adapter):
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "msg": "success"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(adapter, "get_tenant_access_token", new=AsyncMock(return_value="test_token")), \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await adapter.reply_message("om_test123", "测试回复")
            assert result is True

    @pytest.mark.asyncio
    async def test_reply_message_no_token(self, adapter):
        with patch.object(adapter, "get_tenant_access_token", new=AsyncMock(return_value=None)):
            result = await adapter.reply_message("om_test123", "测试回复")
            assert result is False


class TestSendMessage:
    """发送消息测试"""

    @pytest.mark.asyncio
    async def test_send_message_success(self, adapter):
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "msg": "success"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(adapter, "get_tenant_access_token", new=AsyncMock(return_value="test_token")), \
             patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await adapter.send_message("oc_test123", "测试消息")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_message_no_token(self, adapter):
        with patch.object(adapter, "get_tenant_access_token", new=AsyncMock(return_value=None)):
            result = await adapter.send_message("oc_test123", "测试消息")
            assert result is False


class TestFeishuHeartbeat:
    """飞书长耗时处理心跳测试"""

    @pytest.mark.asyncio
    async def test_fast_processing_before_initial_delay_does_not_send_heartbeat(self):
        with patch.object(settings, "feishu_heartbeat_enabled", True), \
             patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 5), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 20), \
             patch.object(settings, "feishu_heartbeat_max_count", 12), \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send, \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:
            mock_process.return_value.answer = "机器人回答"

            await process_feishu_message({
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
                "chat_type": "p2p",
                "message_id": "om_test456",
                "text": "你好",
                "dedupe_key": "event_fast",
            })

        mock_send.assert_not_awaited()
        mock_reply.assert_awaited_once_with("om_test456", "机器人回答")

    @pytest.mark.asyncio
    async def test_slow_processing_sends_heartbeat_then_final_reply(self):
        async def slow_process(_request):
            await asyncio.sleep(0.01)
            response = MagicMock()
            response.answer = "机器人回答"
            return response

        with patch.object(settings, "feishu_heartbeat_enabled", True), \
             patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_max_count", 1), \
             patch("app.main.orchestrator.process", new=AsyncMock(side_effect=slow_process)), \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send, \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:

            await process_feishu_message({
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
                "chat_type": "p2p",
                "message_id": "om_test456",
                "text": "你好",
                "dedupe_key": "event_slow",
            })

        assert mock_send.await_args_list[0].args == ("oc_test789", "正在处理，请稍候...")
        mock_reply.assert_awaited_once_with("om_test456", "机器人回答")

    @pytest.mark.asyncio
    async def test_heartbeat_failure_does_not_block_final_reply(self):
        async def slow_process(_request):
            await asyncio.sleep(0.01)
            response = MagicMock()
            response.answer = "机器人回答"
            return response

        with patch.object(settings, "feishu_heartbeat_enabled", True), \
             patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_max_count", 1), \
             patch("app.main.orchestrator.process", new=AsyncMock(side_effect=slow_process)), \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(side_effect=RuntimeError("send failed"))), \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:

            await process_feishu_message({
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
                "chat_type": "p2p",
                "message_id": "om_test456",
                "text": "你好",
                "dedupe_key": "event_heartbeat_failed",
            })

        mock_reply.assert_awaited_once_with("om_test456", "机器人回答")

    @pytest.mark.asyncio
    async def test_orchestrator_error_still_replies_friendly_message_with_heartbeat_enabled(self):
        async def slow_error(_request):
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")

        with patch.object(settings, "feishu_heartbeat_enabled", True), \
             patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_max_count", 1), \
             patch("app.main.orchestrator.process", new=AsyncMock(side_effect=slow_error)), \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)), \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:

            await process_feishu_message({
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
                "chat_type": "p2p",
                "message_id": "om_test456",
                "text": "你好",
                "dedupe_key": "event_error_with_heartbeat",
            })

        mock_reply.assert_awaited_once_with("om_test456", "抱歉，处理您的问题时出现错误，请稍后重试。")

    @pytest.mark.asyncio
    async def test_heartbeat_max_count_sends_long_processing_notice(self):
        with patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 0), \
             patch.object(settings, "feishu_heartbeat_max_count", 2), \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:

            await send_feishu_processing_heartbeat("oc_test789", "event_max")

        assert mock_send.await_args_list[0].args == ("oc_test789", "正在处理，请稍候...")
        assert mock_send.await_args_list[1].args == ("oc_test789", "还在检索和整理资料，请稍候...")
        assert mock_send.await_args_list[2].args == ("oc_test789", "处理时间较长，我会继续尝试完成回答。")

    @pytest.mark.asyncio
    async def test_heartbeat_uses_five_second_initial_delay_then_twenty_second_interval(self):
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch.object(settings, "feishu_heartbeat_initial_delay_seconds", 5), \
             patch.object(settings, "feishu_heartbeat_interval_seconds", 20), \
             patch.object(settings, "feishu_heartbeat_max_count", 2), \
             patch("app.main.asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)), \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)):

            await send_feishu_processing_heartbeat("oc_test789", "event_delay")

        assert sleep_calls == [5, 20]


class TestFeishuPersonalKnowledgeFiles:
    """飞书个人知识库文件保存流程测试"""

    def setup_method(self):
        pending_feishu_files.clear()

    def teardown_method(self):
        pending_feishu_files.clear()

    def test_decide_feishu_knowledge_scope_defaults_to_combined(self):
        assert decide_feishu_knowledge_scope("儒家大同思想是什么") == ["enterprise", "personal"]

    def test_decide_feishu_knowledge_scope_enterprise_only(self):
        assert decide_feishu_knowledge_scope("只查企业知识库里的内容") == ["enterprise"]

    def test_decide_feishu_knowledge_scope_personal(self):
        assert decide_feishu_knowledge_scope("只查我的知识库里的内容") == ["personal"]

    def test_decide_feishu_knowledge_scope_combined(self):
        assert decide_feishu_knowledge_scope("结合企业和个人知识库回答") == ["enterprise", "personal"]

    @pytest.mark.asyncio
    async def test_file_event_sends_confirm_card_without_ingest(self):
        event_data = FeishuAdapter().parse_event(_file_event(file_name="资料.xlsx"))

        with patch("app.main.feishu_adapter.send_interactive_card", new=AsyncMock(return_value=True)) as mock_card, \
             patch("app.main.ingest_file") as mock_ingest, \
             patch("app.main.feishu_adapter.download_message_resource", new=AsyncMock()) as mock_download:
            await process_feishu_file_event(event_data)

        assert len(pending_feishu_files) == 1
        pending = next(iter(pending_feishu_files.values()))
        assert pending["file_name"] == "资料.xlsx"
        assert pending["file_key"] == "file_key_1"
        mock_card.assert_awaited_once()
        card = mock_card.await_args.args[1]
        actions = card["elements"][1]["actions"]
        values = [action["value"]["action"] for action in actions]
        assert "confirm_save_personal_file" in values
        assert "confirm_save_enterprise_file" in values
        assert "cancel_save_personal_file" in values
        mock_ingest.assert_not_called()
        mock_download.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_file_event_rejects_unsupported_extension(self):
        event_data = FeishuAdapter().parse_event(_file_event(file_name="宏文件.docm"))

        with patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply, \
             patch("app.main.feishu_adapter.send_interactive_card", new=AsyncMock()) as mock_card:
            await process_feishu_file_event(event_data)

        mock_reply.assert_awaited_once()
        assert "仅支持" in mock_reply.await_args.args[1]
        mock_card.assert_not_awaited()
        assert pending_feishu_files == {}

    @pytest.mark.asyncio
    async def test_confirm_card_downloads_and_ingests_personal_file(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
            "message_id": "om_file456",
            "file_key": "file_key_1",
            "file_name": "资料.txt",
            "expires_at": 9999999999,
        }

        with patch("app.main.feishu_adapter.download_message_resource", new=AsyncMock(return_value=b"hello")) as mock_download, \
             patch("app.main.save_uploaded_file", return_value="D:/LLM/Unified_API_service/data/personal_uploads/stored.txt") as mock_save, \
             patch("app.main.ingest_file") as mock_ingest, \
             patch("app.main.orchestrator.retriever.refresh") as mock_refresh, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            mock_ingest.return_value = {"document_id": "doc1", "filename": "资料.txt", "chunks": 2}

            await process_feishu_card_action({
                "action": "confirm_save_personal_file",
                "pending_id": "pending_1",
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
            })

        mock_download.assert_awaited_once_with("om_file456", "file_key_1")
        mock_save.assert_called_once()
        mock_ingest.assert_called_once()
        assert mock_ingest.call_args.kwargs["knowledge_base_type"] == "personal"
        assert mock_ingest.call_args.kwargs["owner_open_id"] == "ou_test123"
        assert mock_ingest.call_args.kwargs["chat_id"] == "oc_test789"
        assert mock_ingest.call_args.kwargs["channel"] == "feishu"
        mock_refresh.assert_called_once()
        assert "已保存到个人知识库" in mock_send.await_args.args[1]
        assert "pending_1" not in pending_feishu_files

    @pytest.mark.asyncio
    async def test_confirm_card_downloads_and_ingests_enterprise_file(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
            "message_id": "om_file456",
            "file_key": "file_key_1",
            "file_name": "资料.xlsx",
            "expires_at": 9999999999,
        }

        with patch("app.main.feishu_adapter.download_message_resource", new=AsyncMock(return_value=b"hello")) as mock_download, \
             patch("app.main.save_uploaded_file", return_value="D:/LLM/Unified_API_service/data/uploads/stored.xlsx") as mock_save, \
             patch("app.main.ingest_file") as mock_ingest, \
             patch("app.main.orchestrator.retriever.refresh") as mock_refresh, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            mock_ingest.return_value = {"document_id": "doc1", "filename": "资料.xlsx", "chunks": 2}

            await process_feishu_card_action({
                "action": "confirm_save_enterprise_file",
                "pending_id": "pending_1",
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
            })

        mock_download.assert_awaited_once_with("om_file456", "file_key_1")
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["upload_dir"] == settings.upload_dir
        mock_ingest.assert_called_once()
        assert mock_ingest.call_args.kwargs["knowledge_base_type"] == "enterprise"
        assert mock_ingest.call_args.kwargs["owner_open_id"] is None
        assert mock_ingest.call_args.kwargs["chat_id"] == "oc_test789"
        assert mock_ingest.call_args.kwargs["channel"] == "feishu"
        mock_refresh.assert_called_once()
        assert "已保存到企业知识库" in mock_send.await_args.args[1]
        assert "pending_1" not in pending_feishu_files

    @pytest.mark.asyncio
    async def test_cancel_card_does_not_download_or_ingest(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
            "file_name": "资料.pdf",
            "expires_at": 9999999999,
        }

        with patch("app.main.feishu_adapter.download_message_resource", new=AsyncMock()) as mock_download, \
             patch("app.main.ingest_file") as mock_ingest, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            await process_feishu_card_action({
                "action": "cancel_save_personal_file",
                "pending_id": "pending_1",
                "open_id": "ou_test123",
            })

        mock_download.assert_not_awaited()
        mock_ingest.assert_not_called()
        assert "已取消保存" in mock_send.await_args.args[1]
        assert "pending_1" not in pending_feishu_files

    @pytest.mark.asyncio
    async def test_wrong_user_cannot_consume_pending_confirmation(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_owner",
            "chat_id": "oc_test789",
            "file_name": "资料.pdf",
            "expires_at": 9999999999,
        }

        with patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            await process_feishu_card_action({
                "action": "confirm_save_personal_file",
                "pending_id": "pending_1",
                "open_id": "ou_other",
            })

        assert "pending_1" in pending_feishu_files
        assert "只有上传文件的用户" in mock_send.await_args.args[1]

    @pytest.mark.asyncio
    async def test_missing_open_id_cannot_consume_pending_confirmation(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_owner",
            "chat_id": "oc_test789",
            "file_name": "资料.pdf",
            "expires_at": 9999999999,
        }

        with patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            await process_feishu_card_action({
                "action": "confirm_save_personal_file",
                "pending_id": "pending_1",
            })

        assert "pending_1" in pending_feishu_files
        assert "只有上传文件的用户" in mock_send.await_args.args[1]

    @pytest.mark.asyncio
    async def test_expired_pending_is_removed_and_not_downloaded(self):
        pending_feishu_files["pending_1"] = {
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
            "file_name": "资料.pdf",
            "expires_at": 0,
        }

        with patch("app.main.feishu_adapter.download_message_resource", new=AsyncMock()) as mock_download, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            await process_feishu_card_action({
                "action": "confirm_save_personal_file",
                "pending_id": "pending_1",
                "open_id": "ou_test123",
                "chat_id": "oc_test789",
            })

        mock_download.assert_not_awaited()
        assert "文件确认已过期" in mock_send.await_args.args[1]
        assert "pending_1" not in pending_feishu_files

    def test_parse_card_action(self):
        action = parse_feishu_card_action(_card_action("pending_1"))
        assert action == {
            "action": "confirm_save_personal_file",
            "pending_id": "pending_1",
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
        }

    def test_parse_card_action_operator_open_id_fallback(self):
        body = _card_action("pending_1")
        body["event"]["operator"] = {"open_id": "ou_operator"}

        action = parse_feishu_card_action(body)

        assert action["open_id"] == "ou_operator"

    def test_parse_card_action_event_user_id_fallback(self):
        body = _card_action("pending_1")
        body["event"]["operator"] = {}
        body["event"]["user_id"] = {"open_id": "ou_event_user"}

        action = parse_feishu_card_action(body)

        assert action["open_id"] == "ou_event_user"


class TestFeishuEndpoint:
    """飞书回调入口集成测试"""

    def test_challenge_requires_valid_token(self, client):
        body = {
            "type": "url_verification",
            "challenge": "challenge_1",
            "token": "wrong_token",
        }
        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post("/channels/feishu/events", json=body)

        assert response.status_code == 403

    def test_challenge_returns_value_with_valid_token(self, client):
        body = {
            "type": "url_verification",
            "challenge": "challenge_1",
            "token": "expected_token",
        }
        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post("/channels/feishu/events", json=body)

        assert response.status_code == 200
        assert response.json() == {"challenge": "challenge_1"}

    def test_encrypted_challenge_returns_value_with_valid_token(self, client):
        encrypt_key = "test_encrypt_key_123456789012345"
        body = _encrypted_body(
            {
                "type": "url_verification",
                "challenge": "challenge_encrypted",
                "token": "expected_token",
            },
            encrypt_key,
        )

        with patch("app.channels.feishu.settings") as mock_settings:
            mock_settings.feishu_verification_token = "expected_token"
            mock_settings.feishu_encrypt_key = encrypt_key
            response = client.post("/channels/feishu/events", json=body)

        assert response.status_code == 200
        assert response.json() == {"challenge": "challenge_encrypted"}

    def test_valid_message_runs_background_processing_once(self, client):
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process, \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:
            mock_settings.feishu_verification_token = "expected_token"
            mock_process.return_value.answer = "机器人回答"

            response = client.post("/channels/feishu/events", json=_message_event(event_id="event_once"))

        assert response.status_code == 200
        assert response.json() == {"code": 0}
        mock_process.assert_awaited_once()
        mock_reply.assert_awaited_once_with("om_test456", "机器人回答")

    def test_duplicate_event_is_not_processed_twice(self, client):
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process, \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:
            mock_settings.feishu_verification_token = "expected_token"
            mock_process.return_value.answer = "机器人回答"

            first = client.post("/channels/feishu/events", json=_message_event(event_id="event_dup"))
            second = client.post("/channels/feishu/events", json=_message_event(event_id="event_dup"))

        assert first.status_code == 200
        assert second.status_code == 200
        mock_process.assert_awaited_once()
        mock_reply.assert_awaited_once()

    def test_invalid_or_non_text_event_is_acknowledged_without_processing(self, client):
        body = _message_event(event_id="event_image")
        body["event"]["message"]["message_type"] = "image"
        body["event"]["message"]["content"] = '{"image_key": "img_xxx"}'

        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post("/channels/feishu/events", json=body)

        assert response.status_code == 200
        assert response.json() == {"code": 0}
        mock_process.assert_not_called()

    def test_orchestrator_error_replies_friendly_message(self, client):
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.orchestrator.process", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(return_value=True)) as mock_reply:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post("/channels/feishu/events", json=_message_event(event_id="event_error"))

        assert response.status_code == 200
        mock_reply.assert_awaited_once_with("om_test456", "抱歉，处理您的问题时出现错误，请稍后重试。")

    def test_reply_failure_does_not_break_endpoint(self, client):
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process, \
             patch("app.main.feishu_adapter.reply_message", new=AsyncMock(side_effect=RuntimeError("reply failed"))):
            mock_settings.feishu_verification_token = "expected_token"
            mock_process.return_value.answer = "机器人回答"
            response = client.post("/channels/feishu/events", json=_message_event(event_id="event_reply_failed"))

        assert response.status_code == 200
        assert response.json() == {"code": 0}

    def test_file_message_endpoint_sends_card_not_orchestrator(self, client):
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.feishu_adapter.send_interactive_card", new=AsyncMock(return_value=True)) as mock_card, \
             patch("app.main.orchestrator.process", new=AsyncMock()) as mock_process:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post("/channels/feishu/events", json=_file_event(event_id="event_file_endpoint"))

        assert response.status_code == 200
        assert response.json() == {"code": 0}
        mock_card.assert_awaited_once()
        mock_process.assert_not_awaited()
        pending_feishu_files.clear()

    def test_card_action_endpoint_acknowledges_and_schedules_processing(self, client):
        pending_feishu_files["pending_endpoint"] = {
            "open_id": "ou_test123",
            "chat_id": "oc_test789",
            "file_name": "资料.txt",
            "expires_at": 9999999999,
        }
        with patch("app.channels.feishu.settings") as mock_settings, \
             patch("app.main.feishu_adapter.send_message", new=AsyncMock(return_value=True)) as mock_send:
            mock_settings.feishu_verification_token = "expected_token"
            response = client.post(
                "/channels/feishu/events",
                json=_card_action("pending_endpoint", action="cancel_save_personal_file"),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["toast"]["content"] == "已收到操作，正在处理。"
        assert "已取消保存" in mock_send.await_args.args[1]
        assert "pending_endpoint" not in pending_feishu_files

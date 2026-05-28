"""飞书 Channel Adapter 测试"""

import base64
import hashlib
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from app.channels.feishu import FeishuAdapter
from app.main import app


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

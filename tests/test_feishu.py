"""飞书 Channel Adapter 测试"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.channels.feishu import FeishuAdapter, feishu_adapter


@pytest.fixture
def adapter():
    return FeishuAdapter()


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
            assert adapter.verify_token(body) is True


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

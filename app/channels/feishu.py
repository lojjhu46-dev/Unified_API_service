"""飞书 Channel Adapter

基于飞书开放平台文档：
- 事件订阅：https://open.feishu.cn/document/server-docs/im-v1/message/events/receive
- 回复消息：https://open.feishu.cn/document/server-docs/im-v1/message/reply
"""

import base64
import hashlib
import json
import time
import httpx
from typing import Optional
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 飞书 API 基础地址
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAdapter:
    """飞书适配器"""

    def __init__(self):
        self._tenant_access_token: Optional[str] = None
        self._tenant_access_token_expires_at: float = 0
        self._processed_events: dict[str, float] = {}
        self._dedupe_ttl_seconds = 24 * 60 * 60

    def verify_challenge(self, body: dict) -> Optional[dict]:
        """处理飞书 challenge 验证

        飞书配置事件订阅时会发送 challenge 请求，必须在 1 秒内返回 challenge 值。
        """
        # Schema 2.0 格式
        if body.get("type") == "url_verification":
            challenge = body.get("challenge")
            if challenge:
                return {"challenge": challenge}

        # Schema 1.0 格式（兼容）
        challenge = body.get("challenge")
        if challenge and body.get("token"):
            return {"challenge": challenge}

        return None

    def decrypt_event_body(self, body: dict) -> Optional[dict]:
        if "encrypt" not in body:
            return body

        if not settings.feishu_encrypt_key:
            logger.warning("收到飞书加密事件，但未配置 encrypt_key")
            return None

        try:
            key = hashlib.sha256(settings.feishu_encrypt_key.encode("utf-8")).digest()
            encrypted = base64.b64decode(body["encrypt"])
            if len(encrypted) <= 16:
                logger.warning("飞书加密事件密文长度无效")
                return None
            iv = encrypted[:16]
            encrypted_event = encrypted[16:]
            decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
            padded_data = decryptor.update(encrypted_event) + decryptor.finalize()
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            decrypted_text = data.decode("utf-8")
            start = decrypted_text.find("{")
            end = decrypted_text.rfind("}")
            if start >= 0 and end >= start:
                decrypted_text = decrypted_text[start:end + 1]
            decrypted_body = json.loads(decrypted_text)
        except Exception as e:
            logger.warning(f"飞书加密事件解密失败: {e}")
            return None

        return decrypted_body if isinstance(decrypted_body, dict) else None

    def verify_token(self, body: dict) -> bool:
        """验证飞书事件 token"""
        token = body.get("token") or body.get("header", {}).get("token")
        if not settings.feishu_verification_token:
            return False
        return token == settings.feishu_verification_token

    def parse_event(self, body: dict) -> Optional[dict]:
        """解析飞书事件

        支持 Schema 1.0 和 2.0 格式。
        """
        # Schema 2.0
        header = body.get("header", {})
        event = body.get("event", {})

        if header:
            event_type = header.get("event_type")
            if event_type == "im.message.receive_v1":
                return self._parse_message_event_v2(event, header)
            return None

        # Schema 1.0（兼容）
        event_type = body.get("event", {}).get("type")
        if event_type == "message":
            return self._parse_message_event_v1(body.get("event", {}))
        return None

    def _parse_message_event_v2(self, event: dict, header: dict) -> Optional[dict]:
        """解析 Schema 2.0 消息事件"""
        message = event.get("message", {})
        sender = event.get("sender", {})

        message_type = message.get("message_type")
        if message_type != "text":
            return None

        chat_type = message.get("chat_type")  # p2p 或 group
        chat_id = message.get("chat_id")
        message_id = message.get("message_id")

        sender_id = sender.get("sender_id", {})
        open_id = sender_id.get("open_id")

        # 解析消息内容
        content_str = message.get("content", "{}")
        try:
            content_obj = json.loads(content_str)
            text = content_obj.get("text", "").strip()
        except json.JSONDecodeError:
            text = ""

        if not text:
            return None

        # 群聊需要 @机器人，提取 mentions 信息
        mentions = message.get("mentions", [])
        if chat_type == "group":
            if not mentions:
                return None
            # 移除 @机器人 的占位符
            for mention in mentions:
                key = mention.get("key", "")
                name = mention.get("name", "")
                text = text.replace(key, "").replace(f"@{name}", "").strip()

        if not open_id or not chat_id or not message_id or not text:
            return None

        event_id = header.get("event_id")
        uuid = header.get("uuid")
        return {
            "open_id": open_id,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "message_id": message_id,
            "text": text,
            "event_id": event_id,
            "uuid": uuid,
            "dedupe_key": self.get_dedupe_key(event_id, uuid, message_id),
        }

    def _parse_message_event_v1(self, event: dict) -> Optional[dict]:
        """解析 Schema 1.0 消息事件（兼容）"""
        msg_type = event.get("msg_type")
        if msg_type != "text":
            return None

        text = event.get("text_without_at_bot", "").strip()
        if not text:
            text = event.get("text", "").strip()

        open_id = event.get("open_id")
        chat_id = event.get("open_chat_id")
        message_id = event.get("root_id") or event.get("message_id")
        if not open_id or not chat_id or not message_id or not text:
            return None

        return {
            "open_id": open_id,
            "chat_id": chat_id,
            "chat_type": "group" if event.get("chat_type") == "group" else "p2p",
            "message_id": message_id,
            "text": text,
            "event_id": event.get("event_id"),
            "uuid": event.get("uuid"),
            "dedupe_key": self.get_dedupe_key(event.get("event_id"), event.get("uuid"), message_id),
        }

    def get_dedupe_key(
        self,
        event_id: Optional[str],
        uuid: Optional[str],
        message_id: Optional[str],
    ) -> Optional[str]:
        """获取事件去重 key，优先使用事件级唯一标识"""
        return event_id or uuid or message_id

    def mark_event_seen(self, dedupe_key: Optional[str]) -> bool:
        """登记事件并返回是否为首次处理"""
        if not dedupe_key:
            return False

        now = time.time()
        expired_keys = [
            key for key, expires_at in self._processed_events.items()
            if expires_at <= now
        ]
        for key in expired_keys:
            self._processed_events.pop(key, None)

        if dedupe_key in self._processed_events:
            return False

        self._processed_events[dedupe_key] = now + self._dedupe_ttl_seconds
        return True

    def generate_session_id(self, open_id: str, chat_id: str) -> str:
        """生成稳定的 session_id

        使用 open_id + chat_id 的 MD5 哈希，确保同一用户在同一会话中保持上下文。
        """
        raw = f"feishu:{open_id}:{chat_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    async def get_tenant_access_token(self) -> Optional[str]:
        """获取 tenant_access_token"""
        if self._tenant_access_token and time.time() < self._tenant_access_token_expires_at:
            return self._tenant_access_token

        if not settings.feishu_app_id or not settings.feishu_app_secret:
            logger.warning("飞书 app_id 或 app_secret 未配置")
            return None

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                    json={
                        "app_id": settings.feishu_app_id,
                        "app_secret": settings.feishu_app_secret,
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    self._tenant_access_token = data.get("tenant_access_token")
                    expire_seconds = int(data.get("expire") or 7200)
                    self._tenant_access_token_expires_at = time.time() + max(expire_seconds - 300, 60)
                    return self._tenant_access_token
                else:
                    logger.error(f"获取 tenant_access_token 失败: {data}")
                    return None
        except Exception as e:
            logger.error(f"获取 tenant_access_token 异常: {e}")
            return None

    def clear_tenant_access_token(self) -> None:
        """清空 tenant_access_token 缓存"""
        self._tenant_access_token = None
        self._tenant_access_token_expires_at = 0

    def _is_auth_error(self, code: int) -> bool:
        """判断是否为飞书鉴权错误"""
        return code in {99991663, 99991664, 99991668}

    async def reply_message(self, message_id: str, text: str) -> bool:
        """回复飞书消息

        API: POST /im/v1/messages/{message_id}/reply
        """
        token = await self.get_tenant_access_token()
        if not token:
            return False

        # 截断过长的消息
        if len(text) > 150000:  # 150KB 限制
            text = text[:150000] + "...(消息过长已截断)"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/reply",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "content": json.dumps({"text": text}, ensure_ascii=False),
                        "msg_type": "text",
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    return True
                else:
                    if self._is_auth_error(data.get("code")):
                        self.clear_tenant_access_token()
                    logger.error(f"回复消息失败: {data}")
                    return False
        except Exception as e:
            logger.error(f"回复消息异常: {e}")
            return False

    async def send_message(self, chat_id: str, text: str) -> bool:
        """发送飞书消息

        API: POST /im/v1/messages
        """
        token = await self.get_tenant_access_token()
        if not token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{FEISHU_API_BASE}/im/v1/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"receive_id_type": "chat_id"},
                    json={
                        "receive_id": chat_id,
                        "content": json.dumps({"text": text}, ensure_ascii=False),
                        "msg_type": "text",
                    },
                )
                data = resp.json()
                if data.get("code") == 0:
                    return True
                else:
                    if self._is_auth_error(data.get("code")):
                        self.clear_tenant_access_token()
                    logger.error(f"发送消息失败: {data}")
                    return False
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            return False


feishu_adapter = FeishuAdapter()

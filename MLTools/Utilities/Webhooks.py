import io
from PIL import Image

import io
import requests
from typing import Optional, List, Dict, Any


class DiscordWebhook:
    """Minimal Discord webhook client (text + files + embeds)."""

    def __init__(self, url: str, username: Optional[str] = None, noexcept = True):
        self.url = url
        self.username = username
        self.session = requests.Session()
        self.noexcept = noexcept

    def __send_message(
        self,
        text: Optional[str] = None,
        files: Optional[List[bytes]] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Send a message with optional files or embeds.
        - text: plain message text
        - files: list of bytes (e.g., image in memory)
        - embeds: list of embed dicts (see Discord embed spec)
        """
        data: Dict[str, Any] = {}
        if text:
            data["content"] = text
        if self.username:
            data["username"] = self.username
        if embeds:
            data["embeds"] = embeds

        # Prepare multipart if files present
        if files:
            file_payload = []
            for i, content in enumerate(files):
                file_payload.append(
                    (
                        f"file{i}",
                        (f"upload_{i}.png", io.BytesIO(content), "application/octet-stream"),
                    )
                )
            resp = self.session.post(self.url, data=data, files=file_payload)
        else:
            resp = self.session.post(self.url, json=data)

        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Discord webhook error {resp.status_code}: {resp.text}")
    
    def send_message(
        self,
        text: Optional[str] = None,
        files: Optional[List[bytes]] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
    ):
        try:
            self.__send_message(text=text, files=files, embeds=embeds)
        except Exception as e:
            if self.noexcept:
                class_name = type(self).__name__
                print(f"{class_name} raised {type(e)}: {str(e)}")
            else:
                raise e

    @staticmethod
    def kvtable_to_embed(
        title: str,
        kv: Dict[str, Any],
        color: int = 0x5865F2,
        inline: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert a key-value dict into a simple Discord embed.
        Example:
            kvtable_to_embed("Training Stats", {"loss": 0.123, "acc": 99.2})
        """
        fields = []
        for k, v in kv.items():
            value = f"`{v}`" if not isinstance(v, float) else f"`{v:.6f}`"
            fields.append({"name": str(k), "value": value, "inline": inline})
        return {"title": title, "color": color, "fields": fields}

"""DiscordWebhook with a mocked requests.Session — no network involved."""
import json

import pytest

from MLTools.Utilities import DiscordWebhook


class FakeResponse:
    def __init__(self, status_code=204, text="ok"):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, status_code=204):
        self.status_code = status_code
        self.calls = []

    def post(self, url, data=None, json=None, files=None):
        self.calls.append({"url": url, "data": data, "json": json, "files": files})
        return FakeResponse(self.status_code)


@pytest.fixture
def hook():
    h = DiscordWebhook("https://discord.example/webhook", username="bot", noexcept=False)
    h.session = FakeSession()
    return h


def test_text_message_uses_json_body(hook):
    hook.send_message(text="hello")
    call = hook.session.calls[0]
    assert call["files"] is None
    assert call["json"] == {"content": "hello", "username": "bot"}


def test_embeds_without_files(hook):
    embed = DiscordWebhook.kvtable_to_embed("Stats", {"loss": 0.5, "acc": 99})
    hook.send_message(embeds=[embed])
    call = hook.session.calls[0]
    assert call["json"]["embeds"][0]["title"] == "Stats"


def test_files_use_payload_json(hook):
    """Multipart + embeds must be sent as payload_json, not form fields."""
    embed = DiscordWebhook.kvtable_to_embed("T", {"a": 1})
    hook.send_message(text="caption", files=[b"\x89PNG fake"], embeds=[embed])
    call = hook.session.calls[0]
    assert call["json"] is None
    assert "payload_json" in call["data"]
    payload = json.loads(call["data"]["payload_json"])
    assert payload["content"] == "caption"
    assert payload["embeds"][0]["title"] == "T"
    assert len(call["files"]) == 1
    name, (fname, buf, mime) = call["files"][0]
    assert name == "file0" and fname.startswith("upload_")


def test_multiple_files(hook):
    hook.send_message(files=[b"a", b"b", b"c"])
    assert len(hook.session.calls[0]["files"]) == 3


def test_error_status_raises_when_noexcept_false(hook):
    hook.session = FakeSession(status_code=400)
    with pytest.raises(RuntimeError):
        hook.send_message(text="x")


def test_noexcept_swallows_errors(capsys):
    h = DiscordWebhook("https://discord.example/webhook", noexcept=True)
    h.session = FakeSession(status_code=500)
    h.send_message(text="x")  # must not raise
    assert "DiscordWebhook" in capsys.readouterr().out


def test_kvtable_to_embed_formatting():
    embed = DiscordWebhook.kvtable_to_embed("Title", {"f": 0.123456789, "i": 3, "s": "txt"})
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    assert fields["f"] == "`0.123457`"  # floats: 6 decimal places
    assert fields["i"] == "`3`"
    assert fields["s"] == "`txt`"
    assert embed["title"] == "Title"

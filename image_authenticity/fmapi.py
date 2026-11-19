"""Foundation Model API helper -- version-independent serving-endpoint calls.

Uses the Databricks SDK's low-level ApiClient.do(), which authenticates the same
way regardless of OAuth / PAT / notebook context. This avoids depending on newer
SDK conveniences (e.g. get_open_ai_client) that aren't on every runtime.
"""
from __future__ import annotations

from typing import List, Optional


def fmapi_chat(
    endpoint: str,
    messages: List[dict],
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    timeout: int = 120,
) -> dict:
    """POST a chat payload to a serving endpoint; return the parsed response dict.

    Works for text + vision chat models and for image-generation chat endpoints
    (whose assistant message content carries the image data URI).
    """
    from databricks.sdk import WorkspaceClient

    body = {"messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    w = WorkspaceClient()
    return w.api_client.do(
        "POST", f"/serving-endpoints/{endpoint}/invocations", body=body
    )


def first_message_content(resp: dict):
    """Pull choices[0].message.content out of a chat response as a string.

    Normalizes the two response shapes we see across FMAPI models:
    - string content (Claude, GPT): returned as-is.
    - list-of-parts content (Gemini: [{"type":"text","text":"..."}]): the text
      parts are concatenated. Without this, Gemini responses arrive as a list and
      downstream JSON parsing fails (seen as null scores).
    """
    content = resp["choices"][0]["message"]["content"]
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return content

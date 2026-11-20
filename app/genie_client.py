"""Ask a question in plain English through a Genie space and return the answer, the SQL it ran and the rows."""
from __future__ import annotations


def ask(w, space_id: str, question: str, conversation_id: str | None = None) -> dict:
    if conversation_id:
        msg = w.genie.create_message_and_wait(space_id, conversation_id, question)
    else:
        msg = w.genie.start_conversation_and_wait(space_id, question)
    out = {"conversation_id": msg.conversation_id, "message_id": msg.message_id, "status": str(getattr(msg.status, "value", msg.status)),
           "text": "", "sql": "", "columns": [], "rows": []}
    for a in msg.attachments or []:
        if a.text and a.text.content:
            out["text"] = a.text.content
        if a.query:
            out["sql"] = a.query.query or ""
            res = w.genie.get_message_attachment_query_result(space_id, msg.conversation_id, msg.message_id, a.attachment_id)
            sr = res.statement_response
            if sr and sr.manifest and sr.manifest.schema:
                out["columns"] = [c.name for c in sr.manifest.schema.columns]
            if sr and sr.result and sr.result.data_array:
                out["rows"] = [list(r) for r in sr.result.data_array]
    return out

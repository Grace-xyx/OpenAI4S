"""A session moved into a folder has to say so in the session list.

`POST /frames/{id}/folder` wrote `frames.folder_id`, and `GET /frames/{id}`
read it back, but `GET /frames` builds its rows from `browse_frames`, whose
explicit column list never named `folder_id`. `_frame_json` reads the key with
`.get`, so the omission was not an error: every list row answered
`folder_id: null`. The sidebar groups sessions by matching that field against
the folder list, so a folder always counted zero and every session it held fell
into the ungrouped date buckets -- while the move itself reported success and
the row in SQLite was correct. The frozen `GET /frames [ok]` shape had recorded
`frames[].folder_id` as type null only, because no route-level test had ever
put a session in a folder.

Confirmed in a real browser before the fix: a session moved through the
sidebar's own "Move to folder" menu showed the "moved" toast, persisted its
folder_id, and stayed under "Today" beside a folder labelled 0.

Driven through the real handler on a real socket, unmarked, so the response
schema capture observes a list row that carries a folder.
"""

from __future__ import annotations

import http.client
import json
import threading

from openai4s.config import Config, LLMConfig
from openai4s.server import gateway as gateway_mod
from openai4s.server import local_auth
from openai4s.store import get_store
from tests._ports import bound_gateway_server

API = "/api/v1"


class _NullHub:
    def emitter(self, root_frame_id):
        def emit(event):
            del event

        return emit

    def broadcast(self, root_frame_id, event):
        del root_frame_id, event

    def has_subscriber(self, root_frame_id):
        del root_frame_id
        return False

    def drop_frame(self, root_frame_id):
        del root_frame_id


def test_browse_frames_returns_the_stored_folder_id(tmp_path):
    """The repository contract the route depends on."""
    store = get_store(Config(data_dir=tmp_path).db_path)
    try:
        filed = store.new_frame(kind="turn", project_id="p", name="filed")
        loose = store.new_frame(kind="turn", project_id="p", name="loose")
        folder = store.create_folder(project_id="p", name="Assays")
        store.set_frame_folder(filed, folder["folder_id"])

        rows = {row["frame_id"]: row for row in store.browse_frames(project_id="p")}

        assert rows[filed]["folder_id"] == folder["folder_id"]
        assert rows[loose]["folder_id"] is None
    finally:
        store.close()


def test_the_session_list_places_a_moved_session_in_its_folder(tmp_path):
    httpd, port = bound_gateway_server()
    cfg = Config(
        data_dir=tmp_path,
        llm=LLMConfig(provider="deepseek", api_key="test-key"),
        host="127.0.0.1",
        port=port,
    )
    runner = gateway_mod.SessionRunner(cfg, _NullHub(), start_idle_sweeper=False)
    httpd.RequestHandlerClass = gateway_mod.make_handler(cfg, runner.hub, runner)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    token = local_auth.load_or_mint(cfg.data_dir)

    def call(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        try:
            headers = {local_auth.TOKEN_HEADER: token}
            payload = None
            if body is not None:
                payload = json.dumps(body).encode("utf-8")
                headers["Content-Type"] = "application/json"
            conn.request(method, API + path, body=payload, headers=headers)
            response = conn.getresponse()
            data = json.loads(response.read() or b"{}")
            assert response.status in (200, 201), (method, path, response.status, data)
            return data
        finally:
            conn.close()

    def listed(project_id):
        page = call("GET", f"/frames?project_id={project_id}")
        return {row["id"]: row for row in page["frames"]}

    try:
        profile = call(
            "POST",
            "/model-profiles",
            {
                "name": "pinned",
                "provider": "openai_responses",
                "model": "gpt-4o",
                "api_key": "sk-test",
            },
        )
        call("POST", f"/model-profiles/{profile['id']}/activate", {})
        project = call("POST", "/projects", {"name": "Folder grouping"})
        project_id = project["project_id"]
        sessions = {}
        for label in ("Filed", "Loose"):
            created = call(
                "POST", "/frames", {"project_id": project_id, "model": "deepseek-chat"}
            )
            frame_id = created["id"]
            # A session with no name, message or cell is hidden from the list
            # as abandoned; naming it keeps it visible without running a turn.
            call(
                "PATCH",
                f"/frames/{frame_id}",
                {"name": f"{label} session", "task_summary": f"{label} summary"},
            )
            # What a finished turn would have metered.
            runner.store.add_frame_tokens(frame_id, input_tokens=12, output_tokens=3)
            # And the model configuration a first send pins (D2).
            pinned = call("POST", f"/frames/{frame_id}/model-binding", {})
            assert pinned["binding"]["bound"] is True, pinned
            sessions[label] = frame_id
        filed, loose = sessions["Filed"], sessions["Loose"]
        folder = call("POST", f"/projects/{project_id}/folders", {"name": "Assays"})
        folder_id = folder["folder_id"]

        moved = call("POST", f"/frames/{filed}/folder", {"folder_id": folder_id})
        assert moved == {"ok": True}

        rows = listed(project_id)
        assert rows[filed]["folder_id"] == folder_id
        assert rows[loose]["folder_id"] is None

        # The general form of the defect: one serializer, two queries. Every
        # field the detail route reports for a session, its list row must
        # report identically -- a column the list query leaves out is
        # otherwise a silent null, not an error. The comparison only proves
        # anything for a field that is set, so first insist that every field
        # a root session can carry is.
        for frame_id, unset in (
            (filed, {"parent_frame_id"}),
            (loose, {"parent_frame_id", "folder_id"}),
        ):
            detail = call("GET", f"/frames/{frame_id}")
            assert {key for key, value in detail.items() if value is None} == unset
            assert {key: rows[frame_id].get(key) for key in detail} == detail

        # Moving it back out is the same field, cleared.
        call("POST", f"/frames/{filed}/folder", {"folder_id": None})
        assert listed(project_id)[filed]["folder_id"] is None
    finally:
        httpd.shutdown()
        httpd.server_close()
        runner.close()

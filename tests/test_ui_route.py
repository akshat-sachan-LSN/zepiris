from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import ui as ui_routes


def test_ui_page_renders_single_mode() -> None:
    app = FastAPI()
    app.include_router(ui_routes.router)
    client = TestClient(app)
    r = client.get("/ui")
    assert r.status_code == 200
    html = r.text
    assert "s3_url" in html or "S3 URL" in html
    assert "Save to DB" not in html
    assert "/v1/faces/facematch/verify" in html
    assert "/v1/faces/docmatch/verify" in html

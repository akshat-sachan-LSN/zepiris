from fastapi import APIRouter

from zepiris.api.routes import face, health


def build_api_router() -> APIRouter:
    root = APIRouter()
    root.include_router(health.router, tags=["health"])
    root.include_router(face.router, prefix="/v1/faces", tags=["faces"])
    return root

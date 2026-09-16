from fastapi import APIRouter, Depends, Request

from aquaops.config import Settings

router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "ok",
        "environment": settings.environment,
    }

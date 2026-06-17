from fastapi import APIRouter, status

from schemas import MagtGeoJsonResponse
from services import GeoService
from src.api import DBManagerDep

from ..tags import Tags

router = APIRouter(prefix="/magt", tags=[Tags.MAGT])


@router.get(
    "/geojson",
    status_code=status.HTTP_200_OK,
    response_model=MagtGeoJsonResponse,
    summary="Отправка GEOJSON.",
    description="При вызове этого endpoint отпрляется GEOJSON в зависимости от указанного года.",
    name="magt_geojson",
)
async def get_magt_geojson(
    year: int,
    db: DBManagerDep,
) -> MagtGeoJsonResponse:

    services = GeoService(db)

    features = await services.get_geojson(year)

    return MagtGeoJsonResponse(features=features)

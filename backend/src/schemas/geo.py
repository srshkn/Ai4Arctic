from .config import APIModel


class PropertiesResponse(APIModel):
    year: int
    class_id: int
    color: str
    magt_min: float | None
    magt_max: float | None
    label: str


class FeatureResponse(APIModel):
    type: str = "Feature"
    properties: PropertiesResponse
    geometry: dict


class MagtGeoJsonResponse(APIModel):
    type: str = "FeatureCollection"
    crs: dict = {"type": "name", "properties": {"name": "EPSG:4326"}}
    features: list[FeatureResponse]

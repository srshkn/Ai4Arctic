from geoalchemy2 import Geometry
from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base


class MagtFeature(Base):
    __tablename__ = "magt_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    year: Mapped[int] = mapped_column(Integer)

    class_id: Mapped[int] = mapped_column(Integer)

    color: Mapped[str] = mapped_column(String(32))

    magt_min: Mapped[float | None] = mapped_column(Float, nullable=True)

    magt_max: Mapped[float | None] = mapped_column(Float, nullable=True)

    label: Mapped[str] = mapped_column(String(255))

    geometry: Mapped[object] = mapped_column(
        Geometry(
            geometry_type="MULTIPOLYGON",
            srid=4326,
            spatial_index=False,
        )
    )

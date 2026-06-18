import json
from pathlib import Path

import grpc
from tqdm import tqdm

from core import get_settings
from grpc_client.contract import geo_pb2, geo_pb2_grpc
from grpc_client.parse_geojson import parse_geojson_directory

settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent

TARGET_DIRECTORY = "results/geojson_smooth/"

DIR_GEO = BASE_DIR / TARGET_DIRECTORY


def run():
    channel = grpc.insecure_channel(f"{settings.GRPC_SERVER}:50051")

    stub = geo_pb2_grpc.GeoImportServiceStub(channel)

    list_geo_dict = parse_geojson_directory(DIR_GEO)

    for geo_dict in tqdm(list_geo_dict):
        request = geo_pb2.ImportMagtDatasetRequest(  # type: ignore[attr-defined]
            geojson=json.dumps(geo_dict)
        )

        response = stub.ImportMagtDataset(request)

        print("response:", response.success, response.message)


if __name__ == "__main__":
    run()

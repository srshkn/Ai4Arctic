import json

import grpc

from src.db import DBManager, SessionLocal
from src.grpc_server.contract import geo_pb2, geo_pb2_grpc
from src.services import GeoService


class GeoImportService(geo_pb2_grpc.GeoImportServiceServicer):
    def __init__(self, session):
        self.session = session

    async def ImportMagtDataset(self, request, context):
        print("geojson go")
        async with self.session as db:
            service = GeoService(db)

            for feature in json.loads(request.geojson)["features"]:
                await service.add_geojson_db(feature)

        return geo_pb2.ImportMagtDatasetResponse(  # type: ignore[attr-defined]
            success=True, message="Imported successfully"
        )


async def server():
    server = grpc.aio.server()

    geo_pb2_grpc.add_GeoImportServiceServicer_to_server(
        GeoImportService(DBManager(SessionLocal)), server
    )

    server.add_insecure_port("[::]:50051")
    await server.start()
    print("gRPC server started on :50051")
    await server.wait_for_termination()


# if __name__ == "__main__":
#    serve()

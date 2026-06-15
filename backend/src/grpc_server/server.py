from concurrent import futures

import grpc

from src.grpc_server.contract import geo_pb2, geo_pb2_grpc


class GeoImportService(geo_pb2_grpc.GeoImportServiceServicer):
    def ImportMagtDataset(self, request, context):
        print("dataset:", request.dataset_name)
        print("year:", request.year)
        print("layers:", len(request.layers))

        # TODO: сохранить в PostGIS

        return geo_pb2.ImportMagtDatasetResponse(  # type: ignore[attr-defined]
            success=True, dataset_id=123, message="Imported successfully"
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    geo_pb2_grpc.add_GeoImportServiceServicer_to_server(GeoImportService(), server)

    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC server started on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

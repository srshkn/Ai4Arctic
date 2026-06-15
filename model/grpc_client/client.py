import grpc

from grpc_client.contract import geo_pb2, geo_pb2_grpc


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = geo_pb2_grpc.GeoImportServiceStub(channel)

    request = geo_pb2.ImportMagtDatasetRequest(  # type: ignore[attr-defined]
        dataset_name="arctic_2026",
        year=2026,
        layers=[
            geo_pb2.Layer(class_id=1, color="red", label="ice", geometry="POLYGON(...)")  # type: ignore[attr-defined]
        ],
    )

    response = stub.ImportMagtDataset(request)

    print("response:", response.success, response.dataset_id)


if __name__ == "__main__":
    run()

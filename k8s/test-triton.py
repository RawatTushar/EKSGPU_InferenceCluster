import numpy as np
import tritonclient.http as httpclient
import time
from concurrent.futures import ThreadPoolExecutor

TRITON_URL = "triton.inference.svc.cluster.local:8000"
MODEL_NAME = "pytorch-resnet50"


def inference(request_id):

    # API input shape from Triton config:
    # dims = [3, 224, 224]
    # max_batch_size = 0
    data = np.random.rand(
        3, 224, 224
    ).astype(np.float32)

    # One client per thread
    client = httpclient.InferenceServerClient(
        url=TRITON_URL
    )

    input_tensor = httpclient.InferInput(
        "input__0",
        data.shape,
        "FP32"
    )

    input_tensor.set_data_from_numpy(data)

    output_tensor = httpclient.InferRequestedOutput(
        "output__0"
    )

    start = time.perf_counter()

    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    output = result.as_numpy(
        "output__0"
    )

    print(
        f"Request {request_id}: "
        f"latency={latency_ms:.2f} ms "
        f"output_shape={output.shape}"
    )

    client.close()

    return latency_ms


# ==========================================
# TEST 1 - SEQUENTIAL
# ==========================================

print("\n===================================")
print("TEST 1 - 2 SEQUENTIAL REQUESTS")
print("===================================\n")

latencies = []

for i in range(2):

    latency = inference(i + 1)

    latencies.append(latency)


print("\nResults")

print(
    f"Average latency: "
    f"{np.mean(latencies):.2f} ms"
)

print(
    f"Minimum latency: "
    f"{np.min(latencies):.2f} ms"
)

print(
    f"Maximum latency: "
    f"{np.max(latencies):.2f} ms"
)


# ==========================================
# TEST 2 - CONCURRENT
# ==========================================

print("\n===================================")
print("TEST 2 - 2 CONCURRENT REQUESTS")
print("===================================\n")

start = time.perf_counter()

with ThreadPoolExecutor(
    max_workers=2
) as executor:

    futures = [
        executor.submit(
            inference,
            i + 1
        )
        for i in range(2)
    ]

    concurrent_latencies = [
        future.result()
        for future in futures
    ]


total_time_ms = (
    time.perf_counter() - start
) * 1000


print("\nConcurrent Results")

print(
    f"Average request latency: "
    f"{np.mean(concurrent_latencies):.2f} ms"
)

print(
    f"Total wall time: "
    f"{total_time_ms:.2f} ms"
)

print("\nTest completed.")

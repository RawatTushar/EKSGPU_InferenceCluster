import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.http as http

TRITON_URL = "triton.inference.svc.cluster.local:8000"


def infer_pytorch():
    client = http.InferenceServerClient(TRITON_URL)

    x = np.random.rand(1, 3, 224, 224).astype(np.float32)

    inp = http.InferInput("input__0", x.shape, "FP32")
    inp.set_data_from_numpy(x)

    out = http.InferRequestedOutput("output__0")

    start = time.perf_counter()

    result = client.infer(
        "pytorch-resnet50",
        inputs=[inp],
        outputs=[out]
    )

    latency = (time.perf_counter() - start) * 1000

    return latency, result.as_numpy("output__0").shape


def infer_onnx():
    client = http.InferenceServerClient(TRITON_URL)

    x = np.random.rand(1, 224, 224, 3).astype(np.float32)

    inp = http.InferInput("inputs", x.shape, "FP32")
    inp.set_data_from_numpy(x)

    out = http.InferRequestedOutput("output_0")

    start = time.perf_counter()

    result = client.infer(
        "tensorflow-resnet50",
        inputs=[inp],
        outputs=[out]
    )

    latency = (time.perf_counter() - start) * 1000

    return latency, result.as_numpy("output_0").shape


def run_test(model, concurrency):

    fn = infer_pytorch if model == "pytorch-resnet50" else infer_onnx

    print()
    print("=" * 60)
    print(f"MODEL: {model}")
    print(f"CONCURRENCY: {concurrency}")
    print("=" * 60)

    start = time.perf_counter()

    latencies = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:

        futures = [
            executor.submit(fn)
            for _ in range(concurrency)
        ]

        for future in as_completed(futures):

            try:
                latency, shape = future.result()
                latencies.append(latency)

            except Exception as e:
                errors += 1
                print("ERROR:", e)

    wall_time = time.perf_counter() - start

    successful = len(latencies)

    throughput = (
        successful / wall_time
        if wall_time > 0
        else 0
    )

    print(f"Successful requests : {successful}")
    print(f"Failed requests     : {errors}")

    if latencies:

        print(f"Min latency         : {min(latencies):.2f} ms")
        print(f"Average latency     : {np.mean(latencies):.2f} ms")
        print(f"Max latency         : {max(latencies):.2f} ms")

    print(f"Total wall time     : {wall_time:.4f} sec")
    print(f"Throughput          : {throughput:.2f} req/s")

    if latencies:
        print(f"Output shape        : {shape}")


print("Running warmup...")

for _ in range(3):
    infer_pytorch()
    infer_onnx()

print("Warmup complete.")


for model in [
    "pytorch-resnet50",
    "tensorflow-resnet50"
]:

    for concurrency in [5, 10]:

        run_test(model, concurrency)

        time.sleep(2)

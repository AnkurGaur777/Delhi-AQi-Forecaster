import statistics
import time

from fastapi.testclient import TestClient

from aqi.api import main


def measure(requests: int = 200) -> dict[str, float]:
    with TestClient(main.app) as client:
        client.get("/forecast", params={"location": "near_ito"})
        timings = []
        for _ in range(requests):
            start = time.perf_counter()
            client.get("/forecast", params={"location": "near_ito"})
            timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    return {"p50_ms": statistics.median(ordered), "p95_ms": ordered[int(0.95 * len(ordered)) - 1]}


if __name__ == "__main__":
    print(measure())

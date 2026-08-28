import json
import sys
import time
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
nonce = str(time.time_ns())
dataset_digest = "dataset-" + nonce
schema_digest = "schema-" + nonce


def evaluation(artifact, accuracy, latency):
    return {
        "createdAt": "2026-08-28T11:30:00Z",
        "artifactDigest": artifact,
        "datasetDigest": dataset_digest,
        "schemaDigest": schema_digest,
        "accuracy": accuracy,
        "latencyMs": latency,
        "sizeBytes": 500000,
        "slices": {"critical": 0.85},
    }


payload = {
    "asOf": "2026-08-28T12:00:00Z",
    "championVersion": "1",
    "policy": {
        "datasetDigest": dataset_digest,
        "schemaDigest": schema_digest,
        "maxAgeSeconds": 3600,
        "accuracyFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "maxLatencyMs": 100,
        "maxSizeBytes": 1000000,
        "minImprovement": 0.01,
    },
    "versions": [
        {
            "version": "1",
            "artifactDigest": "artifact-1",
            "tags": {"accuracy": 1.0, "description": "Mutable claim; ignored"},
            "evaluation": evaluation("artifact-1", 0.80, 50),
        },
        {
            "version": "2",
            "artifactDigest": "artifact-2",
            "tags": {"accuracy": 0.1, "description": "Also ignored"},
            "evaluation": evaluation("artifact-2", 0.83, 60),
        },
    ],
}


def post(value):
    request = urllib.request.Request(
        base_url + "/promote",
        data=json.dumps(value, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


status, promoted = post(payload)
print("FIRST REQUEST HTTP", status)
print(json.dumps(promoted, ensure_ascii=False, indent=2))

status, replayed = post(payload)
print("REPLAY HTTP", status)
print(json.dumps(replayed, ensure_ascii=False, indent=2))

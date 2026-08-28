import copy
import hashlib
import json
import sys
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def post(payload):
    request = urllib.request.Request(
        base_url + "/quantize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


freeze_request = {
    "phase": "freeze",
    "freezeId": "quantize-example-v1",
    "calibrationDigest": "calibration-v1",
    "tokenizerDigest": "tokenizer-v1",
    "allowedUnsupportedReasons": ["NO_FP8_KERNEL"],
    "candidates": [
        {
            "name": "int8",
            "files": {
                "model.safetensors": "abc",
                "config.json": "é",
            },
            "loadable": True,
            "calibrationDigest": "calibration-v1",
            "tokenizerDigest": "tokenizer-v1",
        },
        {
            "name": "int4",
            "files": {"model.safetensors": "xy"},
            "loadable": True,
            "calibrationDigest": "calibration-v1",
            "tokenizerDigest": "tokenizer-v1",
        },
        {
            "name": "fp8",
            "files": {"model.safetensors": "z"},
            "loadable": False,
            "calibrationDigest": "different",
            "tokenizerDigest": "different",
            "unsupportedReason": "NO_FP8_KERNEL",
        },
    ],
}

status, frozen = post(freeze_request)
print("FREEZE HTTP", status)
print(json.dumps(frozen, ensure_ascii=False, indent=2))
assert status == 200
assert [candidate["name"] for candidate in frozen["candidates"]] == [
    "fp8",
    "int4",
    "int8",
]

by_name = {candidate["name"]: candidate for candidate in frozen["candidates"]}
assert by_name["fp8"]["status"] == "unsupported"
assert by_name["int4"]["status"] == "frozen"
assert by_name["int8"]["status"] == "frozen"

# UTF-8 encodes "é" as two bytes, so int8 contains 2 + 3 = 5 bytes.
assert by_name["int8"]["totalBytes"] == 5
expected_package_digest = hashlib.sha256(
    json.dumps(
        by_name["int8"]["inventory"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
assert by_name["int8"]["packageDigest"] == expected_package_digest

# An identical replay must return the frozen response byte-for-byte logically.
assert post(freeze_request) == (200, frozen)

select_request = {
    "phase": "select",
    "freezeId": freeze_request["freezeId"],
    "candidates": frozen["candidates"],
    "policy": {
        "maxBytes": 100,
        "aggregateFloor": 0.8,
        "requiredSlices": {"critical": 0.75},
        "maxLatencyMs": 100,
        "candidateOrder": ["int4", "int8", "fp8"],
    },
    "latencies": {"int4": 40, "int8": 60, "fp8": 10},
    "rows": [
        {
            "label": 1,
            "slice": "critical",
            "predictions": {"int4": 1, "int8": 1, "fp8": 1},
        },
        {
            "label": 0,
            "slice": "critical",
            "predictions": {"int4": 1, "int8": 0, "fp8": 0},
        },
    ],
}

status, selected = post(select_request)
print("SELECT HTTP", status)
print(json.dumps(selected, ensure_ascii=False, indent=2))
assert status == 200
assert [result["name"] for result in selected["results"]] == [
    "int4",
    "int8",
    "fp8",
]
assert selected["selected"] == "int8"
assert selected["packageManifest"] == by_name["int8"]
assert selected["results"][0]["reasonCodes"] == [
    "AGGREGATE_FLOOR",
    "SLICE_FLOOR:critical",
]
assert selected["results"][2]["reasonCodes"] == ["NOT_FROZEN"]

# Changing a claimed size proves both the frozen lineage and manifest are checked.
tampered_request = copy.deepcopy(select_request)
for candidate in tampered_request["candidates"]:
    if candidate["name"] == "int8":
        candidate["totalBytes"] += 1

status, tampered = post(tampered_request)
assert status == 200
tampered_int8 = next(
    result for result in tampered["results"] if result["name"] == "int8"
)
assert "INVALID_LINEAGE" in tampered_int8["reasonCodes"]
assert "INVALID_MANIFEST" in tampered_int8["reasonCodes"]
assert tampered_int8["totalBytes"] == 5

# Reusing the freeze ID for different files is a conflict.
conflicting_request = copy.deepcopy(freeze_request)
conflicting_request["candidates"][0]["files"]["model.safetensors"] = "changed"
assert post(conflicting_request) == (409, {"error": "FREEZE_ID_CONFLICT"})

print("All quantization checks passed.")

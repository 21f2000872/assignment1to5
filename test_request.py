import json
import sys
import urllib.error
import urllib.request


def crc32c_hex(data: bytes) -> str:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return f"{(crc ^ 0xFFFFFFFF):08x}"


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

row = {
    "id": "row-1",
    "entity": "  ＡCME   School  ",
    "eventTime": "2026-01-02T05:30:00+05:30",
    "revision": 0,
    "text": "  Hello\tWORLD  ",
}
content = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"

payload = {
    "policy": {
        "minTime": "2026-01-01T00:00:00Z",
        "maxTime": "2026-12-31T23:59:59.999Z",
        "contaminationThreshold": 0.8,
    },
    "objects": [
        {
            "uri": "gs://demo-bucket/training.jsonl",
            "generation": "7",
            "fetchedGeneration": "7",
            "crc32c": crc32c_hex(content.encode("utf-8")),
            "schemaId": "training-v1",
            "content": content,
        }
    ],
}

request = urllib.request.Request(
    base_url + "/build-corpus",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json", "Accept": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        print("HTTP", response.status)
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
except urllib.error.HTTPError as error:
    print("HTTP", error.code)
    print(error.read().decode("utf-8"))
except urllib.error.URLError as error:
    print("Could not reach the service:", error.reason)

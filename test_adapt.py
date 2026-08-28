import json
import sys
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def post(payload):
    request = urllib.request.Request(
        base_url + "/adapt",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


choose = {
    "operation": "choose",
    "policy": {
        "minQuality": 0.8,
        "freshnessRequired": True,
        "maxLatencyMs": 100,
        "maxMemoryMb": 1024,
        "maxLabeledExamples": 100,
        "maxTotalCost": 1000,
        "horizonRequests": 10000,
    },
    "candidates": [
        {
            "name": "prompt_only",
            "available": True,
            "quality": 0.79,
            "freshness": True,
            "latencyMs": 20,
            "memoryMb": 10,
            "labeledExamples": 0,
            "oneTimeCost": 1,
            "recurringCost": 0.001,
        },
        {
            "name": "retrieval",
            "available": True,
            "quality": 0.85,
            "freshness": True,
            "latencyMs": 50,
            "memoryMb": 256,
            "labeledExamples": 0,
            "oneTimeCost": 10,
            "recurringCost": 0.01,
        },
        {
            "name": "lora",
            "available": True,
            "quality": 0.90,
            "freshness": True,
            "latencyMs": 80,
            "memoryMb": 800,
            "labeledExamples": 50,
            "oneTimeCost": 100,
            "recurringCost": 0.02,
        },
        {
            "name": "qlora",
            "available": False,
            "quality": 0.95,
            "freshness": False,
            "latencyMs": 101,
            "memoryMb": 1025,
            "labeledExamples": 101,
            "oneTimeCost": 900,
            "recurringCost": 0.02,
        },
    ],
}

status, chosen = post(choose)
print("CHOOSE HTTP", status)
print(json.dumps(chosen, ensure_ascii=False, indent=2))

dataset_digest = "b" * 64
code_digest = "c" * 64
config_digest = "d" * 64

repair = {
    "operation": "repair",
    "tokens": [
        {"id": 0, "role": "system", "padding": False, "text": "rules"},
        {"id": 1, "role": "user", "padding": False, "text": "question"},
        {"id": 42, "role": "assistant", "padding": False, "text": "answer"},
        {"id": 43, "role": "assistant", "padding": True, "text": "padding"},
    ],
    "templateApplications": 1,
    "parameters": [
        {"name": "z.lora_B.weight", "target": "q_proj", "numel": 3},
        {"name": "a.lora_A.weight", "target": "q_proj", "numel": 2},
        {"name": "base.weight", "target": "q_proj", "numel": 100},
    ],
    "allowedTargets": ["q_proj"],
    "inferenceMode": False,
    "trainRowIds": ["train-1", "train-2"],
    "evalRowIds": ["eval-1"],
    "dropoutActiveDuringEval": False,
    "artifactFiles": ["adapter_model.safetensors", "adapter_config.json"],
    "baseRevision": "a" * 40,
    "datasetDigest": dataset_digest,
    "codeDigest": code_digest,
    "configDigest": config_digest,
    "expectedDigests": {
        "datasetDigest": dataset_digest,
        "codeDigest": code_digest,
        "configDigest": config_digest,
    },
    "microBatch": 2,
    "gradientAccumulation": 4,
    "replicas": 2,
    "expectedEffectiveBatch": 16,
    "checkpoint": {
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "step": 10,
        "rng": {},
        "dataPosition": 5,
    },
    "uninterruptedWeights": [1.0, 2.0],
    "resumedWeights": [1.0005, 1.9995],
    "resumeTolerance": 0.001,
}

status, repaired = post(repair)
print("REPAIR HTTP", status)
print(json.dumps(repaired, ensure_ascii=False, indent=2))

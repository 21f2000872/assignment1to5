import copy
import hashlib
import json
import sys
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def post(payload):
    request = urllib.request.Request(
        base_url + "/verify-bundle",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rebuild_inventory(files):
    """Replace inventory.json with an independently recomputed inventory."""
    inventory = []
    names = sorted(
        (name for name in files if name != "inventory.json"),
        key=lambda name: name.encode("utf-8"),
    )
    for name in names:
        exact_bytes = files[name].encode("utf-8")
        inventory.append(
            {
                "name": name,
                "bytes": len(exact_bytes),
                "sha256": hashlib.sha256(exact_bytes).hexdigest(),
            }
        )
    files["inventory.json"] = compact(inventory)
    return files


policy = {
    "requiredSlices": ["critical"],
    "license": "Apache-2.0",
    "intendedUse": "Teaching demonstrations",
    "limitations": "Not for medical or safety-critical decisions",
}

# Although this assignment calls the file .safetensors, its request representation
# is UTF-8 text. The verifier hashes those exact UTF-8 bytes without executing it.
model_text = "pretend-safetensors-weights-é"
model_digest = digest(model_text)

evaluation_text = compact(
    {
        "modelArtifactDigest": model_digest,
        "aggregate": 0.9,
        "slices": {"critical": 0.8, "optional": "not checked"},
        "extraEvaluationProperty": True,
    }
)

manifest = {
    "baseRevision": "a" * 40,
    "task": "classification",
    "datasetDigest": "dataset-digest-v1",
    "codeDigest": "code-digest-v1",
    "trainingConfigDigest": "training-config-v1",
    "modelArtifactDigest": model_digest,
    "evaluationArtifactDigest": digest(evaluation_text),
    "extraManifestProperty": "allowed",
}

card = {
    "task": "classification",
    "baseRevision": "a" * 40,
    "datasetDigest": "dataset-digest-v1",
    "modelArtifactDigest": model_digest,
    "license": policy["license"],
    "intendedUse": policy["intendedUse"],
    "limitations": policy["limitations"],
    # These braces prove that the parser does not stop at the first closing brace.
    "note": "A JSON string containing {still text} is ordinary data",
}

files = rebuild_inventory(
    {
        "README.md": (
            "# Verifiable model card\n\n"
            "<!-- tds-model-card " + compact(card) + " -->\n\n"
            "Additional prose is allowed."
        ),
        "training_manifest.json": compact(manifest),
        "evaluation.json": evaluation_text,
        "adapter_model.safetensors": model_text,
        "adapter_config.json": compact(
            {
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "extraConfigProperty": "allowed",
            }
        ),
    }
)

status, valid = post({"policy": policy, "files": files})
print("VALID BUNDLE HTTP", status)
print(json.dumps(valid, ensure_ascii=False, indent=2))
assert status == 200
assert valid["decision"] == "admit"
assert valid["violations"] == []
expected_inventory_digest = digest(files["inventory.json"])
assert valid["inventoryDigest"] == expected_inventory_digest

# Changing the model while repairing the inventory proves the verifier does not
# trust inventory.json alone. The immutable manifest and evaluation still catch it.
tampered_model_files = copy.deepcopy(files)
tampered_model_files["adapter_model.safetensors"] = "changed model"
rebuild_inventory(tampered_model_files)
status, tampered = post({"policy": policy, "files": tampered_model_files})
assert status == 200
assert tampered["violations"] == [
    "EVALUATION_ARTIFACT_MISMATCH",
    "MODEL_ARTIFACT_MISMATCH",
]

# A syntactically valid but pretty-printed inventory is not the required exact
# compact artifact.
noncompact_files = copy.deepcopy(files)
noncompact_files["inventory.json"] = json.dumps(
    json.loads(noncompact_files["inventory.json"]),
    ensure_ascii=False,
    indent=2,
)
status, noncompact = post({"policy": policy, "files": noncompact_files})
assert status == 200
assert noncompact["violations"] == ["INVENTORY_MISMATCH"]

# Extra unsafe weights are reported independently as unsafe and untracked.
unsafe_files = copy.deepcopy(files)
unsafe_files["full-model.pkl"] = "untrusted pickle contents"
rebuild_inventory(unsafe_files)
status, unsafe = post({"policy": policy, "files": unsafe_files})
assert status == 200
assert unsafe["violations"] == ["UNSAFE_WEIGHTS", "UNTRACKED_FILE"]

# The specification says two otherwise-valid markers emit only MODEL_CARD_COUNT.
two_marker_files = copy.deepcopy(files)
two_marker_files["README.md"] += "\n<!-- tds-model-card " + compact(card) + " -->"
rebuild_inventory(two_marker_files)
status, two_markers = post({"policy": policy, "files": two_marker_files})
assert status == 200
assert two_markers["violations"] == ["MODEL_CARD_COUNT"]

# Missing top-level policy and a non-object files value are the special HTTP 400 cases.
assert post({"files": {}}) == (400, {"error": "INVALID_INPUT"})
assert post({"policy": policy, "files": []}) == (
    400,
    {"error": "INVALID_INPUT"},
)

print("All model-bundle verification checks passed.")

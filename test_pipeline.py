import copy
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def post(payload):
    request = urllib.request.Request(
        base_url + "/pipeline",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def cache_key(values):
    compact_array = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(compact_array).hexdigest()


def event(
    event_id,
    revision,
    node,
    attempt,
    status,
    key,
    artifact_digest=None,
    receipt_id=None,
):
    return {
        "eventId": event_id,
        "revision": revision,
        "node": node,
        "attempt": attempt,
        "status": status,
        "key": key,
        "artifactDigest": artifact_digest,
        "receiptId": receipt_id,
    }


def request_body(session, revision, inputs, events=None):
    return {
        "session": session,
        "revision": revision,
        "inputs": inputs,
        "events": events or [],
    }


def node(response, name):
    return next(item for item in response["nodes"] if item["node"] == name)


unique = str(time.time_ns())
session = "pipeline-example-" + unique
inputs = {
    "generation": "generation-1",
    "checksum": "checksum-1",
    "canonicalData": "canonical-data-1",
    "prepareCode": "prepare-code-1",
    "prepareConfig": "prepare-config-1",
    "trainCode": "train-code-1",
    "trainConfig": "train-config-1",
    "runtime": "python-runtime-1",
    "evaluateCode": "evaluate-code-1",
    "evaluateConfig": "evaluate-config-1",
    "schemaDigest": "schema-1",
    "publishConfig": "publish-config-1",
    "extraMetadata": {"owner": "example-team"},
}

# The first read has no cache. Only verify_data can run yet.
status, initial = post(request_body(session, 1, inputs))
print("INITIAL HTTP", status)
print(json.dumps(initial, ensure_ascii=False, indent=2))
assert status == 200
assert [item["node"] for item in initial["nodes"]] == [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]
verify_key = cache_key(["generation-1", "checksum-1"])
assert node(initial, "verify_data")["dependencyDigests"]["cacheKey"] == verify_key
assert node(initial, "verify_data")["reasonCodes"] == ["CACHE_MISS"]
assert node(initial, "prepare")["reasonCodes"] == ["UPSTREAM_PENDING"]

# Demonstrate the permitted retry chain for verify_data.
verify_events = [
    event("verify-start-1", 1, "verify_data", 1, "started", verify_key),
    event(
        "verify-retry-1",
        1,
        "verify_data",
        1,
        "retryable_failed",
        verify_key,
    ),
    event("verify-start-2", 1, "verify_data", 2, "started", verify_key),
    event(
        "verify-success-2",
        1,
        "verify_data",
        2,
        "succeeded",
        verify_key,
        "verified-artifact",
    ),
]
status, verified = post(request_body(session, 1, inputs, verify_events))
assert status == 200
assert verified["acceptedEventIds"] == [item["eventId"] for item in verify_events]
assert node(verified, "verify_data")["reasonCodes"] == ["CACHE_HIT"]

# Once a parent succeeds, its child becomes ready. These keys use the exact
# dependency order from the assignment.
prepare_key = cache_key(
    ["canonical-data-1", "prepare-code-1", "prepare-config-1"]
)
train_key = cache_key(
    [
        "prepared-artifact",
        "train-code-1",
        "train-config-1",
        "python-runtime-1",
    ]
)
evaluate_key = cache_key(
    [
        "trained-artifact",
        "canonical-data-1",
        "evaluate-code-1",
        "evaluate-config-1",
    ]
)
register_key = cache_key(["evaluated-artifact", "schema-1"])
publish_key = cache_key(["registered-artifact", "publish-config-1"])

# The bad register receipt is ignored. Its event ID is then reused by the
# corrected event, proving ignored events do not consume IDs.
remaining_events = [
    event("prepare-start", 1, "prepare", 1, "started", prepare_key),
    event(
        "prepare-success",
        1,
        "prepare",
        1,
        "succeeded",
        prepare_key,
        "prepared-artifact",
    ),
    event("train-start", 1, "train", 1, "started", train_key),
    event(
        "train-success",
        1,
        "train",
        1,
        "succeeded",
        train_key,
        "trained-artifact",
    ),
    event("evaluate-start", 1, "evaluate", 1, "started", evaluate_key),
    event(
        "evaluate-success",
        1,
        "evaluate",
        1,
        "succeeded",
        evaluate_key,
        "evaluated-artifact",
    ),
    event("register-start", 1, "register", 1, "started", register_key),
    event(
        "register-success",
        1,
        "register",
        1,
        "succeeded",
        register_key,
        "registered-artifact",
        "wrong-receipt",
    ),
    event(
        "register-success",
        1,
        "register",
        1,
        "succeeded",
        register_key,
        "registered-artifact",
        "receipt:register:" + register_key,
    ),
    event("publish-start", 1, "publish", 1, "started", publish_key),
    event(
        "publish-success",
        1,
        "publish",
        1,
        "succeeded",
        publish_key,
        "published-artifact",
        "receipt:publish:" + publish_key,
    ),
]
status, completed = post(request_body(session, 1, inputs, remaining_events))
print("COMPLETED HTTP", status)
print(json.dumps(completed, ensure_ascii=False, indent=2))
assert status == 200
assert completed["ignoredEventIds"] == ["register-success"]
assert all(item["reasonCodes"] == ["CACHE_HIT"] for item in completed["nodes"])
assert node(completed, "publish")["triggeringEventIds"] == ["publish-success"]

# A new revision clears attempt/failure state but keeps matching cache entries.
revision_two_inputs = copy.deepcopy(inputs)
revision_two_inputs["extraMetadata"]["note"] = "new revision, same content keys"
stale_event = event("old-revision-event", 1, "verify_data", 1, "started", verify_key)
status, reused = post(
    request_body(session, 2, revision_two_inputs, [stale_event])
)
assert status == 200
assert reused["ignoredEventIds"] == ["old-revision-event"]
assert all(item["action"] == "reuse" for item in reused["nodes"])

# Extra metadata is part of revision identity even though it is not a cache-key input.
conflicting_inputs = copy.deepcopy(revision_two_inputs)
conflicting_inputs["extraMetadata"]["note"] = "changed in same revision"
assert post(request_body(session, 2, conflicting_inputs)) == (
    409,
    {"error": "REVISION_CONFLICT"},
)

# A successful cache entry cannot be rebound to a different artifact.
different_evidence = event(
    "different-evidence",
    2,
    "verify_data",
    1,
    "succeeded",
    verify_key,
    "different-artifact",
)
assert post(request_body(session, 2, revision_two_inputs, [different_evidence])) == (
    409,
    {"error": "EVIDENCE_CONFLICT"},
)

# An invalid second event rolls back both the first event and revision change.
revision_three_inputs = copy.deepcopy(revision_two_inputs)
revision_three_inputs["trainConfig"] = "train-config-3"
train_key_three = cache_key(
    [
        "prepared-artifact",
        "train-code-1",
        "train-config-3",
        "python-runtime-1",
    ]
)
rollback_batch = [
    event("rolled-back-start", 3, "train", 1, "started", train_key_three),
    {"eventId": "missing-seven-fields"},
]
assert post(request_body(session, 3, revision_three_inputs, rollback_batch)) == (
    409,
    {"error": "INVALID_EVENT"},
)
status, readback = post(request_body(session, 2, revision_two_inputs))
assert status == 200 and readback["revision"] == 2

# A separate session starts empty and cannot see the first session's cache.
status, isolated = post(request_body("isolated-" + unique, 1, inputs))
assert status == 200
assert node(isolated, "verify_data")["reasonCodes"] == ["CACHE_MISS"]
assert node(isolated, "prepare")["reasonCodes"] == ["UPSTREAM_PENDING"]

# A terminal failure permanently blocks that node and marks every descendant
# as blocked by an upstream terminal failure.
terminal_session = "terminal-" + unique
terminal_events = [
    event("terminal-v-start", 1, "verify_data", 1, "started", verify_key),
    event(
        "terminal-v-success",
        1,
        "verify_data",
        1,
        "succeeded",
        verify_key,
        "verified-artifact",
    ),
    event("terminal-p-start", 1, "prepare", 1, "started", prepare_key),
    event(
        "terminal-p-success",
        1,
        "prepare",
        1,
        "succeeded",
        prepare_key,
        "prepared-artifact",
    ),
    event("terminal-t-start", 1, "train", 1, "started", train_key),
    event(
        "terminal-t-success",
        1,
        "train",
        1,
        "succeeded",
        train_key,
        "trained-artifact",
    ),
    event("terminal-e-start", 1, "evaluate", 1, "started", evaluate_key),
    event(
        "terminal-e-failed",
        1,
        "evaluate",
        1,
        "terminal_failed",
        evaluate_key,
    ),
]
status, terminal = post(
    request_body(terminal_session, 1, inputs, terminal_events)
)
assert status == 200
assert node(terminal, "evaluate")["reasonCodes"] == ["TERMINAL_FAILURE"]
assert node(terminal, "register")["reasonCodes"] == ["UPSTREAM_TERMINAL"]
assert node(terminal, "publish")["reasonCodes"] == ["UPSTREAM_TERMINAL"]
assert post(
    request_body(
        terminal_session,
        1,
        inputs,
        [event("after-terminal", 1, "evaluate", 2, "started", evaluate_key)],
    )
) == (409, {"error": "STATUS_CONFLICT"})

print("All pipeline recovery checks passed.")

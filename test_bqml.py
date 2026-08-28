import json
import sys
import time
import urllib.error
import urllib.request


base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
run_id = f"novice-test-{time.time_ns()}"


def feature(value, available_at):
    return {"value": value, "availableAt": available_at}


def post(payload):
    request = urllib.request.Request(
        base_url + "/bqml",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


selection = {
    "phase": "select",
    "runId": run_id,
    "forbiddenFeatures": ["answer_after_exam"],
    "numTrialsLimit": 10,
    "rows": [
        {
            "id": "train-1",
            "entity": "student-a",
            "eventTime": "2026-01-01T05:30:00+05:30",
            "predictionTime": "2026-01-02T00:00:00Z",
            "version": 1,
            "split": "TRAIN",
            "features": {
                "study_hours": feature(5, "2026-01-01T00:00:00Z"),
                "answer_after_exam": feature("secret", "2026-01-03T00:00:00Z"),
            },
        },
        {
            "id": "eval-1",
            "entity": "student-b",
            "eventTime": "2026-01-03T00:00:00Z",
            "predictionTime": "2026-01-04T00:00:00Z",
            "version": 1,
            "split": "EVAL",
            "features": {
                "study_hours": feature(4, "2026-01-04T00:00:00Z"),
                "answer_after_exam": feature("secret", "2026-01-05T00:00:00Z"),
            },
        },
    ],
    "trials": [
        {"trialId": 9, "status": "SUCCEEDED", "evalMetric": 0.9},
        {"trialId": 4, "status": "SUCCEEDED", "evalMetric": 0.9},
    ],
}

status, selected = post(selection)
print("SELECT HTTP", status)
print(json.dumps(selected, ensure_ascii=False, indent=2))

if status != 200 or selected.get("selectedTrialId") is None:
    raise SystemExit("Selection did not succeed, so evaluation was not sent.")

evaluation = {
    "phase": "evaluate",
    "runId": run_id,
    "selectedTrialId": selected["selectedTrialId"],
    "datasetDigest": selected["datasetDigest"],
    "metricFloor": 0.6,
    "requiredSlices": {"critical": 0.5},
    "rows": [
        {"label": 1, "prediction": 1, "slice": "critical"},
        {"label": 0, "prediction": 1, "slice": "critical"},
        {"label": 0, "prediction": 0, "slice": "other"},
    ],
    "bytesProcessed": 1000,
    "maxBytes": 2000,
}

status, evaluated = post(evaluation)
print("EVALUATE HTTP", status)
print(json.dumps(evaluated, ensure_ascii=False, indent=2))

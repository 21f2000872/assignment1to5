import asyncio
import copy
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

SAFE_INTEGER_MAX = 2**53 - 1
ASCII_DECIMAL_RE = re.compile(r"^[0-9]+$")
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")
GCS_URI_RE = re.compile(r"gs://[^/\r\n]+/[^\r\n]+")
TIMESTAMP_RE = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{1,3}))?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})$"
)
UNICODE_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
BQML_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# Selection responses are frozen here so evaluate requests can prove lineage.
# The deployment command starts one Uvicorn process, so this state is shared by
# all requests handled by that process.
BQML_RUNS: dict[str, dict[str, Any]] = {}
BQML_RUNS_LOCK = asyncio.Lock()

# Promotion alias state is keyed by immutable evidence context, not by mutable
# tags. This makes a replay after promotion retain the new champion.
PROMOTION_ALIASES: dict[str, str] = {}
PROMOTION_ALIASES_LOCK = asyncio.Lock()

# Frozen quantization manifests and responses are retained for the select phase.
QUANTIZE_FREEZES: dict[str, dict[str, Any]] = {}
QUANTIZE_FREEZES_LOCK = asyncio.Lock()


class DuplicateJSONKey(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKey(key)
        result[key] = value
    return result


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sorted_codes(codes: list[str] | set[str]) -> list[str]:
    return sorted(set(codes), key=lambda code: code.encode("utf-8"))


def crc32c_hex(data: bytes) -> str:
    """Return the CRC32C (Castagnoli) checksum as 8 lowercase hex digits."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return f"{(crc ^ 0xFFFFFFFF):08x}"


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Validate the required syntax and return the corresponding UTC datetime."""
    if not isinstance(value, str):
        return None

    match = TIMESTAMP_RE.fullmatch(value)
    if match is None:
        return None

    year, month, day, hour, minute, second = map(
        int, match.group(1, 2, 3, 4, 5, 6)
    )
    fraction = match.group(7) or ""
    offset_text = match.group(8)

    if offset_text == "Z":
        offset = timedelta(0)
    else:
        offset_hour = int(offset_text[1:3])
        offset_minute = int(offset_text[4:6])
        if offset_minute > 59:
            return None
        if offset_hour > 14 or (offset_hour == 14 and offset_minute != 0):
            return None
        offset = timedelta(hours=offset_hour, minutes=offset_minute)
        if offset_text[0] == "-":
            offset = -offset

    milliseconds = int(fraction.ljust(3, "0")) if fraction else 0
    try:
        local_time = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            milliseconds * 1000,
            tzinfo=timezone(offset),
        )
        return local_time.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def utc_timestamp(value: datetime) -> str:
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}T"
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}."
        f"{value.microsecond // 1000:03d}Z"
    )


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    return UNICODE_WHITESPACE_RE.sub(" ", value)


def is_valid_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != {"id", "entity", "eventTime", "revision", "text"}:
        return False
    if not all(isinstance(row[key], str) for key in ("id", "entity", "eventTime", "text")):
        return False
    revision = row["revision"]
    if type(revision) is not int:
        return False
    if revision < 0 or revision > SAFE_INTEGER_MAX:
        return False
    return parse_timestamp(row["eventTime"]) is not None


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    event_time = parse_timestamp(row["eventTime"])
    assert event_time is not None
    return {
        "id": row["id"],
        "entity": canonical_text(row["entity"]),
        "eventTime": utc_timestamp(event_time),
        "revision": row["revision"],
        "text": canonical_text(row["text"]),
    }


def validate_policy(policy: Any) -> Optional[tuple[datetime, datetime, float]]:
    if not isinstance(policy, dict):
        return None

    minimum = parse_timestamp(policy.get("minTime"))
    maximum = parse_timestamp(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")

    if minimum is None or maximum is None or minimum > maximum:
        return None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    if not 0 <= threshold <= 1:
        return None
    if isinstance(threshold, float) and not math.isfinite(threshold):
        return None
    return minimum, maximum, float(threshold)


def word_set(text: str) -> set[str]:
    words: set[str] = set()
    current: list[str] = []
    for character in text.lower():
        if unicodedata.category(character)[0] in {"L", "N"}:
            current.append(character)
        elif current:
            words.add("".join(current))
            current = []
    if current:
        words.add("".join(current))
    return words


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def row_sort_key(row: dict[str, Any]) -> tuple[bytes, bytes]:
    return row["id"].encode("utf-8"), compact_json(row).encode("utf-8")


def nullable_uri_key(uri: Any) -> bytes:
    # null has no URI bytes, so it sorts as the empty byte string.
    return uri.encode("utf-8") if isinstance(uri, str) else b""


def response_item_sort_key(item: dict[str, Any], field: str) -> tuple[bytes, bytes]:
    if field == "uri":
        primary = nullable_uri_key(item[field])
    else:
        primary = item[field].encode("utf-8")
    return primary, compact_json(item).encode("utf-8")


@app.post("/build-corpus")
async def build_corpus(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict) or "policy" not in body or not isinstance(body.get("objects"), list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    accepted_rows: list[dict[str, Any]] = []
    rejected_objects: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []

    for supplied_object in body["objects"]:
        object_data = supplied_object if isinstance(supplied_object, dict) else {}
        supplied_uri = object_data.get("uri")
        output_uri = supplied_uri if isinstance(supplied_uri, str) else None
        reasons: set[str] = set()

        try:
            uri_is_utf8 = isinstance(supplied_uri, str) and bool(supplied_uri.encode("utf-8"))
        except UnicodeEncodeError:
            uri_is_utf8 = False
        if not uri_is_utf8 or GCS_URI_RE.fullmatch(supplied_uri) is None:
            reasons.add("URI_INVALID")

        generation = object_data.get("generation")
        fetched_generation = object_data.get("fetchedGeneration")
        generation_valid = (
            isinstance(generation, str)
            and ASCII_DECIMAL_RE.fullmatch(generation) is not None
            and isinstance(fetched_generation, str)
            and ASCII_DECIMAL_RE.fullmatch(fetched_generation) is not None
        )
        if not generation_valid:
            reasons.add("GENERATION_INVALID")
        if generation != fetched_generation:
            reasons.add("GENERATION_MISMATCH")

        supplied_crc = object_data.get("crc32c")
        crc_syntax_valid = (
            isinstance(supplied_crc, str)
            and CRC32C_RE.fullmatch(supplied_crc) is not None
        )
        if not crc_syntax_valid:
            reasons.add("CRC32C_INVALID")

        content = object_data.get("content")
        content_bytes: Optional[bytes] = None
        if isinstance(content, str):
            try:
                content_bytes = content.encode("utf-8")
            except UnicodeEncodeError:
                reasons.add("SCHEMA_INVALID")
        if isinstance(content, str) and crc_syntax_valid:
            if content_bytes is None or crc32c_hex(content_bytes) != supplied_crc:
                reasons.add("CRC32C_MISMATCH")

        schema_id = object_data.get("schemaId")
        if schema_id != "training-v1" or not isinstance(content, str):
            reasons.add("SCHEMA_INVALID")

        parsed_rows: list[dict[str, Any]] = []
        if isinstance(content, str) and content_bytes is not None:
            nonblank_count = 0
            for line in content.splitlines():
                if line.strip() == "":
                    continue
                nonblank_count += 1
                try:
                    parsed = json.loads(
                        line,
                        object_pairs_hook=reject_duplicate_keys,
                    )
                except DuplicateJSONKey:
                    reasons.add("SCHEMA_INVALID")
                    continue
                except (json.JSONDecodeError, RecursionError):
                    reasons.add("JSONL_INVALID")
                    continue

                if not is_valid_row(parsed):
                    reasons.add("SCHEMA_INVALID")
                    continue
                parsed_rows.append(canonical_row(parsed))

            if nonblank_count == 0:
                reasons.add("SCHEMA_INVALID")

        if reasons:
            rejected_objects.append(
                {"uri": output_uri, "reasonCodes": sorted_codes(reasons)}
            )
        else:
            accepted_rows.extend(parsed_rows)
            lineage.append(
                {
                    "uri": supplied_uri,
                    "generation": generation,
                    "crc32c": supplied_crc,
                    "schemaId": schema_id,
                }
            )

    # Deduplicate canonical rows.  Only one winner from each tuple is retained.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in accepted_rows:
        key = (row["entity"], row["eventTime"], row["text"])
        groups.setdefault(key, []).append(row)

    retained_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for rows in groups.values():
        winner = min(
            rows,
            key=lambda row: (-row["revision"], row["id"].encode("utf-8")),
        )
        retained_rows.append(winner)
        winner_skipped = False
        for row in rows:
            if row is winner and not winner_skipped:
                winner_skipped = True
                continue
            rejected_rows.append({"id": row["id"], "reasonCodes": ["DUPLICATE"]})

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    validated_policy = validate_policy(body["policy"])
    if validated_policy is None:
        for row in retained_rows:
            rejected_rows.append(
                {"id": row["id"], "reasonCodes": ["POLICY_INVALID"]}
            )
    else:
        minimum, maximum, threshold = validated_policy
        in_window: list[tuple[dict[str, Any], str]] = []

        for row in retained_rows:
            row_time = parse_timestamp(row["eventTime"])
            assert row_time is not None
            if not minimum <= row_time <= maximum:
                rejected_rows.append(
                    {"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]}
                )
                continue

            bucket = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0] % 10
            split_name = "train" if bucket <= 5 else "validation" if bucket <= 7 else "test"
            in_window.append((row, split_name))

        train_word_sets = [
            word_set(row["text"])
            for row, split_name in in_window
            if split_name == "train"
        ]

        for row, split_name in in_window:
            if split_name != "train":
                candidate_words = word_set(row["text"])
                contaminated = any(
                    jaccard(candidate_words, train_words) >= threshold
                    for train_words in train_word_sets
                )
                if contaminated:
                    rejected_rows.append(
                        {
                            "id": row["id"],
                            "reasonCodes": ["TRAIN_CONTAMINATION"],
                        }
                    )
                    continue
            split_rows[split_name].append(row)

    digests: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        split_rows[split_name].sort(key=row_sort_key)
        exact_bytes = b"".join(
            compact_json(row).encode("utf-8") + b"\n"
            for row in split_rows[split_name]
        )
        digests[split_name] = hashlib.sha256(exact_bytes).hexdigest()

    for item in rejected_objects:
        item["reasonCodes"] = sorted_codes(item["reasonCodes"])
    for item in rejected_rows:
        item["reasonCodes"] = sorted_codes(item["reasonCodes"])

    rejected_objects.sort(key=lambda item: response_item_sort_key(item, "uri"))
    rejected_rows.sort(key=lambda item: response_item_sort_key(item, "id"))
    lineage.sort(key=lambda item: response_item_sort_key(item, "uri"))

    response = {
        "splits": split_rows,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }
    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Assignment 2: stateful, leakage-safe BigQuery ML experiment gate
# ---------------------------------------------------------------------------


def is_safe_integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= SAFE_INTEGER_MAX


def is_utf8_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def is_valid_run_id(value: Any) -> bool:
    return is_utf8_string(value) and 1 <= len(value) <= 128


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def is_unit_floor(value: Any) -> bool:
    return is_finite_number(value) and 0 <= value <= 1


def selection_fingerprint(body: dict[str, Any]) -> str:
    """Fingerprint parsed JSON while ignoring JSON object key order."""
    canonical_input = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("ascii")
    return hashlib.sha256(canonical_input).hexdigest()


def parse_selection_rows(value: Any) -> Optional[list[dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        return None

    parsed_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in value:
        if not isinstance(row, dict):
            return None

        row_id = row.get("id")
        entity = row.get("entity")
        event_time = parse_timestamp(row.get("eventTime"))
        prediction_time = parse_timestamp(row.get("predictionTime"))
        version = row.get("version")
        split = row.get("split")
        features = row.get("features")

        if not is_utf8_string(row_id) or row_id in seen_ids:
            return None
        if not is_utf8_string(entity):
            return None
        if event_time is None or prediction_time is None:
            return None
        if not is_safe_integer(version):
            return None
        if split not in {"TRAIN", "EVAL"}:
            return None
        if not isinstance(features, dict):
            return None

        parsed_features: dict[str, datetime] = {}
        for feature_name, feature_data in features.items():
            if not is_utf8_string(feature_name):
                return None
            if not isinstance(feature_data, dict):
                return None
            if "value" not in feature_data or "availableAt" not in feature_data:
                return None
            available_at = parse_timestamp(feature_data.get("availableAt"))
            if available_at is None:
                return None
            # feature_data["value"] is deliberately not interpreted. It is data.
            parsed_features[feature_name] = available_at

        seen_ids.add(row_id)
        parsed_rows.append(
            {
                "id": row_id,
                "entity": entity,
                "eventTime": event_time,
                "predictionTime": prediction_time,
                "version": version,
                "split": split,
                "features": parsed_features,
            }
        )

    return parsed_rows


def parse_trials(value: Any) -> Optional[list[dict[str, Any]]]:
    if not isinstance(value, list):
        return None

    parsed_trials: list[dict[str, Any]] = []
    seen_trial_ids: set[int] = set()

    for trial in value:
        if not isinstance(trial, dict):
            return None
        trial_id = trial.get("trialId")
        status = trial.get("status")
        if not is_safe_integer(trial_id) or trial_id in seen_trial_ids:
            return None
        if status not in {"SUCCEEDED", "FAILED"}:
            return None

        seen_trial_ids.add(trial_id)
        parsed_trials.append(
            {
                "trialId": trial_id,
                "status": status,
                "evalMetric": trial.get("evalMetric"),
            }
        )

    return parsed_trials


def freeze_selection_dataset(
    rows: list[dict[str, Any]], forbidden_features: set[str]
) -> tuple[list[str], list[str], list[str], str]:
    groups: dict[tuple[str, datetime], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["entity"], row["eventTime"])
        groups.setdefault(key, []).append(row)

    retained_rows: list[dict[str, Any]] = []
    for candidates in groups.values():
        winner = min(
            candidates,
            key=lambda row: (-row["version"], row["id"].encode("utf-8")),
        )
        retained_rows.append(winner)

    train_ids = sorted(
        (row["id"] for row in retained_rows if row["split"] == "TRAIN"),
        key=lambda value: value.encode("utf-8"),
    )
    eval_ids = sorted(
        (row["id"] for row in retained_rows if row["split"] == "EVAL"),
        key=lambda value: value.encode("utf-8"),
    )

    common_features = set(retained_rows[0]["features"])
    for row in retained_rows[1:]:
        common_features.intersection_update(row["features"])

    eligible_features: list[str] = []
    for feature_name in common_features:
        if feature_name in forbidden_features:
            continue
        point_in_time_safe = all(
            row["features"][feature_name] <= row["predictionTime"]
            for row in retained_rows
        )
        if point_in_time_safe:
            eligible_features.append(feature_name)

    eligible_features.sort(key=lambda value: value.encode("utf-8"))

    digest_input = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": eligible_features,
    }
    dataset_digest = hashlib.sha256(
        compact_json(digest_input).encode("utf-8")
    ).hexdigest()
    return train_ids, eval_ids, eligible_features, dataset_digest


def make_selection_response(body: dict[str, Any]) -> dict[str, Any]:
    supplied_run_id = body.get("runId")
    output_run_id = supplied_run_id if isinstance(supplied_run_id, str) else None
    reasons: set[str] = set()

    forbidden_value = body.get("forbiddenFeatures")
    forbidden_valid = (
        isinstance(forbidden_value, list)
        and all(is_utf8_string(name) for name in forbidden_value)
    )
    forbidden_features = set(forbidden_value) if forbidden_valid else set()

    trials_limit = body.get("numTrialsLimit")
    limit_valid = type(trials_limit) is int and trials_limit > 0
    parsed_rows = parse_selection_rows(body.get("rows"))
    parsed_trials = parse_trials(body.get("trials"))

    input_valid = (
        is_valid_run_id(supplied_run_id)
        and forbidden_valid
        and limit_valid
        and parsed_rows is not None
        and parsed_trials is not None
    )

    if not input_valid:
        reasons.add("INVALID_INPUT")

    trial_limit_exceeded = (
        limit_valid
        and isinstance(body.get("trials"), list)
        and len(body["trials"]) > trials_limit
    )
    if trial_limit_exceeded:
        reasons.add("TRIAL_LIMIT_EXCEEDED")

    train_ids: list[str] = []
    eval_ids: list[str] = []
    feature_names: list[str] = []
    dataset_digest: Optional[str] = None
    selected_trial_id: Optional[int] = None

    if input_valid:
        assert parsed_rows is not None and parsed_trials is not None
        train_ids, eval_ids, feature_names, computed_digest = freeze_selection_dataset(
            parsed_rows, forbidden_features
        )

        eligible_trials = [
            trial
            for trial in parsed_trials
            if trial["status"] == "SUCCEEDED"
            and is_finite_number(trial["evalMetric"])
        ]
        if not eligible_trials:
            reasons.add("NO_SUCCESSFUL_TRIAL")
        else:
            best_trial = max(
                eligible_trials,
                key=lambda trial: (trial["evalMetric"], -trial["trialId"]),
            )
            selected_trial_id = best_trial["trialId"]

        # A trial-limit contract failure makes the frozen digest unusable.
        # No-success is not a malformed dataset, so its digest remains useful.
        if not trial_limit_exceeded:
            dataset_digest = computed_digest

    if reasons:
        selected_trial_id = None

    return {
        "runId": output_run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted_codes(reasons),
    }


async def handle_bqml_select(body: dict[str, Any]) -> JSONResponse:
    run_id = body.get("runId")
    fingerprint = selection_fingerprint(body)

    async with BQML_RUNS_LOCK:
        if is_valid_run_id(run_id) and run_id in BQML_RUNS:
            stored = BQML_RUNS[run_id]
            if stored["fingerprint"] == fingerprint:
                return JSONResponse(copy.deepcopy(stored["response"]))
            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = make_selection_response(body)
        if is_valid_run_id(run_id):
            BQML_RUNS[run_id] = {
                "fingerprint": fingerprint,
                "response": copy.deepcopy(response),
                "successful": (
                    response["selectedTrialId"] is not None
                    and response["datasetDigest"] is not None
                    and not response["reasonCodes"]
                ),
            }

    return JSONResponse(response)


def parse_required_slices(value: Any) -> Optional[dict[str, float | int]]:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, float | int] = {}
    for name, floor in value.items():
        if not is_utf8_string(name) or name == "" or not is_unit_floor(floor):
            return None
        parsed[name] = floor
    return parsed


def parse_test_rows(value: Any) -> tuple[bool, list[dict[str, Any]]]:
    if not isinstance(value, list):
        return False, []

    parsed: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            return False, []
        label = row.get("label")
        prediction = row.get("prediction")
        slice_name = row.get("slice")
        if type(label) is not int or label not in {0, 1}:
            return False, []
        if type(prediction) is not int or prediction not in {0, 1}:
            return False, []
        if not is_utf8_string(slice_name) or slice_name == "":
            return False, []
        parsed.append(
            {
                "label": label,
                "prediction": prediction,
                "slice": slice_name,
            }
        )
    return bool(parsed), parsed


def rounded_accuracy(rows: list[dict[str, Any]]) -> float:
    correct = sum(row["label"] == row["prediction"] for row in rows)
    return round(correct / len(rows), 12)


async def handle_bqml_evaluate(body: dict[str, Any]) -> JSONResponse:
    supplied_run_id = body.get("runId")
    supplied_trial_id = body.get("selectedTrialId")
    supplied_digest = body.get("datasetDigest")
    supplied_bytes = body.get("bytesProcessed")

    output_run_id = supplied_run_id if isinstance(supplied_run_id, str) else None
    output_trial_id = supplied_trial_id if type(supplied_trial_id) is int else None
    output_digest = supplied_digest if isinstance(supplied_digest, str) else None
    output_bytes = supplied_bytes if type(supplied_bytes) is int else None

    reasons: set[str] = set()
    input_invalid = False

    lineage_fields_well_formed = (
        is_valid_run_id(supplied_run_id)
        and is_safe_integer(supplied_trial_id)
        and isinstance(supplied_digest, str)
        and BQML_DIGEST_RE.fullmatch(supplied_digest) is not None
    )
    if not lineage_fields_well_formed:
        input_invalid = True

    metric_floor = body.get("metricFloor")
    metric_floor_valid = is_unit_floor(metric_floor)
    if not metric_floor_valid:
        input_invalid = True

    required_slices = parse_required_slices(body.get("requiredSlices"))
    required_slices_valid = required_slices is not None
    if not required_slices_valid:
        input_invalid = True

    rows_value = body.get("rows")
    rows_container_valid = isinstance(rows_value, list)
    if not rows_container_valid:
        input_invalid = True
    test_rows_valid, test_rows = parse_test_rows(rows_value)

    bytes_processed_valid = is_safe_integer(supplied_bytes)
    max_bytes = body.get("maxBytes")
    max_bytes_valid = is_safe_integer(max_bytes)
    if not bytes_processed_valid or not max_bytes_valid:
        input_invalid = True

    if input_invalid:
        reasons.add("INVALID_INPUT")

    async with BQML_RUNS_LOCK:
        stored = (
            copy.deepcopy(BQML_RUNS.get(supplied_run_id))
            if is_valid_run_id(supplied_run_id)
            else None
        )

    lineage_valid = (
        lineage_fields_well_formed
        and stored is not None
        and stored["successful"]
        and stored["response"]["selectedTrialId"] == supplied_trial_id
        and stored["response"]["datasetDigest"] == supplied_digest
    )
    if not lineage_valid:
        reasons.add("INVALID_LINEAGE")

    if rows_container_valid and not test_rows_valid:
        reasons.add("INVALID_TEST_ROW")

    test_metric: Optional[float] = None
    slice_gates_pass = False

    if test_rows_valid:
        test_metric = rounded_accuracy(test_rows)
        if metric_floor_valid and test_metric < metric_floor:
            reasons.add("AGGREGATE_FLOOR")

        if required_slices_valid:
            assert required_slices is not None
            slice_gates_pass = True
            for slice_name in sorted(
                required_slices,
                key=lambda value: value.encode("utf-8"),
            ):
                slice_rows = [
                    row for row in test_rows if row["slice"] == slice_name
                ]
                if not slice_rows:
                    reasons.add(f"MISSING_SLICE:{slice_name}")
                    slice_gates_pass = False
                    continue
                slice_metric = rounded_accuracy(slice_rows)
                if slice_metric < required_slices[slice_name]:
                    reasons.add(f"SLICE_FLOOR:{slice_name}")
                    slice_gates_pass = False

    if bytes_processed_valid and max_bytes_valid and supplied_bytes > max_bytes:
        reasons.add("BYTE_LIMIT")

    critical_slice_pass = (
        not input_invalid
        and lineage_valid
        and test_rows_valid
        and required_slices_valid
        and slice_gates_pass
    )

    reason_codes = sorted_codes(reasons)
    response = {
        "runId": output_run_id,
        "selectedTrialId": output_trial_id,
        "datasetDigest": output_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": "admit" if not reason_codes else "reject",
        "bytesProcessed": output_bytes,
        "reasonCodes": reason_codes,
    }
    return JSONResponse(response)


@app.post("/bqml")
async def bqml(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict) or body.get("phase") not in {"select", "evaluate"}:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if body["phase"] == "select":
        return await handle_bqml_select(body)
    return await handle_bqml_evaluate(body)


# ---------------------------------------------------------------------------
# Assignment 3: deterministic MLflow-style model promotion gate
# ---------------------------------------------------------------------------


CANONICAL_VERSION_RE = re.compile(r"^[1-9][0-9]*$")


def canonical_version_number(value: Any) -> Optional[int]:
    if not is_utf8_string(value) or CANONICAL_VERSION_RE.fullmatch(value) is None:
        return None
    # The largest safe integer has 16 decimal digits. Checking the length first
    # also avoids converting arbitrarily large untrusted strings.
    if len(value) > 16:
        return None
    number = int(value)
    return number if 1 <= number <= SAFE_INTEGER_MAX else None


def promotion_json_fingerprint(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def occurrence_key(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def failed_gate_key(value: Any) -> str:
    if is_utf8_string(value):
        return value
    return occurrence_key(value)


def parse_promotion_policy(value: Any) -> tuple[bool, dict[str, Any]]:
    if not isinstance(value, dict):
        return False, {
            "datasetDigest": None,
            "schemaDigest": None,
            "maxAgeSeconds": None,
            "accuracyFloor": None,
            "requiredSlices": None,
            "maxLatencyMs": None,
            "maxSizeBytes": None,
            "minImprovement": None,
        }

    dataset_digest = value.get("datasetDigest")
    dataset_valid = is_utf8_string(dataset_digest) and dataset_digest != ""

    schema_digest = value.get("schemaDigest")
    schema_valid = is_utf8_string(schema_digest) and schema_digest != ""

    max_age = value.get("maxAgeSeconds")
    max_age_valid = is_safe_integer(max_age)

    accuracy_floor = value.get("accuracyFloor")
    accuracy_floor_valid = is_unit_floor(accuracy_floor)

    required_value = value.get("requiredSlices")
    required_valid = isinstance(required_value, dict)
    required_slices: Optional[dict[str, float | int]] = None
    if required_valid:
        required_slices = {}
        for name, floor in required_value.items():
            if not is_utf8_string(name) or name == "" or not is_unit_floor(floor):
                required_valid = False
                required_slices = None
                break
            required_slices[name] = floor

    max_latency = value.get("maxLatencyMs")
    max_latency_valid = is_finite_number(max_latency) and max_latency >= 0

    max_size = value.get("maxSizeBytes")
    max_size_valid = is_safe_integer(max_size)

    min_improvement = value.get("minImprovement")
    min_improvement_valid = is_unit_floor(min_improvement)

    policy_valid = all(
        (
            dataset_valid,
            schema_valid,
            max_age_valid,
            accuracy_floor_valid,
            required_valid,
            max_latency_valid,
            max_size_valid,
            min_improvement_valid,
        )
    )

    return policy_valid, {
        "datasetDigest": dataset_digest if dataset_valid else None,
        "schemaDigest": schema_digest if schema_valid else None,
        "maxAgeSeconds": max_age if max_age_valid else None,
        "accuracyFloor": accuracy_floor if accuracy_floor_valid else None,
        "requiredSlices": required_slices,
        "maxLatencyMs": max_latency if max_latency_valid else None,
        "maxSizeBytes": max_size if max_size_valid else None,
        "minImprovement": min_improvement if min_improvement_valid else None,
    }


def promotion_context_fingerprint(body: dict[str, Any]) -> str:
    # Tags and descriptions are intentionally omitted: mutable claims must not
    # change the evidence identity or the replayed champion alias.
    evidence_versions: list[Any] = []
    for version in body["versions"]:
        if isinstance(version, dict):
            evidence_versions.append(
                {
                    "version": version.get("version"),
                    "artifactDigest": version.get("artifactDigest"),
                    "evaluation": version.get("evaluation"),
                }
            )
        else:
            evidence_versions.append(version)

    evidence_versions.sort(
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("ascii")
    )
    context = {
        "asOf": body.get("asOf"),
        "policy": body.get("policy"),
        "versions": evidence_versions,
    }
    return promotion_json_fingerprint(context)


def version_evidence_gates(
    record: dict[str, Any],
    policy_valid: bool,
    policy: dict[str, Any],
    as_of: Optional[datetime],
) -> tuple[set[str], Optional[dict[str, Any]]]:
    reasons: set[str] = set()
    if not policy_valid:
        reasons.add("INVALID_POLICY")

    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        reasons.add("MISSING_EVALUATION")
        return reasons, None

    created_at = parse_timestamp(evaluation.get("createdAt"))
    if as_of is None or created_at is None:
        reasons.add("INVALID_TIMESTAMP")
    elif policy["maxAgeSeconds"] is not None:
        if created_at > as_of:
            reasons.add("FUTURE_EVALUATION")
        else:
            try:
                earliest = as_of - timedelta(seconds=policy["maxAgeSeconds"])
            except OverflowError:
                earliest = datetime.min.replace(tzinfo=timezone.utc)
            if created_at < earliest:
                reasons.add("STALE_EVALUATION")

    registered_artifact = record.get("artifactDigest")
    evaluation_artifact = evaluation.get("artifactDigest")
    if (
        not is_utf8_string(registered_artifact)
        or registered_artifact == ""
        or evaluation_artifact != registered_artifact
    ):
        reasons.add("ARTIFACT_MISMATCH")

    if (
        policy["datasetDigest"] is not None
        and evaluation.get("datasetDigest") != policy["datasetDigest"]
    ):
        reasons.add("DATASET_MISMATCH")
    if (
        policy["schemaDigest"] is not None
        and evaluation.get("schemaDigest") != policy["schemaDigest"]
    ):
        reasons.add("SCHEMA_MISMATCH")

    accuracy = evaluation.get("accuracy")
    accuracy_valid = is_finite_number(accuracy)
    if not accuracy_valid:
        reasons.add("NON_FINITE")
    elif not 0 <= accuracy <= 1:
        reasons.add("METRIC_RANGE")
        accuracy_valid = False
    elif (
        policy["accuracyFloor"] is not None
        and accuracy < policy["accuracyFloor"]
    ):
        reasons.add("ACCURACY_FLOOR")

    latency = evaluation.get("latencyMs")
    latency_valid = is_finite_number(latency)
    if not latency_valid:
        reasons.add("NON_FINITE")
    elif latency < 0:
        reasons.add("METRIC_RANGE")
        latency_valid = False
    elif (
        policy["maxLatencyMs"] is not None
        and latency > policy["maxLatencyMs"]
    ):
        reasons.add("LATENCY_LIMIT")

    size = evaluation.get("sizeBytes")
    size_valid = is_safe_integer(size)
    if not size_valid:
        if not is_finite_number(size):
            reasons.add("NON_FINITE")
        else:
            reasons.add("METRIC_RANGE")
    elif (
        policy["maxSizeBytes"] is not None
        and size > policy["maxSizeBytes"]
    ):
        reasons.add("SIZE_LIMIT")

    required_slices = policy["requiredSlices"]
    if required_slices is not None:
        evaluation_slices = evaluation.get("slices")
        for slice_name in sorted(
            required_slices,
            key=lambda value: value.encode("utf-8"),
        ):
            if (
                not isinstance(evaluation_slices, dict)
                or slice_name not in evaluation_slices
            ):
                reasons.add(f"MISSING_SLICE:{slice_name}")
                continue
            slice_value = evaluation_slices[slice_name]
            if not is_finite_number(slice_value) or not 0 <= slice_value <= 1:
                reasons.add(f"SLICE_RANGE:{slice_name}")
            elif slice_value < required_slices[slice_name]:
                reasons.add(f"SLICE_FLOOR:{slice_name}")

    parsed = {
        "evaluation": evaluation,
        "accuracy": accuracy,
        "latencyMs": latency,
        "sizeBytes": size,
    }
    return reasons, parsed


def make_promotion_response(
    body: dict[str, Any], stored_alias: Optional[str]
) -> tuple[dict[str, Any], Optional[str]]:
    supplied_champion = body["championVersion"]
    effective_champion = stored_alias if stored_alias is not None else supplied_champion
    policy_valid, parsed_policy = parse_promotion_policy(body["policy"])
    as_of = parse_timestamp(body.get("asOf"))

    versions = body["versions"]
    raw_versions = [
        item.get("version") if isinstance(item, dict) else None
        for item in versions
    ]
    occurrence_counts: dict[str, int] = {}
    for value in raw_versions:
        key = occurrence_key(value)
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1

    failed_accumulator: dict[str, set[str]] = {}
    lookup: dict[str, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []

    for item, raw_version in zip(versions, raw_versions):
        output_key = failed_gate_key(raw_version)
        failed_accumulator.setdefault(output_key, set())
        base_reasons: set[str] = set()
        version_number = canonical_version_number(raw_version)
        if version_number is None:
            base_reasons.add("INVALID_VERSION")
        if occurrence_counts[occurrence_key(raw_version)] > 1:
            base_reasons.add("DUPLICATE_VERSION")

        failed_accumulator[output_key].update(base_reasons)
        if base_reasons:
            continue

        assert isinstance(item, dict) and isinstance(raw_version, str)
        reasons, parsed = version_evidence_gates(
            item,
            policy_valid,
            parsed_policy,
            as_of,
        )
        failed_accumulator[output_key].update(reasons)
        lookup[raw_version] = {
            "record": item,
            "versionNumber": version_number,
            "reasons": reasons,
            "parsed": parsed,
        }
        if not reasons and parsed is not None:
            eligible.append(
                {
                    "version": raw_version,
                    "versionNumber": version_number,
                    "evaluation": parsed["evaluation"],
                    "accuracy": parsed["accuracy"],
                    "latencyMs": parsed["latencyMs"],
                    "sizeBytes": parsed["sizeBytes"],
                }
            )

    eligible.sort(
        key=lambda candidate: (
            -candidate["accuracy"],
            candidate["latencyMs"],
            candidate["sizeBytes"],
            candidate["versionNumber"],
        )
    )
    eligible_versions = [candidate["version"] for candidate in eligible]

    failed_gates: dict[str, list[str]] = {}
    for version_key in sorted(
        failed_accumulator,
        key=lambda value: value.encode("utf-8"),
    ):
        failed_gates[version_key] = sorted_codes(failed_accumulator[version_key])

    action = "block"
    selected_version: Optional[str] = None
    selected_evidence: Optional[dict[str, Any]] = None
    alias_mutation: Optional[dict[str, str]] = None
    new_alias: Optional[str] = None

    champion_lookup = lookup.get(effective_champion)
    champion_is_eligible = (
        champion_lookup is not None
        and not champion_lookup["reasons"]
        and champion_lookup["parsed"] is not None
    )

    if champion_is_eligible:
        champion_candidate = next(
            candidate
            for candidate in eligible
            if candidate["version"] == effective_champion
        )
        challenger = eligible[0]
        improvement = round(
            challenger["accuracy"] - champion_candidate["accuracy"],
            12,
        )

        if (
            challenger["version"] != effective_champion
            and improvement >= parsed_policy["minImprovement"]
        ):
            action = "promote"
            selected_version = challenger["version"]
            selected_evidence = copy.deepcopy(challenger["evaluation"])
            alias_mutation = {
                "alias": "champion",
                "version": challenger["version"],
            }
            new_alias = challenger["version"]
        else:
            action = "retain"
            selected_version = effective_champion
            selected_evidence = copy.deepcopy(champion_candidate["evaluation"])

    response = {
        "action": action,
        "championVersion": effective_champion,
        "selectedVersion": selected_version,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": alias_mutation,
        "evidence": selected_evidence,
    }
    return response, new_alias


@app.post("/promote")
async def promote(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if (
        not isinstance(body, dict)
        or "policy" not in body
        or not isinstance(body.get("versions"), list)
        or not isinstance(body.get("championVersion"), str)
    ):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    context_key = promotion_context_fingerprint(body)
    async with PROMOTION_ALIASES_LOCK:
        stored_alias = PROMOTION_ALIASES.get(context_key)
        response, new_alias = make_promotion_response(body, stored_alias)
        if new_alias is not None:
            PROMOTION_ALIASES[context_key] = new_alias

    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Assignment 4: minimal adaptation choice and PEFT repair gate
# ---------------------------------------------------------------------------


ADAPTATION_PRIORITY = ("prompt_only", "retrieval", "lora", "qlora")
REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_choose_policy(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    min_quality = value.get("minQuality")
    freshness_required = value.get("freshnessRequired")
    max_latency = value.get("maxLatencyMs")
    max_memory = value.get("maxMemoryMb")
    max_labeled = value.get("maxLabeledExamples")
    max_cost = value.get("maxTotalCost")
    horizon = value.get("horizonRequests")

    if not is_unit_floor(min_quality):
        return None
    if type(freshness_required) is not bool:
        return None
    if not is_finite_number(max_latency) or max_latency < 0:
        return None
    if not is_finite_number(max_memory) or max_memory < 0:
        return None
    if not is_safe_integer(max_labeled):
        return None
    if not is_finite_number(max_cost) or max_cost < 0:
        return None
    if not is_safe_integer(horizon):
        return None

    return {
        "minQuality": min_quality,
        "freshnessRequired": freshness_required,
        "maxLatencyMs": max_latency,
        "maxMemoryMb": max_memory,
        "maxLabeledExamples": max_labeled,
        "maxTotalCost": max_cost,
        "horizonRequests": horizon,
    }


def parse_adaptation_candidate(value: Any, expected_name: str) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("name") != expected_name:
        return None

    available = value.get("available")
    quality = value.get("quality")
    freshness = value.get("freshness")
    latency = value.get("latencyMs")
    memory = value.get("memoryMb")
    labeled = value.get("labeledExamples")
    one_time = value.get("oneTimeCost")
    recurring = value.get("recurringCost")

    if type(available) is not bool:
        return None
    if not is_unit_floor(quality):
        return None
    if type(freshness) is not bool:
        return None
    if not is_finite_number(latency) or latency < 0:
        return None
    if not is_finite_number(memory) or memory < 0:
        return None
    if not is_safe_integer(labeled):
        return None
    if not is_finite_number(one_time) or one_time < 0:
        return None
    if not is_finite_number(recurring) or recurring < 0:
        return None

    return {
        "name": expected_name,
        "available": available,
        "quality": quality,
        "freshness": freshness,
        "latencyMs": latency,
        "memoryMb": memory,
        "labeledExamples": labeled,
        "oneTimeCost": one_time,
        "recurringCost": recurring,
    }


def handle_adapt_choose(body: dict[str, Any]) -> JSONResponse:
    total_costs: dict[str, Optional[float | int]] = {
        name: None for name in ADAPTATION_PRIORITY
    }
    candidate_reasons: dict[str, set[str]] = {
        name: set() for name in ADAPTATION_PRIORITY
    }

    policy = parse_choose_policy(body.get("policy"))
    supplied_candidates = body.get("candidates")
    roster_valid = isinstance(supplied_candidates, list)
    candidate_map: dict[str, dict[str, Any]] = {}

    if roster_valid:
        for candidate in supplied_candidates:
            if not isinstance(candidate, dict):
                roster_valid = False
                continue
            name = candidate.get("name")
            if name not in ADAPTATION_PRIORITY or name in candidate_map:
                roster_valid = False
                continue
            candidate_map[name] = candidate
        if set(candidate_map) != set(ADAPTATION_PRIORITY):
            roster_valid = False

    if policy is None or not roster_valid:
        for name in ADAPTATION_PRIORITY:
            candidate_reasons[name].add("INVALID_INPUT")
    else:
        for name in ADAPTATION_PRIORITY:
            candidate = parse_adaptation_candidate(candidate_map[name], name)
            if candidate is None:
                candidate_reasons[name].add("INVALID_INPUT")
                continue

            try:
                raw_total = (
                    candidate["oneTimeCost"]
                    + policy["horizonRequests"] * candidate["recurringCost"]
                )
                if not is_finite_number(raw_total):
                    raise ArithmeticError
                total_cost = round(raw_total, 12)
            except (ArithmeticError, OverflowError):
                candidate_reasons[name].add("INVALID_INPUT")
                continue

            total_costs[name] = total_cost
            if not candidate["available"]:
                candidate_reasons[name].add("UNAVAILABLE")
            if candidate["quality"] < policy["minQuality"]:
                candidate_reasons[name].add("QUALITY_FLOOR")
            if policy["freshnessRequired"] and not candidate["freshness"]:
                candidate_reasons[name].add("FRESHNESS_REQUIRED")
            if candidate["latencyMs"] > policy["maxLatencyMs"]:
                candidate_reasons[name].add("LATENCY_LIMIT")
            if candidate["memoryMb"] > policy["maxMemoryMb"]:
                candidate_reasons[name].add("MEMORY_LIMIT")
            if candidate["labeledExamples"] > policy["maxLabeledExamples"]:
                candidate_reasons[name].add("DATA_LIMIT")
            if total_cost > policy["maxTotalCost"]:
                candidate_reasons[name].add("COST_LIMIT")

    reason_codes = {
        name: sorted_codes(candidate_reasons[name])
        for name in ADAPTATION_PRIORITY
    }
    eligible = [
        name for name in ADAPTATION_PRIORITY if not reason_codes[name]
    ]
    response = {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }
    return JSONResponse(response)


def valid_id_list(value: Any) -> tuple[bool, set[str]]:
    if not isinstance(value, list) or not value:
        return False, set()
    if not all(is_utf8_string(item) and item != "" for item in value):
        return False, set()
    if len(set(value)) != len(value):
        return False, set()
    return True, set(value)


def looks_like_full_model_artifact(filename: str) -> bool:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if basename in {
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    }:
        return True
    return (
        (basename.startswith("pytorch_model-") and basename.endswith(".bin"))
        or (basename.startswith("model-") and basename.endswith(".safetensors"))
    )


def handle_adapt_repair(body: dict[str, Any]) -> JSONResponse:
    reasons: set[str] = set()

    tokens = body.get("tokens")
    token_list_has_length = isinstance(tokens, list) and bool(tokens)
    tokens_valid = token_list_has_length
    if tokens_valid:
        for token in tokens:
            if not isinstance(token, dict):
                tokens_valid = False
                break
            if not is_safe_integer(token.get("id")):
                tokens_valid = False
                break
            if token.get("role") not in {"system", "user", "assistant"}:
                tokens_valid = False
                break
            if type(token.get("padding")) is not bool:
                tokens_valid = False
                break
            if not isinstance(token.get("text"), str):
                tokens_valid = False
                break

    if tokens_valid:
        labels = [
            token["id"]
            if token["role"] == "assistant" and token["padding"] is False
            else -100
            for token in tokens
        ]
    else:
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []
        reasons.add("INVALID_TOKEN")

    template_pass = (
        type(body.get("templateApplications")) is int
        and body["templateApplications"] == 1
    )
    if not template_pass:
        reasons.add("CHAT_TEMPLATE_COUNT")

    parameters = body.get("parameters")
    allowed_targets_value = body.get("allowedTargets")
    parameters_valid = isinstance(parameters, list)
    allowed_targets_valid = (
        isinstance(allowed_targets_value, list)
        and bool(allowed_targets_value)
        and all(
            is_utf8_string(target) and target != ""
            for target in allowed_targets_value
        )
        and len(set(allowed_targets_value)) == len(allowed_targets_value)
    )
    allowed_targets = set(allowed_targets_value) if allowed_targets_valid else set()
    parsed_parameters: list[dict[str, Any]] = []
    seen_parameter_names: set[str] = set()

    if parameters_valid:
        for parameter in parameters:
            if not isinstance(parameter, dict):
                parameters_valid = False
                break
            name = parameter.get("name")
            target = parameter.get("target")
            numel = parameter.get("numel")
            if not is_utf8_string(name) or name in seen_parameter_names:
                parameters_valid = False
                break
            if not is_utf8_string(target):
                parameters_valid = False
                break
            if not is_safe_integer(numel, minimum=1):
                parameters_valid = False
                break
            seen_parameter_names.add(name)
            parsed_parameters.append(
                {"name": name, "target": target, "numel": numel}
            )

    trainable_parameters: list[dict[str, Any]] = []
    if parameters_valid and allowed_targets_valid:
        trainable_parameters = [
            parameter
            for parameter in parsed_parameters
            if parameter["target"] in allowed_targets
            and parameter["name"].endswith(
                (".lora_A.weight", ".lora_B.weight")
            )
        ]
        if not trainable_parameters:
            parameters_valid = False

    safe_trainable_count = 0
    if parameters_valid and allowed_targets_valid:
        for parameter in trainable_parameters:
            if safe_trainable_count > SAFE_INTEGER_MAX - parameter["numel"]:
                parameters_valid = False
                break
            safe_trainable_count += parameter["numel"]

    if not parameters_valid or not allowed_targets_valid:
        reasons.add("INVALID_PARAMETER")
        trainable_names: list[str] = []
        trainable_count = 0
        parameter_config_pass = False
    else:
        trainable_names = sorted(
            (parameter["name"] for parameter in trainable_parameters),
            key=lambda value: value.encode("utf-8"),
        )
        trainable_count = safe_trainable_count
        parameter_config_pass = True

    inference_mode_pass = body.get("inferenceMode") is False
    if not inference_mode_pass:
        reasons.add("INFERENCE_MODE")
    peft_config_pass = parameter_config_pass and inference_mode_pass

    artifact_files_value = body.get("artifactFiles")
    artifact_list_valid = (
        isinstance(artifact_files_value, list)
        and all(is_utf8_string(filename) for filename in artifact_files_value)
    )
    adapter_files = (
        sorted(set(artifact_files_value), key=lambda value: value.encode("utf-8"))
        if artifact_list_valid
        else []
    )
    adapter_file_set_pass = (
        artifact_list_valid
        and sorted(artifact_files_value, key=lambda value: value.encode("utf-8"))
        == list(REQUIRED_ADAPTER_FILES)
    )
    if not adapter_file_set_pass:
        reasons.add("ADAPTER_FILE_SET")
    if artifact_list_valid and any(
        looks_like_full_model_artifact(filename)
        for filename in artifact_files_value
    ):
        reasons.add("FULL_MODEL_ARTIFACT")

    checkpoint = body.get("checkpoint")
    checkpoint_keys = {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition",
    }
    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and checkpoint_keys.issubset(checkpoint.keys())
    )
    if not checkpoint_complete:
        reasons.add("INCOMPLETE_CHECKPOINT")

    base_revision = body.get("baseRevision")
    mutable_base_ok = (
        isinstance(base_revision, str)
        and HEX40_RE.fullmatch(base_revision) is not None
    )
    if not mutable_base_ok:
        reasons.add("MUTABLE_BASE_REVISION")

    expected_digests = body.get("expectedDigests")
    lineage_digests_pass = isinstance(expected_digests, dict)
    for digest_name in ("datasetDigest", "codeDigest", "configDigest"):
        actual_digest = body.get(digest_name)
        expected_digest = (
            expected_digests.get(digest_name)
            if isinstance(expected_digests, dict)
            else None
        )
        if (
            not isinstance(actual_digest, str)
            or HEX64_RE.fullmatch(actual_digest) is None
            or not isinstance(expected_digest, str)
            or HEX64_RE.fullmatch(expected_digest) is None
            or actual_digest != expected_digest
        ):
            lineage_digests_pass = False
    if not lineage_digests_pass:
        reasons.add("LINEAGE_MISMATCH")
    lineage_pass = mutable_base_ok and lineage_digests_pass

    micro_batch = body.get("microBatch")
    accumulation = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected_batch = body.get("expectedEffectiveBatch")
    batch_factors_valid = all(
        is_safe_integer(value, minimum=1)
        for value in (micro_batch, accumulation, replicas, expected_batch)
    )
    effective_batch_pass = False
    if batch_factors_valid:
        product = micro_batch * accumulation * replicas
        effective_batch_pass = (
            product <= SAFE_INTEGER_MAX and product == expected_batch
        )
    if not effective_batch_pass:
        reasons.add("EFFECTIVE_BATCH_MISMATCH")

    train_ids_valid, train_ids = valid_id_list(body.get("trainRowIds"))
    eval_ids_valid, eval_ids = valid_id_list(body.get("evalRowIds"))
    eval_isolated = (
        train_ids_valid
        and eval_ids_valid
        and train_ids.isdisjoint(eval_ids)
    )
    if not eval_isolated:
        reasons.add("EVAL_LEAKAGE")

    evaluation_deterministic = body.get("dropoutActiveDuringEval") is False
    if not evaluation_deterministic:
        reasons.add("EVAL_DROPOUT_ACTIVE")

    uninterrupted = body.get("uninterruptedWeights")
    resumed = body.get("resumedWeights")
    tolerance = body.get("resumeTolerance")
    resume_arrays_valid = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and bool(uninterrupted)
        and len(uninterrupted) == len(resumed)
        and all(is_finite_number(value) for value in uninterrupted)
        and all(is_finite_number(value) for value in resumed)
        and is_finite_number(tolerance)
        and tolerance >= 0
    )
    resume_values_pass = False
    if resume_arrays_valid:
        resume_values_pass = all(
            abs(left - right) <= tolerance
            for left, right in zip(uninterrupted, resumed)
        )
    if not resume_values_pass:
        reasons.add("RESUME_DIVERGENCE")
    # Checkpoint ownership and numerical resume equivalence are reported as two
    # independent fields: checkpointComplete and resumePass.
    resume_pass = resume_values_pass

    response = {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sorted_codes(reasons),
    }
    return JSONResponse(response)


@app.post("/adapt")
async def adapt(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict) or body.get("operation") not in {"choose", "repair"}:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if body["operation"] == "choose":
        return handle_adapt_choose(body)
    return handle_adapt_repair(body)


# ---------------------------------------------------------------------------
# Assignment 5: stateful quantization freeze and admission gate
# ---------------------------------------------------------------------------


def build_file_inventory(files: Any) -> tuple[bool, list[dict[str, Any]], Optional[int], Optional[str]]:
    files_valid = isinstance(files, dict) and bool(files)
    if files_valid:
        for filename, content in files.items():
            if not is_utf8_string(filename) or filename == "":
                files_valid = False
                break
            if not is_utf8_string(content):
                files_valid = False
                break

    if not files_valid:
        return False, [], None, None

    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for filename in sorted(files, key=lambda value: value.encode("utf-8")):
        exact_bytes = files[filename].encode("utf-8")
        byte_count = len(exact_bytes)
        if total_bytes > SAFE_INTEGER_MAX - byte_count:
            return False, [], None, None
        total_bytes += byte_count
        inventory.append(
            {
                "name": filename,
                "bytes": byte_count,
                "sha256": hashlib.sha256(exact_bytes).hexdigest(),
            }
        )

    package_digest = hashlib.sha256(
        compact_json(inventory).encode("utf-8")
    ).hexdigest()
    return True, inventory, total_bytes, package_digest


def make_quantize_freeze_response(body: dict[str, Any]) -> dict[str, Any]:
    freeze_id = body.get("freezeId")
    output_freeze_id = freeze_id if is_utf8_string(freeze_id) else None
    calibration_digest = body.get("calibrationDigest")
    tokenizer_digest = body.get("tokenizerDigest")
    allowed_value = body.get("allowedUnsupportedReasons")

    freeze_id_valid = is_utf8_string(freeze_id) and 1 <= len(freeze_id) <= 128
    calibration_valid = (
        is_utf8_string(calibration_digest) and calibration_digest != ""
    )
    tokenizer_valid = is_utf8_string(tokenizer_digest) and tokenizer_digest != ""
    allowed_valid = (
        isinstance(allowed_value, list)
        and all(is_utf8_string(reason) and reason != "" for reason in allowed_value)
        and len(set(allowed_value)) == len(allowed_value)
    )
    allowed_reasons = set(allowed_value) if allowed_valid else set()

    candidate_names: list[Any] = [
        candidate.get("name") if isinstance(candidate, dict) else None
        for candidate in body["candidates"]
    ]
    names_valid = (
        all(is_utf8_string(name) and name != "" for name in candidate_names)
        and len(set(candidate_names)) == len(candidate_names)
    )
    global_valid = (
        freeze_id_valid
        and calibration_valid
        and tokenizer_valid
        and allowed_valid
        and names_valid
    )

    frozen_candidates: list[dict[str, Any]] = []
    for supplied_candidate, supplied_name in zip(body["candidates"], candidate_names):
        candidate = supplied_candidate if isinstance(supplied_candidate, dict) else {}
        output_name = supplied_name if is_utf8_string(supplied_name) else None
        reasons: set[str] = set()
        if not global_valid:
            reasons.add("INVALID_INPUT")

        files_valid, inventory, total_bytes, package_digest = build_file_inventory(
            candidate.get("files")
        )
        if not files_valid:
            reasons.add("INVALID_INPUT")

        loadable = candidate.get("loadable")
        candidate_calibration = candidate.get("calibrationDigest")
        candidate_tokenizer = candidate.get("tokenizerDigest")
        unsupported_reason = candidate.get("unsupportedReason")

        candidate_fields_valid = (
            isinstance(supplied_candidate, dict)
            and is_utf8_string(supplied_name)
            and supplied_name != ""
            and type(loadable) is bool
            and is_utf8_string(candidate_calibration)
            and candidate_calibration != ""
            and is_utf8_string(candidate_tokenizer)
            and candidate_tokenizer != ""
            and (
                unsupported_reason is None
                or unsupported_reason == ""
                or (is_utf8_string(unsupported_reason) and unsupported_reason != "")
            )
        )
        if not candidate_fields_valid:
            reasons.add("INVALID_INPUT")

        has_unsupported_reason = (
            isinstance(unsupported_reason, str) and unsupported_reason != ""
        )
        allowed_unsupported = (
            has_unsupported_reason
            and allowed_valid
            and unsupported_reason in allowed_reasons
        )

        if candidate_fields_valid and calibration_valid and tokenizer_valid:
            if allowed_unsupported:
                pass
            else:
                if has_unsupported_reason:
                    reasons.add("UNALLOWED_UNSUPPORTED_REASON")
                if not loadable:
                    reasons.add("NOT_LOADABLE")
                if candidate_calibration != calibration_digest:
                    reasons.add("CALIBRATION_MISMATCH")
                if candidate_tokenizer != tokenizer_digest:
                    reasons.add("TOKENIZER_MISMATCH")

        if reasons:
            status = "invalid"
        elif allowed_unsupported:
            status = "unsupported"
        else:
            status = "frozen"

        frozen_candidates.append(
            {
                "name": output_name,
                "status": status,
                "inventory": inventory,
                "totalBytes": total_bytes,
                "packageDigest": package_digest,
                "reasonCodes": sorted_codes(reasons),
            }
        )

    frozen_candidates.sort(
        key=lambda item: (
            item["name"].encode("utf-8") if is_utf8_string(item["name"]) else b"",
            compact_json(item).encode("utf-8"),
        )
    )
    return {
        "freezeId": output_freeze_id,
        "candidates": frozen_candidates,
    }


async def handle_quantize_freeze(body: dict[str, Any]) -> JSONResponse:
    freeze_id = body.get("freezeId")
    fingerprint = selection_fingerprint(body)

    async with QUANTIZE_FREEZES_LOCK:
        if is_valid_run_id(freeze_id) and freeze_id in QUANTIZE_FREEZES:
            stored = QUANTIZE_FREEZES[freeze_id]
            if stored["fingerprint"] == fingerprint:
                return JSONResponse(copy.deepcopy(stored["response"]))
            return JSONResponse(
                {"error": "FREEZE_ID_CONFLICT"},
                status_code=409,
            )

        response = make_quantize_freeze_response(body)
        if is_valid_run_id(freeze_id):
            QUANTIZE_FREEZES[freeze_id] = {
                "fingerprint": fingerprint,
                "response": copy.deepcopy(response),
            }

    return JSONResponse(response)


def recompute_quantized_manifest(
    candidate: Any,
) -> tuple[bool, Optional[int]]:
    if not isinstance(candidate, dict):
        return False, None
    inventory = candidate.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        return False, None

    seen_names: set[str] = set()
    previous_name_bytes: Optional[bytes] = None
    computed_total = 0
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"}:
            return False, None
        name = item.get("name")
        byte_count = item.get("bytes")
        sha256_value = item.get("sha256")
        if not is_utf8_string(name) or name == "" or name in seen_names:
            return False, None
        name_bytes = name.encode("utf-8")
        if previous_name_bytes is not None and name_bytes <= previous_name_bytes:
            return False, None
        if not is_safe_integer(byte_count):
            return False, None
        if (
            not isinstance(sha256_value, str)
            or HEX64_RE.fullmatch(sha256_value) is None
        ):
            return False, None
        if computed_total > SAFE_INTEGER_MAX - byte_count:
            return False, None
        computed_total += byte_count
        seen_names.add(name)
        previous_name_bytes = name_bytes

    recomputed_package = hashlib.sha256(
        compact_json(inventory).encode("utf-8")
    ).hexdigest()
    manifest_valid = (
        is_safe_integer(candidate.get("totalBytes"))
        and candidate["totalBytes"] == computed_total
        and isinstance(candidate.get("packageDigest"), str)
        and candidate["packageDigest"] == recomputed_package
    )
    return manifest_valid, computed_total


def parse_quantize_policy(
    value: dict[str, Any], candidate_names: list[str]
) -> tuple[bool, dict[str, Any]]:
    max_bytes = value.get("maxBytes")
    aggregate_floor = value.get("aggregateFloor")
    required_value = value.get("requiredSlices")
    max_latency = value.get("maxLatencyMs")
    candidate_order = value.get("candidateOrder")

    max_bytes_valid = is_safe_integer(max_bytes)
    aggregate_valid = is_unit_floor(aggregate_floor)
    max_latency_valid = is_finite_number(max_latency) and max_latency >= 0

    required_valid = isinstance(required_value, dict)
    required_slices: Optional[dict[str, float | int]] = None
    if required_valid:
        required_slices = {}
        for name, floor in required_value.items():
            if not is_utf8_string(name) or name == "" or not is_unit_floor(floor):
                required_valid = False
                required_slices = None
                break
            required_slices[name] = floor

    order_valid = (
        isinstance(candidate_order, list)
        and all(is_utf8_string(name) and name != "" for name in candidate_order)
        and len(set(candidate_order)) == len(candidate_order)
        and set(candidate_order) == set(candidate_names)
    )
    policy_valid = all(
        (
            max_bytes_valid,
            aggregate_valid,
            required_valid,
            max_latency_valid,
            order_valid,
        )
    )
    return policy_valid, {
        "maxBytes": max_bytes if max_bytes_valid else None,
        "aggregateFloor": aggregate_floor if aggregate_valid else None,
        "requiredSlices": required_slices,
        "maxLatencyMs": max_latency if max_latency_valid else None,
        "candidateOrder": candidate_order if order_valid else None,
    }


async def handle_quantize_select(body: dict[str, Any]) -> JSONResponse:
    freeze_id = body.get("freezeId")
    output_freeze_id = freeze_id if is_utf8_string(freeze_id) else None
    supplied_candidates = body["candidates"]

    candidate_names_valid = all(
        isinstance(candidate, dict)
        and is_utf8_string(candidate.get("name"))
        and candidate["name"] != ""
        for candidate in supplied_candidates
    )
    candidate_names = (
        [candidate["name"] for candidate in supplied_candidates]
        if candidate_names_valid
        else []
    )
    if candidate_names_valid and len(set(candidate_names)) != len(candidate_names):
        candidate_names_valid = False

    policy_valid, policy = parse_quantize_policy(
        body["policy"], candidate_names if candidate_names_valid else []
    )

    latencies_value = body.get("latencies")
    latencies_shape_valid = (
        candidate_names_valid
        and isinstance(latencies_value, dict)
        and set(latencies_value) == set(candidate_names)
        and all(
            is_finite_number(latencies_value[name])
            and latencies_value[name] >= 0
            for name in candidate_names
        )
    )
    if not latencies_shape_valid:
        policy_valid = False

    async with QUANTIZE_FREEZES_LOCK:
        stored = (
            copy.deepcopy(QUANTIZE_FREEZES.get(freeze_id))
            if is_valid_run_id(freeze_id)
            else None
        )

    lineage_valid = (
        stored is not None
        and supplied_candidates == stored["response"]["candidates"]
    )

    rows = body["rows"]
    rows_structurally_valid = bool(rows)
    parsed_rows: list[dict[str, Any]] = []
    if rows_structurally_valid:
        for row in rows:
            if not isinstance(row, dict):
                rows_structurally_valid = False
                break
            label = row.get("label")
            slice_name = row.get("slice")
            predictions = row.get("predictions")
            if type(label) is not int or label not in {0, 1}:
                rows_structurally_valid = False
                break
            if not is_utf8_string(slice_name) or slice_name == "":
                rows_structurally_valid = False
                break
            if not isinstance(predictions, dict):
                rows_structurally_valid = False
                break
            parsed_rows.append(
                {"label": label, "slice": slice_name, "predictions": predictions}
            )

    if candidate_names_valid and policy["candidateOrder"] is not None:
        candidate_by_name = {
            candidate["name"]: candidate for candidate in supplied_candidates
        }
        ordered_candidates = [
            candidate_by_name[name] for name in policy["candidateOrder"]
        ]
        order_index = {
            name: index for index, name in enumerate(policy["candidateOrder"])
        }
    else:
        ordered_candidates = sorted(
            supplied_candidates,
            key=lambda candidate: (
                candidate.get("name", "").encode("utf-8")
                if isinstance(candidate, dict)
                and is_utf8_string(candidate.get("name"))
                else b"",
                json.dumps(
                    candidate,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=True,
                ).encode("ascii"),
            ),
        )
        order_index = {}

    result_entries: list[dict[str, Any]] = []
    required_slices = policy["requiredSlices"]
    for candidate in ordered_candidates:
        candidate_object = candidate if isinstance(candidate, dict) else {}
        name = candidate_object.get("name")
        output_name = name if is_utf8_string(name) else None
        reasons: set[str] = set()

        if candidate_object.get("status") != "frozen":
            reasons.add("NOT_FROZEN")
        if not lineage_valid:
            reasons.add("INVALID_LINEAGE")
        if not policy_valid:
            reasons.add("INVALID_POLICY")

        manifest_valid, recomputed_total = recompute_quantized_manifest(
            candidate_object
        )
        if not manifest_valid:
            reasons.add("INVALID_MANIFEST")

        latency_value = (
            latencies_value.get(name)
            if isinstance(latencies_value, dict) and isinstance(name, str)
            else None
        )
        latency_valid = is_finite_number(latency_value) and latency_value >= 0
        output_latency = latency_value if latency_valid else None

        predictions_valid = rows_structurally_valid and isinstance(name, str)
        candidate_rows: list[dict[str, Any]] = []
        if predictions_valid:
            for row in parsed_rows:
                prediction = row["predictions"].get(name)
                if type(prediction) is not int or prediction not in {0, 1}:
                    predictions_valid = False
                    candidate_rows = []
                    break
                candidate_rows.append(
                    {
                        "label": row["label"],
                        "prediction": prediction,
                        "slice": row["slice"],
                    }
                )

        if required_slices is not None:
            slice_values: dict[str, Optional[float]] = {
                slice_name: None
                for slice_name in sorted(
                    required_slices,
                    key=lambda value: value.encode("utf-8"),
                )
            }
        else:
            slice_values = {}

        aggregate: Optional[float] = None
        if not predictions_valid:
            reasons.add("INVALID_PREDICTIONS")
        else:
            aggregate = rounded_accuracy(candidate_rows)
            if (
                policy["aggregateFloor"] is not None
                and aggregate < policy["aggregateFloor"]
            ):
                reasons.add("AGGREGATE_FLOOR")

            if required_slices is not None:
                for slice_name in slice_values:
                    matching_rows = [
                        row for row in candidate_rows if row["slice"] == slice_name
                    ]
                    if not matching_rows:
                        reasons.add(f"MISSING_SLICE:{slice_name}")
                        continue
                    slice_metric = rounded_accuracy(matching_rows)
                    slice_values[slice_name] = slice_metric
                    if slice_metric < required_slices[slice_name]:
                        reasons.add(f"SLICE_FLOOR:{slice_name}")

        if (
            recomputed_total is not None
            and policy["maxBytes"] is not None
            and recomputed_total > policy["maxBytes"]
        ):
            reasons.add("SIZE_LIMIT")
        if (
            latency_valid
            and policy["maxLatencyMs"] is not None
            and latency_value > policy["maxLatencyMs"]
        ):
            reasons.add("LATENCY_LIMIT")

        reason_codes = sorted_codes(reasons)
        result_entries.append(
            {
                "name": output_name,
                "aggregate": aggregate,
                "slices": slice_values,
                "totalBytes": recomputed_total,
                "latencyMs": output_latency,
                "admitted": not reason_codes,
                "reasonCodes": reason_codes,
            }
        )

    admitted_results = [result for result in result_entries if result["admitted"]]
    selected: Optional[str] = None
    package_manifest: Optional[dict[str, Any]] = None
    if admitted_results:
        winner = min(
            admitted_results,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(result["name"], SAFE_INTEGER_MAX),
                result["name"].encode("utf-8"),
            ),
        )
        selected = winner["name"]
        if stored is not None:
            for recorded_candidate in stored["response"]["candidates"]:
                if recorded_candidate.get("name") == selected:
                    package_manifest = copy.deepcopy(recorded_candidate)
                    break

    response = {
        "freezeId": output_freeze_id,
        "selected": selected,
        "results": result_entries,
        "packageManifest": package_manifest,
    }
    return JSONResponse(response)


@app.post("/quantize")
async def quantize(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not isinstance(body, dict) or body.get("phase") not in {"freeze", "select"}:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if body["phase"] == "freeze":
        if not isinstance(body.get("candidates"), list) or not body["candidates"]:
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
        return await handle_quantize_freeze(body)

    if (
        not isinstance(body.get("candidates"), list)
        or not isinstance(body.get("rows"), list)
        or not isinstance(body.get("policy"), dict)
    ):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    return await handle_quantize_select(body)

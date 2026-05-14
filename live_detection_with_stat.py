#!/usr/bin/env python3
import argparse
import os
import pickle
import subprocess
import tempfile
import time
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


# -------------------------
# XML field extraction
# -------------------------

def _safe_name(raw: Optional[str], empty_name: str) -> str:
    if raw is None or raw == "":
        return empty_name
    return raw.replace(".", "_")


def get_nested_field_names(field_element: ET.Element) -> List[str]:
    nested_field_names: List[str] = []
    for nested_field in field_element.findall("field"):
        field_name = _safe_name(nested_field.get("name"), "nested_empty_field_name")
        nested_field_names.extend([
            f"{field_name}_name",
            f"{field_name}_showname",
            f"{field_name}_size",
            f"{field_name}_pos",
            f"{field_name}_show",
            f"{field_name}_value",
        ])
        nested_field_names.extend(get_nested_field_names(nested_field))
    return nested_field_names


def get_packet_field_names(proto: ET.Element) -> List[str]:
    packet_fields: List[str] = []
    for field in proto.findall("field"):
        field_name = _safe_name(field.get("name"), "empty_field_name")
        packet_fields.extend([
            f"{field_name}_name",
            f"{field_name}_showname",
            f"{field_name}_size",
            f"{field_name}_pos",
            f"{field_name}_show",
            f"{field_name}_value",
        ])
        packet_fields.extend(get_nested_field_names(field))
    return packet_fields


def get_nested_fields(field_element: ET.Element) -> List[Optional[str]]:
    nested_fields: List[Optional[str]] = []
    for nested_field in field_element.findall("field"):
        nested_fields.extend([
            nested_field.get("name"),
            nested_field.get("showname"),
            nested_field.get("size"),
            nested_field.get("pos"),
            nested_field.get("show"),
            nested_field.get("value"),
        ])
        nested_fields.extend(get_nested_fields(nested_field))
    return nested_fields


def align_lists(target_cols: List[str], cols: List[str], values: List[Optional[str]]) -> List[Optional[str]]:
    first_idx: Dict[str, int] = {}
    for i, c in enumerate(cols):
        if c not in first_idx:
            first_idx[c] = i
    out: List[Optional[str]] = []
    for c in target_cols:
        if c in first_idx:
            out.append(values[first_idx[c]])
        else:
            out.append("")
    return out


def iter_rrc_protos(packet: ET.Element):
    # 1) top-level nr-rrc
    for proto in packet.findall("proto"):
        if proto.get("name") == "nr-rrc" and proto.get("hide") != "yes":
            yield proto

    # 2) nr-rrc nested under mac-nr
    for top_proto in packet.findall("proto"):
        if top_proto.get("name") == "mac-nr":
            for proto in top_proto.iter("proto"):
                if proto.get("name") == "nr-rrc" and proto.get("hide") != "yes":
                    yield proto


def extract_columns_and_values_from_xml(xml_file: str) -> Tuple[List[str], List[List[Optional[str]]]]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    packets = root.findall("packet")

    # Pass 1: collect union schema
    column_names: set[str] = set()
    for packet in packets:
        for rrc_proto in iter_rrc_protos(packet):
            column_names.update(get_packet_field_names(rrc_proto))
            for nas_proto in rrc_proto.iter("proto"):
                if nas_proto.get("name") == "nas-5gs":
                    column_names.update(get_packet_field_names(nas_proto))

    ordered_columns = sorted(column_names)

    # Pass 2: extract aligned rows
    rows: List[List[Optional[str]]] = []
    for packet in packets:
        for rrc_proto in iter_rrc_protos(packet):
            packet_fields: List[Optional[str]] = []
            cols: List[str] = []

            for field in rrc_proto.findall("field"):
                packet_fields.extend([
                    field.get("name"),
                    field.get("showname"),
                    field.get("size"),
                    field.get("pos"),
                    field.get("show"),
                    field.get("value"),
                ])
                packet_fields.extend(get_nested_fields(field))
            cols.extend(get_packet_field_names(rrc_proto))

            for nas_proto in rrc_proto.iter("proto"):
                if nas_proto.get("name") == "nas-5gs":
                    for field in nas_proto.findall("field"):
                        packet_fields.extend([
                            field.get("name"),
                            field.get("showname"),
                            field.get("size"),
                            field.get("pos"),
                            field.get("show"),
                            field.get("value"),
                        ])
                        packet_fields.extend(get_nested_fields(field))
                    cols.extend(get_packet_field_names(nas_proto))

            rows.append(align_lists(ordered_columns, cols, packet_fields))

    return ordered_columns, rows


# -------------------------
# PDML conversion
# -------------------------

def run_tshark_to_pdml(input_pcap: str, xml_output_file: str) -> None:
    with open(xml_output_file, "w", encoding="utf-8") as f:
        subprocess.run(["tshark", "-r", input_pcap, "-T", "pdml"], stdout=f, check=True)


def prepare_dataframe_from_xml(xml_file: str) -> pd.DataFrame:
    columns, values = extract_columns_and_values_from_xml(xml_file)
    return pd.DataFrame(values, columns=columns)


# -------------------------
# Encoding + padding
# -------------------------

def encode_dataframe(df: pd.DataFrame, category_maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        cmap = category_maps.get(col)
        if cmap is None:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1).astype(np.float32)
            continue
        if isinstance(cmap, dict) and cmap.get("__type__") == "fbs_binary":
            primary_causes = set(cmap.get("__primary__", []))

            def encode_cause(val):
                try:
                    cause = int(float(str(val)))
                    if cause in primary_causes:
                        return 30
                    elif cause > 0:
                        return 0
                    else:
                        return -1
                except (ValueError, TypeError):
                    return -1

            out[col] = out[col].apply(encode_cause).astype(np.float32)
        else:
            cmap_str = {str(k): int(v) for k, v in cmap.items()
                        if not str(k).startswith("__")}
            s = out[col].astype("string")
            out[col] = s.map(cmap_str).fillna(-1).astype(np.float32)
    return out


def pad_sequence_list(x_list: List[np.ndarray], max_len: int, pad_value: float = 0.0) -> Tuple[np.ndarray, int]:
    if len(x_list) == 0:
        raise ValueError("Cannot pad an empty sequence list.")
    n_feat = x_list[0].shape[1]
    X = np.full((len(x_list), max_len, n_feat), pad_value, dtype=np.float32)
    for i, x in enumerate(x_list):
        L = min(len(x), max_len)
        if L > 0:
            X[i, :L, :] = x[:L, :]
    return X, max_len


# -------------------------
# Artifacts
# -------------------------

def load_artifacts(artifacts_dir: str):
    model_path = os.path.join(artifacts_dir, "sequence_model.keras")
    preproc_path = os.path.join(artifacts_dir, "preprocess.pkl")
    encoding_path = os.path.join(artifacts_dir, "feature_encoding.pkl")

    model = keras.models.load_model(model_path, compile=False)

    with open(preproc_path, "rb") as f:
        meta = pickle.load(f)

    with open(encoding_path, "rb") as f:
        enc = pickle.load(f)

    category_maps = enc["category_maps"] if isinstance(enc, dict) and "category_maps" in enc else enc
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]
    max_len = int(meta["max_len"])

    return model, scaler, feature_cols, max_len, category_maps


# -------------------------
# Inference
# -------------------------

def infer_once(
    pcap_path: str,
    model,
    scaler,
    feature_cols: List[str],
    max_len: int,
    category_maps: Dict[str, Dict[str, int]],
    threshold: float
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
    # ↑ add two more return values: elapsed_time, packet_count

    xml_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as tmp:
            xml_path = tmp.name
        run_tshark_to_pdml(pcap_path, xml_path)
        df = prepare_dataframe_from_xml(xml_path)
    finally:
        if xml_path and os.path.exists(xml_path):
            os.remove(xml_path)

    if df.empty or df.shape[0] == 0:
        return None, None, None, None

    packet_count = len(df)   # ← how many packets were in the pcap

    df = encode_dataframe(df, category_maps)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.reindex(columns=feature_cols, fill_value=-1.0)

    x = df.apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    if x.shape[0] == 0:
        return None, None, None, None

    x = scaler.transform(x).astype(np.float32)
    X, _ = pad_sequence_list([x], max_len=max_len, pad_value=0.0)

    # ── Timer wraps only the model.predict() call ──────────────────────────
    t_start = time.perf_counter()
    probs = model.predict(X, verbose=0).ravel()
    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000   # convert to milliseconds

    prob = float(probs[0])
    pred = int(prob >= threshold)
    return prob, pred, elapsed_ms, packet_count


# -------------------------
# Main loop
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=str, default="artifacts_2/")
    parser.add_argument("--interval",      type=float, default=4.0)
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--pcap",          type=str, default="/tmp/ue_mac_nr.pcap")

    # ── NEW ARGUMENTS FOR HARDWARE TESTING ─────────────────────────────────
    parser.add_argument("--scenario",      type=str, default="unknown",
                        help="Label for current test (e.g. 'cause_11', 'benign')")
    parser.add_argument("--ground-truth",  type=int, default=-1, choices=[0, 1],
                        help="Actual label: 0=benign, 1=attack")
    parser.add_argument("--log-csv",       type=str, default=None,
                        help="Path to CSV file for logging results")
    args = parser.parse_args()

    model, scaler, feature_cols, max_len, category_maps = load_artifacts(args.artifacts_dir)

    # ── CSV setup ──────────────────────────────────────────────────────────
    if args.log_csv:
        write_header = not os.path.exists(args.log_csv)
        log_file = open(args.log_csv, "a", newline="")
        import csv
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow([
                "timestamp", "scenario", "ground_truth",
                "probability", "prediction", "correct",
                "elapsed_ms", "packet_count"
            ])
    else:
        writer = None
        log_file = None

    # ── Session tracking ───────────────────────────────────────────────────
    # Detect new UE sessions by monitoring pcap modification time
    last_mtime = 0.0
    session_count = 0
    attacks_detected = 0
    benign_detected = 0
    inference_count = 0
    total_elapsed_ms = 0.0

    print(f"\n{'='*60}")
    print(f"  Hardware Test Mode")
    print(f"  Scenario:     {args.scenario}")
    print(f"  Ground Truth: {'ATTACK (1)' if args.ground_truth == 1 else 'BENIGN (0)' if args.ground_truth == 0 else 'NOT SET'}")
    print(f"  Log CSV:      {args.log_csv or 'disabled'}")
    print(f"{'='*60}\n")

    try:
        while True:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not os.path.exists(args.pcap) or os.path.getsize(args.pcap) == 0:
                print(f"[{ts}] Waiting for pcap...")
                time.sleep(args.interval)
                continue

            current_mtime = os.path.getmtime(args.pcap)

            # ── Detect new session: pcap was rewritten ─────────────────────
            if current_mtime > last_mtime + 1.0:
                # Wait for pcap to stabilize (UE finished registering)
                time.sleep(args.interval)

                # Check again that the file is still there and non-empty
                if not os.path.exists(args.pcap) or os.path.getsize(args.pcap) == 0:
                    last_mtime = current_mtime
                    continue

                session_count += 1
                last_mtime = os.path.getmtime(args.pcap)

                try:
                    prob, pred, elapsed_ms, packet_count = infer_once(
                        pcap_path=args.pcap,
                        model=model,
                        scaler=scaler,
                        feature_cols=feature_cols,
                        max_len=max_len,
                        category_maps=category_maps,
                        threshold=args.threshold
                    )
                except Exception as e:
                    print(f"[{ts}] Warning: inference failed ({e}). Skipping session.")
                    continue

                if prob is None:
                    print(f"[{ts}] No NR-RRC packets in session. Skipping.")
                    continue

                inference_count += 1
                total_elapsed_ms += elapsed_ms
                correct = (pred == args.ground_truth) if args.ground_truth != -1 else None

                if pred == 1:
                    attacks_detected += 1
                    label = "⚠️  FBS ATTACK DETECTED"
                else:
                    benign_detected += 1
                    label = "✅  BENIGN"

                correct_str = ""
                if correct is not None:
                    correct_str = "✓ CORRECT" if correct else "✗ WRONG"

                print(
                    f"[{ts}] Session #{session_count} | {label} | "
                    f"Prob: {prob:.4f} | Pkts: {packet_count} | "
                    f"Time: {elapsed_ms:.1f}ms | {correct_str}"
                )

                # ── Write to CSV ───────────────────────────────────────────
                if writer:
                    writer.writerow([
                        ts,
                        args.scenario,
                        args.ground_truth,
                        f"{prob:.6f}",
                        pred,
                        int(correct) if correct is not None else "",
                        f"{elapsed_ms:.2f}",
                        packet_count
                    ])
                    log_file.flush()

            else:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"  Scenario Summary: {args.scenario}")
        print(f"{'='*60}")
        print(f"  Sessions processed: {session_count}")
        print(f"  Inferences made:    {inference_count}")
        print(f"  Attack detections:  {attacks_detected}")
        print(f"  Benign detections:  {benign_detected}")
        if inference_count > 0:
            print(f"  Avg predict time:   {total_elapsed_ms / inference_count:.1f} ms")
        print(f"{'='*60}\n")
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
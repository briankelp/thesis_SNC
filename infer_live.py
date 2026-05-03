#!/usr/bin/env python3
"""
Usage:
    python infer_live.py --pcap /tmp/ue_mac_nr.pcap
    python infer_live.py --pcap /tmp/ue_mac_nr.pcap --artifacts-dir artifacts_2/ --threshold 0.5
"""

import argparse
import os
import sys
import tempfile
import subprocess
import pickle
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from tensorflow import keras

# ── XML Parsing (embedded from pcap_to_csv_inference) ─────────────────────────

def get_nested_field_names(field_element):
    nested = []
    for f in field_element.findall('field'):
        name = f.get('name', '').replace('.', '_')
        if not name:
            continue
        nested.extend([
            name+'_name', name+'_showname', name+'_size',
            name+'_pos',  name+'_show',     name+'_value'
        ])
        nested.extend(get_nested_field_names(f))
    return nested


def get_nested_fields(field_element):
    nested = []
    for f in field_element.findall('field'):
        if not f.get('name', ''):
            continue
        nested.extend([
            f.get('name'),     f.get('showname'),
            f.get('size'),     f.get('pos'),
            f.get('show'),     f.get('value')
        ])
        nested.extend(get_nested_fields(f))
    return nested


def get_packet_field_names(packet):
    fields = []
    for p1 in packet.findall('proto'):
        if p1.get('name') == 'mac-nr':
            for p2 in p1.findall('proto'):
                if p2.get('name') == 'nr-rrc':
                    fields = []
                    for f in p2.findall('field'):
                        n = f.get('name', '').replace('.', '_')
                        if not n:
                            continue
                        fields.extend([
                            n+'_name', n+'_showname', n+'_size',
                            n+'_pos',  n+'_show',     n+'_value'
                        ])
                        fields.extend(get_nested_field_names(f))
    return fields


def align_lists(master, packet_cols, packet_vals):
    aligned = []
    for col in master:
        if col in packet_cols:
            aligned.append(packet_vals[packet_cols.index(col)])
        else:
            aligned.append('')
    return aligned


def parse_xml_to_df(xml_path: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    packets = root.findall('packet')

    # Build master column list
    column_names = set()
    for pkt in packets:
        column_names.update(get_packet_field_names(pkt))
    column_names = list(column_names)

    rows = []
    for pkt in packets:
        flag = False
        packet_fields = []
        for p1 in pkt.findall('proto'):
            if p1.get('name') == 'mac-nr':
                for p2 in p1.findall('proto'):
                    if p2.get('name') == 'nr-rrc':
                        flag = True
                        packet_fields = []
                        for f in p2.findall('field'):
                            if not f.get('name', ''):
                                continue
                            packet_fields.extend([
                                f.get('name'),     f.get('showname'),
                                f.get('size'),     f.get('pos'),
                                f.get('show'),     f.get('value')
                            ])
                            packet_fields.extend(get_nested_fields(f))
        if flag:
            cols = get_packet_field_names(pkt)
            rows.append(align_lists(column_names, cols, packet_fields))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=column_names)


# ── Core Functions ─────────────────────────────────────────────────────────────

def run_tshark(pcap_path: str, xml_path: str):
    subprocess.run(
        ["tshark", "-r", pcap_path, "-T", "pdml"],
        stdout=open(xml_path, "w"),
        stderr=subprocess.DEVNULL,
        check=True
    )


def pad_sequence_list(x_list, max_len=None, pad_value=0.0):
    n_feat = x_list[0].shape[1]
    if max_len is None:
        max_len = max(len(x) for x in x_list)
    X = np.full((len(x_list), max_len, n_feat), pad_value, dtype=np.float32)
    for i, x in enumerate(x_list):
        L = min(len(x), max_len)
        X[i, :L, :] = x[:L, :]
    return X, max_len


def load_artifacts(artifacts_dir):
    model = keras.models.load_model(
        os.path.join(artifacts_dir, "sequence_model.keras"),
        compile=False
    )
    with open(os.path.join(artifacts_dir, "preprocess.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(artifacts_dir, "feature_encoding.pkl"), "rb") as f:
        enc = pickle.load(f)
    return model, meta, enc


def pcap_to_features(pcap_path, enc, meta):
    feature_cols  = meta["feature_cols"]
    scaler        = meta["scaler"]
    max_len       = meta["max_len"]
    category_maps = enc["category_maps"]

    # Step 1: pcap → XML
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        xml_path = tmp.name
    try:
        run_tshark(pcap_path, xml_path)
        df = parse_xml_to_df(xml_path)   # ← now fully inline, no import needed
    finally:
        os.remove(xml_path)

    if df.empty:
        raise ValueError("No NR-RRC packets found in pcap.")

    # Step 2: Encode using trained mapping
    for col in df.columns:
        if col in category_maps:
            df[col] = (
                df[col].astype("string")
                       .map(category_maps[col])
                       .fillna(-1)
                       .astype(np.int32)
            )

    # Step 3: Deduplicate columns, align to training schema
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    aligned = df.reindex(columns=feature_cols, fill_value=-1.0)

    # Step 4: Scale
    x = aligned.apply(pd.to_numeric, errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    x = scaler.transform(x).astype(np.float32)

    return x, max_len


def predict(pcap_path, artifacts_dir, threshold=0.5):
    print(f"Loading model from {artifacts_dir}...")
    model, meta, enc = load_artifacts(artifacts_dir)

    print(f"Processing: {os.path.basename(pcap_path)}")
    x, max_len = pcap_to_features(pcap_path, enc, meta)

    X, _ = pad_sequence_list([x], max_len=max_len, pad_value=0.0)
    prob = model.predict(X, verbose=0).ravel()[0]
    pred = int(prob >= threshold)

    print("\n" + "="*45)
    print(f"  PCAP:        {os.path.basename(pcap_path)}")
    print(f"  Probability: {prob:.4f}")
    print(f"  Threshold:   {threshold}")
    print(f"  Prediction:  {'⚠️  FBS ATTACK DETECTED' if pred == 1 else '✅  BENIGN'}")
    print("="*45 + "\n")

    return pred, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FBS Detector — Live Inference")
    parser.add_argument("--pcap",          required=True)
    parser.add_argument("--artifacts-dir", default="artifacts_2/")
    parser.add_argument("--threshold",     default=0.5, type=float)
    args = parser.parse_args()  

    if not os.path.exists(args.pcap):
        print(f"Error: pcap not found: {args.pcap}")
        sys.exit(1)

    pred, prob = predict(args.pcap, args.artifacts_dir, args.threshold)
    sys.exit(0 if pred == 0 else 1)
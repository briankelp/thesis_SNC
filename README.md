# UE-Centric Fake Base Station Detection

This repository contains the implementation artifacts for a thesis project on
detecting Fake Base Station (FBS) downgrade attacks from the User Equipment
(UE) side. The detector analyzes 5G NR-RRC and NAS-5GS control-plane packet
sequences and classifies a UE registration session as either benign or
malicious using an LSTM-based sequence model.

The project follows a simulation-to-hardware workflow:

1. Generate labeled 5G signaling traces with srsRAN ZMQ and Open5GS.
2. Extract NR-RRC and NAS-5GS fields from PCAP captures using `tshark`.
3. Encode packet-level features into fixed trace-level sequences.
4. Train an LSTM sequence classifier.
5. Validate the trained model on SDR hardware captures from a UE node.

## Project Context

Fake Base Stations can exploit pre-authentication signaling behavior to push a
UE away from 5G and toward weaker radio access technologies. This project
focuses on detecting downgrade-style attack behavior by observing Layer 3
control-plane messages available at the UE, especially suspicious 5GMM
Registration Reject and Authentication Reject patterns.

The primary attack labels in this project include 5GMM reject causes associated
with downgrade behavior:

- `7`: 5GS services not allowed
- `11`: PLMN not allowed
- `15`: No suitable cells in tracking area
- `27`: N1 mode not allowed
- Authentication reject sessions caused by key mismatch behavior

Other reject causes are included as benign/non-attack examples to reduce false
positives when the UE sees ordinary network rejection, congestion, roaming, or
policy events.

## Repository Layout

```text
.
|-- analyze_results.py              # Confusion matrix and metric reporting
|-- dataset_generation.ipynb        # Dataset generation and feature extraction work
|-- live_detection.py               # Continuous inference on a growing UE PCAP
|-- live_detection_with_stat.py     # Hardware-test inference with CSV logging
|-- lstm_2.ipynb                    # Model training and evaluation notebook
|-- pcap_to_csv_inference           # PCAP/XML to model-aligned CSV conversion
|-- artifacts/                      # Model and preprocessing artifacts
|-- artifacts_2/                    # Current default model artifacts
|-- csv_files/                      # Inference CSV output
|-- csv_files_srsRAN_zmq/           # Raw and encoded training datasets
|-- pcap_srsRAN_zmq/
|   |-- attack/                     # Attack PCAP traces
|   `-- benign/                     # Benign and non-attack rejection traces
|-- hardware_results.csv            # Hardware validation run log
|-- hardware_results_filtered.csv   # Filtered hardware validation run log
`-- requirements.txt
```

The live detection scripts default to `artifacts_2/`, which contains:

- `sequence_model.keras`: trained Keras sequence classifier
- `preprocess.pkl`: scaler, feature columns, and sequence length metadata
- `feature_encoding.pkl`: categorical feature encodings used at inference time
- `session_map.csv`: session metadata

## Dataset Summary

The included srsRAN ZMQ dataset contains:

- `759` PCAP sessions total
- `231` attack sessions
- `528` benign/non-attack sessions
- `6,741` extracted packet rows
- `2,752` raw feature columns before reduction
- `892` encoded feature columns after preprocessing

Training data is stored in:

- `csv_files_srsRAN_zmq/train_raw.csv`
- `csv_files_srsRAN_zmq/train_encoded.csv`

## Requirements

Python dependencies are pinned in `requirements.txt`. The packet extraction
pipeline also requires `tshark`, which is usually installed through Wireshark.

For live hardware validation, the surrounding testbed uses:

- srsRAN Project for the gNB/FBS side
- srsRAN 4G for the UE side
- Open5GS for the 5G core
- USRP-B200 SDR hardware for over-the-air validation

Only run radio experiments in a shielded, authorized, and properly configured
test environment.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `tshark` separately if it is not already available:

```bash
sudo apt install tshark
```

Check that the model artifacts are present:

```bash
ls artifacts_2
```

You should see `sequence_model.keras`, `preprocess.pkl`, and
`feature_encoding.pkl`.

## Convert a PCAP for Inference

Use `pcap_to_csv_inference` to convert a PCAP or PDML XML file into a CSV with
the trained feature schema.

```bash
python pcap_to_csv_inference \
  --pcap pcap_srsRAN_zmq/attack/reject_cause_27_1.pcap \
  --output-csv csv_files/infer_input.csv \
  --artifacts-dir artifacts_2 \
  --encoding-mode trained
```

For XML input instead of PCAP:

```bash
python pcap_to_csv_inference \
  --xml path/to/capture.xml \
  --output-csv csv_files/infer_input.csv \
  --artifacts-dir artifacts_2 \
  --encoding-mode trained
```

## Run Live Detection

`live_detection.py` continuously watches a PCAP file, converts the current
capture to PDML using `tshark`, extracts NR-RRC/NAS-5GS features, applies the
saved preprocessing pipeline, and prints the model prediction.

```bash
python live_detection.py \
  --artifacts-dir artifacts_2 \
  --pcap /tmp/ue_mac_nr.pcap \
  --interval 5 \
  --threshold 0.5
```

The default watched file is `/tmp/ue_mac_nr.pcap`, matching the UE-side capture
path used by the srsRAN test setup.

## Run Hardware Test Logging

Use `live_detection_with_stat.py` when running repeated hardware scenarios and
logging predictions to a CSV.

Attack scenario example:

```bash
python live_detection_with_stat.py \
  --artifacts-dir artifacts_2 \
  --pcap /tmp/ue_mac_nr.pcap \
  --scenario reject_27 \
  --ground-truth 1 \
  --log-csv hardware_results.csv
```

Benign scenario example:

```bash
python live_detection_with_stat.py \
  --artifacts-dir artifacts_2 \
  --pcap /tmp/ue_mac_nr.pcap \
  --scenario benign \
  --ground-truth 0 \
  --log-csv hardware_results.csv
```

## Analyze Hardware Results

After collecting hardware runs, compute the confusion matrix and per-scenario
metrics:

```bash
python analyze_results.py --csv hardware_results_filtered.csv
```

Current filtered hardware validation summary:

- `574` evaluated sessions
- Accuracy: `99.13%`
- Precision: `97.20%`
- Recall: `98.11%`
- False positive rate: `0.64%`
- F1-score: `97.65%`
- Average prediction latency: `82.4 ms`

Overall confusion matrix:

```text
                  Predicted Benign   Predicted Attack
Actual Benign            465                  3
Actual Attack              2                104
```

## Reproducing the Training Workflow

The training and dataset development workflow is primarily notebook-based:

- Use `dataset_generation.ipynb` to inspect or regenerate extracted training
  data from PCAP traces.
- Use `lstm_2.ipynb` to train and evaluate the LSTM sequence model.
- Save compatible model artifacts into `artifacts_2/` for live inference.

The live inference scripts expect all preprocessing choices from training to be
serialized with the model. If the feature schema, sequence length, or encoding
logic changes, regenerate `sequence_model.keras`, `preprocess.pkl`, and
`feature_encoding.pkl` together.

## Notes

- `artifacts_2/` is the default artifact directory used by the current live
  scripts.
- `artifacts/` is kept for earlier/local artifact variants.
- `hardware_results_filtered.csv` is the preferred results file for reporting
  final hardware validation metrics.
- The repository contains PCAP captures and trained model files, so expect it to
  be larger than a code-only project.

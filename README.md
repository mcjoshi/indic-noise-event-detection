# indic-noise-event-detection

Transcript tags as supervision: routed and router-free noise event detection in Indic speech.
Our system for IndoML Datathon 2026, Track 1 (noise event detection).

Team Hydra: Manish Joshi, Neeraj Singh Aithani.

Class-agnostic sound event detection on Indic speech from the Vaani corpus: for every clip, predict the onset and offset of each non-speech noise event.
This repository holds the complete training and inference code for our two systems, their configurations, and scripts to fetch the trained checkpoints.

| System | What it is | Leaderboard (Combined = Event F1 + Segment Dice) |
|---|---|---|
| **Router-free** (`generic_224`) | One pipeline for every clip: five students trained on blended natural/synthetic data, two decodes each, one weighted vote | 1.6114 |
| Router-free + long-span rule (`generic`) | The same, with clusters of at least 6 s kept at a lower weight | submitted as 285, no board score available |
| **Routed** (`separate`, leaderboard entry) | A peak-level test sends the organisers' synthetic mixtures to four specialists and natural recordings to a six-model vote | **1.6168** (F1 0.7903, Dice 0.8265), 2nd |

The technical report is `docs/technical_report.pdf`.

## Repository layout

```
README.md
LICENSE                        MIT (src/beats/ keeps Microsoft's MIT licence)
requirements_train.txt         training environment
requirements_inference.txt     inference environment
configs/
  system_generic_224.json      router-free system as scored on the board (1.6114)
  system_generic.json          router-free system with the long-span rule (submission 285)
  system_separate.json         routed system (leaderboard entry): router, natural and synthetic members
  members/                     exact training arguments of every model in the two systems
  lineage/                     exact training arguments of their ancestors (warm starts)
  donors/                      two siblings whose decode configs and refiners sML373 and sML374c reuse
  data/                        Vaani shard lists for the untranscribed audio
  decode/                      tuned decode configurations used at inference
src/                           all model, data, training, labelling and inference code
inference/infer.py             end-to-end inference: audio folder -> submission zip
train/train_member.sh          train one model from its configuration, then fit its decode configs and refiner
scripts/run_inference.sh       one-line inference entry point
scripts/download_assets.py     fetch checkpoints and pretrained encoders
checkpoints/                   downloaded model checkpoints and boundary refiners
pretrained/                    downloaded pretrained encoders
docs/technical_report.pdf
```

## Environment and hardware

- Python 3.12.4.
- PyTorch 2.9.0 with its bundled CUDA 12.8 runtime (`torch 2.9.0+cu128`); NVIDIA driver 535.288.01.
- Training and inference: NVIDIA A100 80 GB PCIe (one GPU per training run; the machine has four).
- Host: AMD EPYC 7V13 (96 cores), 866 GB RAM, Ubuntu 22.04.4 LTS.
- System dependency: `libsndfile` (used by `soundfile`).
- Pretrained model dependencies are listed under "Pretrained models" below with their sources.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cu128   # or requirements_train.txt
python3 scripts/download_assets.py               # checkpoints/ and pretrained/ (about 16 GB)
```

Weights and manifests are released under CC BY-NC 4.0 (the models were trained partly on ESC-50; see the model card). They are hosted at [huggingface.co/manishjoshiai/indic-noise-event-detection](https://huggingface.co/manishjoshiai/indic-noise-event-detection).
The script also downloads `openai/whisper-medium` and `microsoft/wavlm-base-plus` from the Hugging Face Hub.
Add `--train` to also fetch HuBERT-large and the training manifests (see `train/README.md`).
`VAANI_PRETRAINED` redirects the pretrained encoders for download and inference. Training expects `data/`, `work/` and `pretrained/` in the repository root; symlink them to put the data elsewhere.

## Inference

```bash
bash scripts/run_inference.sh /path/to/audio generic_224 0   # router-free system (board 1.6114) on GPU 0
bash scripts/run_inference.sh /path/to/audio generic 0       # router-free + long-span rule
bash scripts/run_inference.sh /path/to/audio separate 0   # routed system on GPU 0
```

Input: a folder of 16 kHz mono `.wav` files, one per clip, named `<clip_id>.wav` (the Phase 2 evaluation layout).
Output: `outputs/<system>/submission_<system>.zip`, holding `predictions.jsonl` at the zip root with one line per clip:

```json
{"clip_id": "<clip_id>", "events": [{"onset": 1.24, "offset": 3.81}]}
```

Every clip appears once; a clip with no detections gets `"events": []`.
Each member's decode is cached under `outputs/<system>/<member>_<route>/`, so a rerun only redoes the fusion.
On one A100 the 5,517 Phase 2 test clips take 21 minutes for the router-free system (10 decodes) and 32 minutes for the routed system (16 decodes plus the residual channel).

Reproducibility, checked from raw audio with the released checkpoints:
- The router-free system reproduces both of its submitted files (224 and 285) on all 5,517 clips.
- The routed system reproduces its submitted file on 5,516 clips. On the remaining clip every voter abstains and the system outputs no event; in the submitted file that clip carried a span patched in from an earlier submission.

Batch composition changes the padded forward pass at the millisecond level, so the system files pin each member's batch size to the one used for the submissions.

## The two systems

**Router-free** (`configs/system_generic_224.json`; `configs/system_generic.json` adds the long-span rule).
- Five students (Whisper-medium x3, BEATs with the localisation pretext, WavLM), each fine-tuned on an equal blend of natural and synthetic clips, run on every clip; nothing in the pipeline knows which domain a clip comes from.
- Each model is decoded twice (threshold decode with the boundary refiner, and the event-proposal decode). The ten decodes are fused by one weighted vote: Whisper and WavLM decodes weigh 0.5, BEATs decodes 1.0; clusters at IoU 0.6 (0.45 for spans of at least 6 s), median edges, kept at summed weight 2.5, or 1.5 for spans of at least 6 s.

**Routed** (`configs/system_separate.json`).
- Router: a clip whose largest absolute sample is below 0.995 is treated as a synthetic mixture, otherwise as a natural recording.
- Natural clips: six models, each decoded twice (a threshold decode refined by a learned boundary refiner, and a span-regression decode), fused by an IoU-0.6 vote over the 12 decodes with median edges. A cluster is kept with 4 votes, or with 3 votes when its span is at least 4 s; long spans are clustered at IoU 0.45. The largest cluster is kept when none qualifies.
- Synthetic clips: four synthetic-mixture specialists fused by an IoU-0.6 vote (3 of 4, mean edges), then a silence gate drops spans more than 30 dB below the clip's 95th-percentile frame level.

## Data

- Competition training data: `ARTPARK-IISc/Vaani-Noise-Event-Dataset`.
- The organisers' validation release (natural and synthetic clips), used for training the final members and for tuning.
- Untranscribed Vaani audio (`ARTPARK-IISc/Vaani`, shard lists in `configs/data/`) for pseudo-labelling and the pretext stage, with every competition clip removed.
- Transcribed Vaani audio (`ARTPARK-IISc/Vaani-transcription-part`), pseudo-labelled by our teachers and filtered by the event tags in each transcript, excluding every recording behind a validation or test clip.
- ESC-50 (`ashraq/esc50`) events for synthetic mixtures and the pretext stage.
- No test labels, test pseudo-labels or hand annotation were used.

## Training

Full stage-by-stage commands are in `train/README.md`.
All training runs replay a recorded configuration: `configs/members/<run>.json` for the models in the two systems and `configs/lineage/<run>.json` for the models they were warm-started from.

```bash
bash train/train_member.sh sML373 0     # train one model on GPU 0, then fit its decode configs and refiner
```

Train ancestors before descendants; each configuration's `init_from` names its parent.
The lineage of every final model:

| Final model | Encoder | Chain (parent -> child) |
|---|---|---|
| sML373 | Whisper-medium | v363_whisper_tp_e6 -> v373_whisper_unf_tp -> sML373 |
| sML374c | Whisper-medium | v363_whisper_tp_e6 -> v374_whisper_unf_tp2 -> sML374c |
| v365_whisper_unf_full | Whisper-medium | v333_whisper_base_pl2_e6 -> v365_whisper_unf_full |
| v137_n3_longunf | BEATs | v11 -> v22_distil -> v30_parent_warm -> v101_n3_ld -> v137_n3_longunf |
| v161_v145_dice3_s2 | BEATs | v11 -> v22_distil -> v30_parent_warm -> v101_n3_ld -> v145_n3_longunf_off -> v161_v145_dice3_s2 |
| v390_wavlmbp_unf_tp | WavLM-base-plus | v251_hubert_long -> v390_wavlmbp_unf_tp |
| v95b_selB | BEATs | v11 -> v22_distil -> v26_residual -> v87_synv3_big -> v95b_selB |
| v112s_dasheng_all / v113s_dasheng_all | Dasheng | v11 -> v90_synv3_giant_v11 -> v112s / v113s |
| s402_whisper_syn | Whisper-medium | v363_whisper_tp_e6 -> v373_whisper_unf_tp -> u373_mix -> s402_whisper_syn |
| sU373, stx374, stx365, st137, st390 (router-free) | Whisper / BEATs / WavLM | one student epoch from v373, v374, v365, v137, v390 |

Data stages (scripts in `src/`):
1. `prep_data.py` extracts the competition release; `make_split.py` builds a speaker-disjoint validation split.
2. `ingest_vaani_tp.py` adds transcribed Vaani recordings from `ARTPARK-IISc/Vaani-transcription-part` (gated; needs `HF_TOKEN`), excluding every competition clip and every recording behind a test or validation clip.
3. `build_synth_v2.py` builds synthetic training mixtures from competition speech and the organisers' noise events.
4. `tp_labels.py`, `pl_fuse.py`, `pl_transcript.py`, `transcript_labels.py` and `build_student_mix.py` produce teacher pseudo-labels for unlabelled audio, kept only where they agree with the event tags in the clip's transcript.
5. `pretrain_loc.py` and `pretrain_loc_dasheng.py` run the localisation pretext stage for the BEATs and Dasheng encoders.
6. `make_residuals.py` precomputes the enhancement residual channel used by one synthetic specialist.

The training manifests referenced by the configurations (`data/splits/*.jsonl`, about 2.7 GB) are released with the checkpoints.

## Pretrained models

| Name | Version / source | Role |
|---|---|---|
| Whisper-medium encoder | `openai/whisper-medium` (Hugging Face) | encoder of the Whisper members |
| WavLM-base-plus | `microsoft/wavlm-base-plus` (Hugging Face) | encoder of the WavLM members |
| HuBERT-large | `facebook/hubert-large-ll60k` (Hugging Face) | encoder of an ancestor of the WavLM member |
| BEATs iter3+ (AS2M) | `BEATs_iter3_plus_AS2M.pt`, microsoft/unilm (BEATs) | encoder of the BEATs members and starting point of the BEATs pretext stage |
| Dasheng base | `dasheng` 0.0.9 (pip), Xiaomi | encoder of the Dasheng specialists and starting point of their pretext stage |
| BEATs / Dasheng localisation pretext | ours (`pretrained/beats_loc_10000.pt`, `pretrained/dasheng_loc_10000.pt`) | encoder initialisation |
| Enhancement U-Net | ours (`pretrained/unet_v4.pt`) | residual channel of one synthetic specialist |

## Disclosures

- The organisers' validation clips are training data for several models: the 1,806 natural clips for v161, sML373, sML374c and sU373 (the last three also train on our validation split), and all 695 synthetic clips for every synthetic specialist and every router-free student. Scores of these models on those clips are in-sample.
- Five natural and one synthetic organiser-validation clip IDs also appear in the competition training release.
- Late changes to both systems were selected on the Phase 2 board, so board deltas between submissions are optimistic.
- The untranscribed Vaani pool was filtered by clip ID only: 108 of its 207,268 clips come from recordings that also contain a test clip. They carry no labels.
- One clip of the routed submission carried a span patched in from an earlier submission; the public code leaves that clip empty (every voter abstains).
- The data generators were refined during the competition, so regenerated synthetic audio follows the released recipe without being bit-identical (see `train/README.md`).
- External data is listed under Data above; no test labels, test pseudo-labels or manual annotation were used.

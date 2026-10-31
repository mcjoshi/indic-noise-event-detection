# Training

The released manifests are the exact training inputs: every label, pseudo-label and synthetic mixture each model saw.
Fetch them with `python3 scripts/download_assets.py --train`; they land in `data/splits/` with paths pointing into `data/`.
Retraining needs the audio those manifests reference; steps 1 and 2 rebuild it, step 3 trains the models.
All commands run from the repository root.

## 1. Audio

Competition data (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`, 182 parquet shards; gated, so accept its terms on Hugging Face and run `hf auth login` first):

```bash
hf download ARTPARK-IISc/Vaani-Noise-Event-Dataset --repo-type dataset --local-dir data/hf_train
python3 src/prep_data.py --shards data/hf_train/data --out data/prepared --workers 24
python3 src/make_split.py --manifest data/prepared/manifest.jsonl --out data/splits --val-hours 4.0
```

Organisers' validation set: download "Validation Dataset" from the Codabench competition page (Files), unzip it to `data/official_val/validation`, then:

```bash
python3 src/build_official_val.py      # data/official_val/prepared
python3 src/build_official_bank.py     # data/official_noise_bank: noise events recovered from the synthetic validation clips
```

Untranscribed Vaani audio for pseudo-labelling and the pretext stage (gated, needs `HF_TOKEN`):

```bash
python3 src/download_vaani_parent.py --shards configs/data/vaani_parent_shards_round1.txt --flat data/parent_raw_flat
python3 src/prep_data.py --shards data/parent_raw_flat --out data/parent_prepared --workers 24
python3 src/download_vaani_parent.py --shards configs/data/vaani_parent_shards_round2.txt --flat data/parent_raw_flat2
python3 src/prep_data.py --shards data/parent_raw_flat2 --out data/parent_prepared2 --workers 24
python3 src/filter_parent.py --manifest data/parent_prepared/manifest.jsonl --out data/splits/parent_pool.jsonl
python3 src/filter_parent.py --manifest data/parent_prepared2/manifest.jsonl --out data/splits/parent_pool2.jsonl
```

`filter_parent.py` removes every competition train, validation and test clip by clip ID; it reads the test clip IDs from `--exclude-audio-dir` (default `data/eval/extracted/audio`, the unzipped Phase 2 test audio) and stops if that folder is missing.

Transcribed Vaani audio (`ARTPARK-IISc/Vaani-transcription-part`, gated); the script skips every competition clip and every recording behind a validation or test clip:

```bash
HF_HUB_DISABLE_XET=1 python3 src/ingest_vaani_tp.py --max-gb 180    # data/vaani_tp
```

ESC-50 (`ashraq/esc50`, CC BY-NC) for the synthetic mixtures and the pretext stage:

```bash
hf download ashraq/esc50 --repo-type dataset --local-dir data/esc50
python3 src/n3_bank.py                 # data/n3_esc_bank.npz
```

## 2. Synthetic mixtures and residuals

Mixtures of competition speech beds with ESC-50 events or the organisers' noise bank.
`SYN` stands for the shared flags:

```bash
SYN="--split train --beds data/splits/train_v29.jsonl --audio-root data/prepared --attempt-multiplier 2 --snr-lo 6 --snr-hi 12 \
  --level-ref abs --peak-norm 0.99 --no-clean-bed --bed-gate-db 6 --bed-gate-atten 40 --min-gap 0.15"
python3 src/build_synth_v2.py $SYN --out data/synth_v3/train  --n 12000 --workers 14 --max-uses 24  --clip-prefix synthv3  --seed 3100
python3 src/build_synth_v2.py $SYN --out data/synth_v3/train2 --n 24000 --workers 12 --max-uses 40  --clip-prefix synthv3b --seed 3200
python3 src/build_synth_v2.py $SYN --out data/synth_v3/train3 --n 48000 --workers 12 --max-uses 80  --clip-prefix synthv3c --seed 3300
python3 src/build_synth_v2.py $SYN --out data/synth_v3/train4 --n 84000 --workers 12 --max-uses 150 --clip-prefix synthv3d --seed 3400
python3 src/build_synth_v2.py --split val --beds data/splits/val.jsonl --audio-root data/prepared --out data/synth_v3/val --n 1000 --workers 6 \
  --max-uses 8 --attempt-multiplier 3 --snr-lo 6 --snr-hi 12 --level-ref abs --peak-norm 0.99 --no-clean-bed --bed-gate-db 6 \
  --bed-gate-atten 40 --min-gap 0.15 --clip-prefix synthv3 --seed 3001
python3 src/build_synth_v2.py --split train --beds data/splits/train_v29.jsonl --audio-root data/prepared --attempt-multiplier 2 \
  --snr-lo -6 --snr-hi 13 --level-ref abs --peak-norm 0.99 --no-clean-bed --bed-gate-db 6 --bed-gate-atten 40 --min-gap 0.2 \
  --count-probs 0.25,0.35,0.25,0.10,0.05 --official-bank data/official_noise_bank/bank.json \
  --out data/synth_v4/train --n 48000 --workers 12 --max-uses 400 --clip-prefix synthv4 --seed 4100
python3 src/build_synth_v2.py --split train --beds data/splits/train_v29.jsonl --audio-root data/prepared --out data/synth_v2/train \
  --n 6000 --workers 16
```

The earliest mixture sets (`data/synthmix_hard`, `data/synthmix_val`) come from `src/build_synth_mixtures.py` with ESC-50 events (22,000 attempts over the training clips, overlapping events allowed); their exact flags were not recorded.

The residual specialist (`v95b_selB` and its ancestors) takes a second input channel, the mixture minus the speech estimate of our enhancement U-Net:

```bash
for m in data/splits/train_v24.jsonl data/splits/val.jsonl data/splits/train_synth_v3_big.jsonl data/synth_v3/val/manifest_abs.jsonl \
         data/splits/train_synth_v4_offall.jsonl data/splits/official_syn_foldB.jsonl; do
  python3 src/make_residuals.py --manifest $m --audio-root data/prepared --out data/residual
done
```

The audio generators were refined during the competition, so regenerated mixtures follow the same recipe but are not bit-identical to ours.
The synthetic rows of the released manifests therefore describe our mixtures, not regenerated ones: when you regenerate the audio, rebuild the synthetic manifests from the generators' own outputs (each run writes a `manifest.jsonl` next to its audio) and keep the released natural rows as they are.

## Pseudo-labels (already contained in the released manifests)

The released manifests carry the exact pseudo-labels every model trained on, so retraining does not need this stage.
It is documented for completeness; the first-round teacher includes `v273_hubert_unf`, which is not released.

1. Teacher decodes: run `src/predict.py` for each teacher and both decodes (threshold with `--refiner`, and the proposal config) over the clip pools, with `--clip-list` naming the clips.
2. `src/pseudo_label_fused.py` (first round): Silver edges moved onto fused spans, Bronze and unlabelled clips given fused spans (writes `train_v22`, `train_v24`, `train_v29` with the flags in `configs/lineage/`).
3. `src/pl_fuse.py --votes 3 --add-votes 5` writes `train_v29_pl2.jsonl` from the round-1 teacher (six decodes).
4. `src/pl_transcript.py --w-pos 3` relabels Bronze with the transcript tags (`train_v29_pl4.jsonl`).
5. `src/tp_labels.py` labels the transcribed Vaani clips the same way (`train_v29_pl4_tp.jsonl`; `--round 2` for the remaining clips).
6. `src/build_student_mix.py` builds the blended student sets from the five-model teacher (`--ext 160000 --natural 120000` gives the 240k blend).

`src/transcript_labels.py` holds the tag parsing and the Hungarian assignment used by steps 4-6, and `src/label_control.py` scores any label set against Gold validation.

## 3. Pretext stage and models

Localisation pretext stage for the BEATs and Dasheng encoders (inserted ESC-50 events in untranscribed Vaani audio); both outputs are also released in `pretrained/`:

```bash
python3 src/pretrain_loc.py --manifest-root data/splits/parent_pool.jsonl data/parent_prepared \
  --manifest-root data/splits/parent_pool2_abs.jsonl / --steps 10000 --workers 10 --out work/n3_loc10k
python3 src/pretrain_loc_dasheng.py --manifest-root data/splits/parent_pool.jsonl data/parent_prepared \
  --manifest-root data/splits/parent_pool2_abs.jsonl / --steps 10000 --workers 10 --out work/n3dasheng_loc10k
```

Every model replays its recorded configuration (`configs/members/` or `configs/lineage/`), parents first:

```bash
bash train/train_member.sh <RUN> <GPU>
```

To fine-tune one model without retraining its ancestors, start from the released parent checkpoint instead, for example:

```bash
python3 src/train.py $(python3 src/replay_args.py configs/members/st390.json init_from=checkpoints/v390_wavlmbp_unf_tp.pt)
```

| Stage | Runs (each starts from a run of the previous stage, named by `init_from`) |
|---|---|
| 1 | `v11`, `v251_hubert_long`, `v333_whisper_base_pl2_e6`, `v363_whisper_tp_e6` |
| 2 | `v22_distil`, `v90_synv3_giant_v11`, `v365_whisper_unf_full`, `v373_whisper_unf_tp`, `v374_whisper_unf_tp2`, `v390_wavlmbp_unf_tp` |
| 3 | `v26_residual`, `v30_parent_warm`, `v112s_dasheng_all`, `v113s_dasheng_all`, `sML373`, `sML374c`, `sU373`, `u373_mix`, `stx365`, `stx374`, `st390`, `sty373`, `st374` |
| 4 | `v87_synv3_big`, `v101_n3_ld`, `s402_whisper_syn` |
| 5 | `v95b_selB`, `v137_n3_longunf`, `v145_n3_longunf_off` |
| 6 | `st137`, `v161_v145_dice3_s2` |

After training, `train_member.sh` fits the decode configurations of a natural-route model on `val_mixed`: threshold decode, boundary refiner, threshold route, span-regression route.
Synthetic specialists are tuned on the organisers' synthetic validation clips instead:

```bash
python3 src/tune_postproc.py --ckpt work/crnn_<RUN>/best.pt --val data/splits/official_val_syn.jsonl --audio-root data/prepared \
  --out work/postproc_<RUN>_offsyn.json --cache work/val_offsyn_probs_<RUN>.pkl --rounds 3
```

Three Whisper students reuse the decode configurations and refiner of a sibling (listed as `decode_assets_from` in `configs/system_*.json`): sML373 uses sty373's, sML374c uses st374's and sU373 uses u373_mix's.
The two siblings that are not in the lineage, sty373 and st374, have their configurations in `configs/donors/` and train with the same `train_member.sh`.
The configurations used for the submissions are in `configs/decode/`.

# CSIE7694 VFX Final Project — WORKLOG

**Project:** VideoCoF: Unified Video Editing with Temporal Reasoner  
**Track:** Research  
**Paper:** CVPR 2026 Highlight · arXiv 2512.07469  
**Repo (upstream):** https://github.com/knightyxp/VideoCoF  
**Our repo:** https://github.com/goog-msft-fb-nflx-nvda-aapl/csie7694-vfx-final  
**GPU server:** gsm-gpu — 8× H200 (143GB each), CUDA 12.4  
**Project dir on server:** `/home/jtan/vfx/`

---

## Research Extension Plan

Baseline: run VideoCoF inference (object removal / addition / swap / local style transfer) on custom videos.

Extension idea: **Temporal Edit-Consistency Evaluator**
- Track edited regions with optical flow (RAFT or FlowFormer)
- Measure per-frame flicker (L2 diff in edited mask region) and identity drift (CLIP similarity)
- Compare short (33-frame) vs long (128/256/512-frame) edits
- Possibly: flow-guided region consistency regularization at inference time

Demo plan (5-min video):
1. Show 3 input videos + 4 task types (removal, addition, swap, style)
2. Show length extrapolation: 33 → 128 → 256 → 512 frames
3. Show evaluator metrics: flicker / drift curves
4. Side-by-side comparison vs baseline (w/o reasoning tokens)

---

## Environment

| Item | Value |
|------|-------|
| Conda env | `videocof` (miniforge3) |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA toolkit | 12.4 |
| tmux session | `vfx_setup` |

Models to download:
- `Wan-AI/Wan2.1-T2V-14B` → `/home/jtan/vfx/Wan2.1-T2V-14B`
- `XiangpengYang/VideoCoF` → `/home/jtan/vfx/videocof_weight`
- FusionX LoRA → `/home/jtan/vfx/videocof_weight/`

---

## Log

### 2026-06-20

**Environment setup started**
- Cloned VideoCoF to `/home/jtan/vfx/VideoCoF`
- Created conda env `videocof` (Python 3.10)
- Installing PyTorch 2.5.1+cu121 (tmux `vfx_setup`, window 0)
- Started Wan2.1-T2V-14B download (tmux `vfx_setup`, window `models`)
- GitHub repo created: https://github.com/goog-msft-fb-nflx-nvda-aapl/csie7694-vfx-final

**Fixes applied:**
- xformers 0.0.35 (built for PyTorch 2.10, leaked from ~/.local) shadowed by installing xformers 0.0.28.post3 in conda env
- cffi missing → `pip install cffi` fixed diffusers import chain

**Sanity inference: PASSED**
- Output: `results/obj_rem_1/gen_two_man.mp4` (33 frames, 400×752), side-by-side compare, reason+edit video
- 4-step diffusion: ~20s on 1× H200, ~0 GPU memory idle after
- Warnings only (FutureWarning autocast, padding mask) — no errors

**All 4 task types: PASSED** (parallel on GPUs 1-3)
- obj_rem: results/obj_rem_1/ (two_man.mp4, 33f)
- obj_add: results/obj_add_1/ (woman_ballon.mp4, 33f)
- obj_swap: results/obj_swap_1/ (sign.mp4, 33f)
- local_style: results/local_style_2/ (ketchup.mp4, 33f)

**Evaluator: eval_consistency.py — implemented and running**
Metrics: flicker (L2 temporal diff in edit region), flow_smoothness (Farneback optical flow), CLIP drift (cosine distance from frame-0)

**Eval results (33-frame baseline):**

| Tag | edit_coverage | flicker↓ | flow_smoothness↓ | clip_drift↓ |
|-----|-----------|---------|--------------|----------|
| obj_rem | 0.391 | 13.95 | 1.881 | 0.106 |
| obj_add | 0.283 | 16.50 | 0.744 | 0.025 |
| obj_swap | 0.545 | 5.19 | 0.458 | 0.013 |
| local_style | 0.190 | 8.85 | 0.386 | 0.019 |

Note: obj_rem has highest flicker/drift (person removal is hardest task); local_style lowest (only color change).

**Long-video extrapolation: IN PROGRESS on GPUs 5/6/7**
Using predict_v2v_dmd_cot_json.py + daiyu_529frames.mp4
- 128f: num_frames=260, source_frames=128 → results/longvid_real128/
- 256f: num_frames=516, source_frames=256 → results/longvid_real256/
- 512f: num_frames=1028, source_frames=512 → results/longvid_real512/

**Long-video extrapolation: 128f + 256f DONE, 512f running (GPU 7, 47GB, 100%)**
- All on daiyu_529frames.mp4 (local_style task)
- 128f → 126 frames actual output
- 256f → 254 frames actual output

**Key research finding — flicker DECREASES at longer lengths:**

| Tag | frames | flicker↓ | flow_smooth↓ | clip_drift↓ |
|-----|--------|----------|-------------|-------------|
| local_style_daiyu_33 | 33 | 65.7 | 9.18 | 0.154 |
| local_style_128 | 126 | 41.0 | 5.24 | 0.274 |
| local_style_256 | 254 | 27.5 | 4.01 | 0.282 |
| local_style_512 | 510 | **20.2** | **2.69** | 0.247 |

Interpretation: VideoCoF's temporal reasoner produces *smoother* edits at longer sequences.
CLIP drift increases slightly (style drifts more at long range) — interesting trade-off to discuss.

**Trend confirmed: monotonic improvement in temporal consistency with length.**
Flicker: 65.7 → 41.0 → 27.5 → 20.2 (33→128→256→512 frames)
Flow: 9.18 → 5.24 → 4.01 → 2.69

**TODO next:**
- [ ] Collect 3 custom videos for demo (download from Pexels/Pixabay)
- [ ] Run 4 task types on custom videos
- [ ] Record 5-min demo video
- [ ] Write report / slides for submission

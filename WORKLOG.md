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

**TODO next:**
- [ ] Finish requirements install (`pip install -r requirements.txt`)
- [ ] Finish model downloads (Wan2.1-14B + VideoCoF weights + FusionX LoRA)
- [ ] Run sanity inference on sample video (obj_rem.sh)
- [ ] Collect 3 custom videos for demo
- [ ] Implement flow-based flicker evaluator
- [ ] Run long-video extrapolation experiments (33→512 frames)
- [ ] Record 5-min demo video

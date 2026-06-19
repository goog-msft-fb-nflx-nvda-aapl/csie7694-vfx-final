# CSIE7694 VFX Final Project

**See [WORKLOG.md](WORKLOG.md) for full experiment log, environment setup, and TODO list.**

## Quick Reference

- **Project:** VideoCoF video editing (Research track, CVPR 2026 Highlight)
- **GitHub:** https://github.com/goog-msft-fb-nflx-nvda-aapl/csie7694-vfx-final
- **GPU server:** gsm-gpu — `/home/jtan/vfx/`
- **Conda env:** `videocof` (miniforge3, Python 3.10, PyTorch 2.5.1+cu121)
- **tmux session:** `vfx_setup` (windows: 0=setup, models=wan-download, weights=vcof-download)

## Key Paths on gsm-gpu

```
/home/jtan/vfx/
├── VideoCoF/          # cloned upstream repo
├── Wan2.1-T2V-14B/    # base model (downloading)
├── videocof_weight/   # VideoCoF LoRA + FusionX LoRA (downloading)
├── download_wan.log
├── download_vcof.log
└── setup_reqs.log
```

## Git Rules

- user.name = james, user.email = j.tan@bluewx.co.jp
- No Claude attribution in commits
- Push to personal account only (goog-msft-fb-nflx-nvda-aapl)

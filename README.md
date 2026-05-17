# chx_resule

Point cloud denoising experiments for the 6th CG AI Challenge, track 2.

## Repository layout

- `REPORT.md`: overall experiment report and leaderboard summary
- `experiments/`: per-experiment notes and results
- `models/`: denoiser architectures
- `losses/`: training losses such as DSM, DCD, EMD, AGT, and surface losses
- `inference/`: inference and TTA scripts
- `scripts/`: smoke tests and launch helpers
- `train_*.py`: training entrypoints for different experiment families

## Notes

- Large artifacts such as checkpoints, logs, and output folders are excluded from git.
- Dataset files are expected to be prepared locally on the training machine.

## Quick start

```bash
python train_p0.py
python train_p1.py
python train_p2_v2.py
```

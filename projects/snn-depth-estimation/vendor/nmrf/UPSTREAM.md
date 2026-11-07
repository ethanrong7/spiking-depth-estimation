# NMRF upstream snapshot

- Repository: https://github.com/aeolusguan/NMRF
- Revision: `a50bd8f407ae481c38f95cf141e1b5256e962cb7`
- Imported: 7 November 2026, from the GitHub commit archive.
- License: MIT; see LICENSE and retained file-level notices.
- Source files are unchanged. The upstream `assets/` directory (25 MB of presentation media) is omitted.
- This revision includes post-CVPR-2024 initialization/proposal-loss changes.
- Native 50-bin adaptation and pilot runner live in the repository-root
  `nmrf_runtime.py` and `run_nmrf.py`.
- The upstream launcher uses RGB stereo datasets; the native runner uses this
  repository's explicit Synthetic spike manifest instead.
- Upstream loss weighting names `proposal_disp`, while its criterion emits
  `loss_prop`. Our wrapper corrects this key so proposal supervision is included;
  the vendored implementation is untouched.

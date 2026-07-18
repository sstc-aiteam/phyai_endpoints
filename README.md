# phyai_endpoints

## Quick Start
### Install python packages
```
pip install -r requirements.txt
```

### Python packages for JetPack 6.2
- Python version MUST be 3.10
- `pip install -r requirements.txt` installs the correct JetPack 6.2 CUDA wheels
  for torch/torchvision/triton directly (pinned by URL, not just version) so it
  won't accidentally resolve to PyPI's generic CPU-only wheel of the same version.
- If you need a different torch/torchvision/triton version than what's pinned,
  search "pytorch package for jetpack 6.2" and install manually with
  `python -m pip install $PkgName --index-url https://pypi.jetson-ai-lab.io/jp6/cu126`
  (use `--index-url`, not `--extra-index-url` — the latter doesn't take priority
  over PyPI when both host the same version).

### SAM2 fp16 + torch.compile on Jetson
`SAM2ROISegmenter` (used by `WardObjectPipelineService`) speeds up SAM2 with
fp16 autocast + `torch.compile`, following
[sam2-speedup](https://github.com/sstc-aiteam/sam2-speedup). On Jetson this
needs a couple of env vars so Triton's JIT can find the system CUDA toolkit;
without them, `torch.compile` fails during warmup and the code falls back
to eager fp16 automatically (still correct, just without the compile speedup):
```
export LD_LIBRARY_PATH="$(python -c 'import nvidia; print(nvidia.__path__[0])')/cu12/lib:$LD_LIBRARY_PATH"
export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas
export TRITON_CUDACRT_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/include
```
(Adjust the `cuda-12.6` paths if your JetPack/CUDA version differs.)

### Run service
```
python run.py
```


## Reference
- [RTDE Examples](https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html) 
- [Use with Robotiq Gripper](https://sdurobotics.gitlab.io/ur_rtde/guides/guides.html#use-with-robotiq-gripper)
- [Jetson Orin + RealSense in 5 minutes](https://jetsonhacks.com/2025/03/20/jetson-orin-realsense-in-5-minutes/)
  - [jetson-orin-librealsense](https://github.com/jetsonhacks/jetson-orin-librealsense)


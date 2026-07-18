# phyai_endpoints

## Quick Start
### Install python packages
```
pip install -r requirements.txt
```

### Python packages for JetPack 6.2
- Google >> "pytorch package for jetpack 6.2"
- Python version MUST be 3.10
- Use `python -m pip install $PkgName` to ensure install package in venv
- Install PyTorch and Torchvision with CUDA support  
  - `python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Run service
```
python run.py
```


## Reference
- [RTDE Examples](https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html) 
- [Use with Robotiq Gripper](https://sdurobotics.gitlab.io/ur_rtde/guides/guides.html#use-with-robotiq-gripper)
- [Jetson Orin + RealSense in 5 minutes](https://jetsonhacks.com/2025/03/20/jetson-orin-realsense-in-5-minutes/)
  - [jetson-orin-librealsense](https://github.com/jetsonhacks/jetson-orin-librealsense)


from rtde_control import RTDEControlInterface
from robotiq_gripper_control import RobotiqGripper

# 1. Initialize the control interface
rtde_c = RTDEControlInterface("192.168.50.75")

# 2. Initialize the gripper helper
gripper = RobotiqGripper(rtde_c)

# 3. Activate the gripper (only needed once after power-on)
gripper.activate()

# 4. Open and Close
gripper.open()   # Fully opens

gripper.close()  # Fully closes

gripper.open()   # Fully opens

# Optional: Partial move (0-255 or mm depending on version)
# gripper.move(50)


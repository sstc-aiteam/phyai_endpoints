import json
import numpy as np

# Your JSON data
json_data = """
{
  "message": "Hand-eye calibration successful with 34 points.",
  "transform_matrix": [
    [0.997931457612946, -0.04435156588336169, -0.046537560208874736, 0.011855480835776292],
    [0.054524749307613016, 0.967422797594866, 0.24722496315434014, -0.09030567718160565],
    [0.03405668244915773, -0.2492510166426944, 0.967839900543029, 0.08310598448576016],
    [0, 0, 0, 1]
  ]
}
"""

# 1. Parse the JSON string
data = json.loads(json_data)

# 2. Extract the matrix and convert to a NumPy array
matrix = np.array(data["transform_matrix"])

# 3. Save as .npy file
filename = "hand_eye_transform.npy"
np.save(filename, matrix)

print(f"Successfully saved matrix to {filename}")


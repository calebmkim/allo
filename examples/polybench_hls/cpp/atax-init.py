import json
import random

# Dimensions
rows = 116
cols = 124

# Generate 2D array "A" with random integer values between 0 and 10
array_A = [[random.randint(0, 10) for _ in range(cols)] for _ in range(rows)]

# Generate 1D array "x" with random integer values between 0 and 10
array_x = [random.randint(0, 10) for _ in range(rows)]

# Generate 1D array "y" with random integer values between 0 and 10
array_y = [random.randint(0, 10) for _ in range(rows)]

# Create JSON object
output_json = {"A": array_A, "x": array_x, "y": array_y}

# Print JSON object
print(json.dumps(output_json, indent=4))

from ultralytics import YOLO

# This will automatically download the model weights on first run
model = YOLO("yolo26n-pose.pt")

# Run inference on a sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Print the shape of the keypoints data
print("Keypoints shape:", results[0].keypoints.data.shape)
print("Success! YOLO26-Pose is working.")

# Optional: display the image with keypoints drawn on it
results[0].show()
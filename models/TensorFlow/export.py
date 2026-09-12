import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from pathlib import Path

model = ResNet50(weights="imagenet")

output_dir = Path("1/model.savedmodel")

tf.saved_model.save(model, str(output_dir))

print(f"SavedModel exported to: {output_dir}")
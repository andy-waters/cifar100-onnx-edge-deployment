"""
onnx_yaml_converter.py

Converts ONNX model metadata to a YAML file for configuration and deployment.
"""
import onnx
import yaml

# Load ONNX model
model = onnx.load("output/resnet_best_model.onnx")

# Extract input and output node names
input_names = [inp.name for inp in model.graph.input]
output_names = [out.name for out in model.graph.output]

# Define YAML structure
yaml_data = {
    "network": {
        "network_name": "custom_model",
        "paths": {"network_path": "resnet_best_model.onnx"},
        "input_shape": [1, 3, 224, 224],  # Adjust based on your model
        "parser": {
            "start_node_shapes": [[1, 3, 224, 224]],
            "end_node_shapes": [[1, 100]],
            "nodes": input_names + output_names,
        },
    }
}

# Save to YAML file
with open("output/resnet_best_model.yaml", "w") as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

print("YAML file created: model.yaml")
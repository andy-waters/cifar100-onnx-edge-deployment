
"""
onnx_converter.py

Exports PyTorch models to ONNX format for interoperability and deployment.
"""
import torch

class ONNXConverter:
    """
    Converts a PyTorch model to ONNX format.
    """
    def __init__(self, model, input_size=(1, 3, 224, 224), output_path="output/resnet_best_model.onnx"):
        """
        Args:
            model (torch.nn.Module): PyTorch model to export.
            input_size (tuple): Shape of dummy input.
            output_path (str): Path to save ONNX file.
        """
        self.model = model
        self.input_size = input_size
        self.output_path = output_path

    def convert_to_onnx(self):
        """
        Converts the model to ONNX and saves to output_path.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Create a dummy input tensor with the specified size
        dummy_input = torch.randn(self.input_size)

        # Export the model to ONNX format
        torch.onnx.export(self.model, dummy_input, self.output_path, export_params=True,
                          opset_version=11, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model has been converted to ONNX and saved at {self.output_path}")
        

if __name__ == "__main__":
    # Example usage
    from resnet_model import ResNetModelBuilder
    from resnet_model import ResNetModelType

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ResNetModelBuilder.get_model(model_type=ResNetModelType.RESNET34, num_classes=100, pretrained=True)
    model.load_state_dict(torch.load('output/resnet_best_model.pth', map_location=device))

    # Initialize the ONNX converter
    onnx_converter = ONNXConverter(model)

    # Convert the model to ONNX format
    onnx_converter.convert_to_onnx()
        
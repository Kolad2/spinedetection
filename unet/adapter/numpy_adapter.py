import torch

class NumpyAdapter:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, image):
        tensor = torch.from_numpy(image).float()
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # CHW -> BCHW
        tensor = tensor / 255.0
        tensor = tensor.cuda()

        with torch.no_grad():
            result = self.model(tensor)

        result = torch.sigmoid(result)
        result = result.squeeze().cpu().numpy()
        return result
import onnxruntime
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)
    return img


def load_onnx_model(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    return session, input_names, output_names


def predict_dissimilarity(onnx_session, input_names, image_path1, image_path2):
    input1 = preprocess_image(image_path1)
    input2 = preprocess_image(image_path2)
    ort_inputs = {input_names[0]: input1.numpy(), input_names[1]: input2.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    output1, output2 = torch.Tensor(ort_outs[0]), torch.Tensor(ort_outs[1])
    euclidean_distance = F.pairwise_distance(output1, output2)
    mean_dissimilarity = torch.mean(euclidean_distance).item()
    return mean_dissimilarity


def run(onnx_path, t1, t2):
    sess, input, _ = load_onnx_model(onnx_path)
    dissimilarity = predict_dissimilarity(sess, input, t1, t2)
    return f"{dissimilarity:.2f}"


if __name__ == "__main__":
    pass

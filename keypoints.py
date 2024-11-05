import onnxruntime as ort
from utils import letterbox, pre_process

class Keypoint:
    def __init__(self, model_path, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        # Initialization of keypoint-specific attributes

    def inference(self, image):
        img, (dw, dh) = letterbox(image)
        data = pre_process(img)
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # Perform keypoint analysis here and return processed frame
        return image

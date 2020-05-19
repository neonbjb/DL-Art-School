import onnxruntime
import numpy as np
import time

session = onnxruntime.InferenceSession("../results/gen.onnx")
v = np.random.randn(1,3,1700,1500)
st = time.time()
prediction = session.run(None, {"lr_input": v.astype(np.float32)})
print("Took %f" % (time.time() - st))
print(prediction[0].shape)
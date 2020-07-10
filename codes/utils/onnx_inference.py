import onnx
import numpy as np
import time

model = onnx.load('../results/gen.onnx')

outputs = {}
for n in model.graph.node:
    for o in n.output:
        outputs[o] = n

res = 0
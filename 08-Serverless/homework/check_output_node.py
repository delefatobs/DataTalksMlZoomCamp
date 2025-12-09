import onnx

model = onnx.load("hair_classifier_v1.onnx")
for o in model.graph.output:
    print(o.name)

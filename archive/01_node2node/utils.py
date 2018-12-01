import json
from keras.models import model_from_json

import numpy as np

def saveModel(model, filePath):
    weightsJson = [wtTensor.tolist() for wtTensor in model.get_weights()]
    modelJson = model.to_json()
    with open(filePath, "w") as fp:
        json.dump({"model":modelJson, "weights":weightsJson}, fp)

def loadModel(filePath, envDict=None):
    if envDict is None:
        envDict = globals()

    with open(filePath, "r") as fp:
        data = json.load(fp)
    modelJson = data["model"]
    weightsJson = data["weights"]

    model = model_from_json(modelJson, envDict)
    weights = [np.array(jsonWeights) for jsonWeights in weightsJson]

    model.set_weights(weights)
    return model


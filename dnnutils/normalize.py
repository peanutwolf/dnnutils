import numpy as np

def reweight_predictions(predictions):
    class_0_est_instances = predictions[:,0].sum()
    others_est_instances = predictions[:,1:].sum()
    new_p = predictions * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(predictions.shape[1])]])
    predictions = (new_p / np.sum(new_p,axis=1,keepdims=1))
    predictions = np.concatenate((predictions[:,:1],np.sum(predictions[:,1:],1,keepdims=True)), 1)
    return predictions
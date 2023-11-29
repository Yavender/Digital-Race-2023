import numpy as np
import cv2
import onnxruntime as rt

sess = rt.InferenceSession("deeplab.onnx", providers=['CPUExecutionProvider'])

def segment(image, sess):
    image = cv2.resize(image, (256,256), interpolation= cv2.INTER_LINEAR)
    image = image.transpose(2,0,1)
    image = image.astype('float32')
    image = np.expand_dims(image, axis = 0)
    ortvalue = rt.OrtValue.ortvalue_from_numpy(image)
    ortvalue.device_name()  # 'cpu'
    ortvalue.shape()        # shape of the numpy array X
    ortvalue.data_type()    # 'tensor(float)'
    ortvalue.is_tensor()    # 'True'
    np.array_equal(ortvalue.numpy(), image)  # 'True
    results = sess.run(["outputs"], {"inputs": ortvalue})
    print(results)
    return results

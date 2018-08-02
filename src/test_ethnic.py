from face_network import create_face_network
import cv2
import argparse
import numpy as np
from keras.optimizers import Adam, SGD

ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}

def predict_ethnic(image_path):

	means = np.load('means_ethnic.npy')

	model = create_face_network(nb_class=4, hidden_dim=512, shape=(224, 224, 3))

	model.load_weights('weights_ethnic.hdf5')

	im = cv2.imread(image_path, cv2.IMREAD_COLOR)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
	im = cv2.resize(im, (224, 224))
	im = np.float64(im)
	im /= 255.0
	im = im - means
	#print("NP: ", np)

	#print("np.array([im]): ", np.array([im]))
	#print("RESULT: ", model.predict(np.array([im])))
	result =  model.predict(np.array([im]))
	race = ETHNIC[np.argmax(result)]
	return race



'''
if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=False,
		help="path to test image")
	args = vars(ap.parse_args())

	image = 'joe-biden.jpg'


	args["image"] = image
	#print(args["image"])

	result = predict_ethnic(args["image"])
	print("RESULT: ", result)
	print("[np.argmax(result)] : ", [np.argmax(result)])
	print("argmax(result) : ", np.argmax(result))
	print(ETHNIC[np.argmax(result)])
'''


"""

if __name__ == "__main__":
	image_path = './joe-biden.jpg'
	
	result = predict_ethnic(image_path)
	#print(result)
	print(ETHNIC[np.argmax(result)])

"""
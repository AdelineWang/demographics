import argparse


if __name__ == "__main__":
	image = 'biden.jpg'
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=False,
		help="path to test image")
	args = vars(ap.parse_args())
	#print(args)
	print(args["image"])

	args["image"] = image

	print(args["image"])

	#result = predict_ethnic(args["image"])
	#print(result)
	#print(ETHNIC[np.argmax(result)])
import os
import numpy as np
from PIL import Image
import h5py
import cv2

# Global variables
MALE = np.arange(0, 5)
FEMALE = np.arange(5, 10)

# Folder path
PATH = "image"
FILE_FORMAT = (".tif", ".jpg")

# Get first three digits
def getImageId(name):
	return name[:3]

images = [None]
imagesResized = [None]
sex = [None]
ethnic = [None]
emotion = [None]

print("FILE3")
print(os.walk(PATH))
for subdir, dirs, files in os.walk(PATH):
	for file in files:
		if file.endswith(FILE_FORMAT):
			name = os.path.join(subdir, file)
			im = cv2.imread(name, cv2.IMREAD_COLOR)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
			
			# im.show()
			
			images.append(np.array(im))

			im = cv2.resize(im, (224, 224))
			
			imagesResized.append(np.array(im))
			 

			imageId = getImageId(file)

			#print(imageId[1])

			try:
				user_number = int(imageId[1])
				print("Your number is {0}.".format(user_number))
			except ValueError as e:
				print("ERROR:", e)
				break
			#print(sex)



			if user_number in MALE:
				#print(file)
				sex.append(1)
			else:
				sex.append(0)
			
			ethnic.append(int(imageId[0]) - 1)
			if (imageId[2].isdigit()):
				emotion.append(int(imageId[2]))
			else:
				emotion.append(0)


# Concatenate

images = np.float64(np.stack(images))
print(images)

imagesResized = np.float64(np.stack(imagesResized))
sex = np.stack(sex)
#print("SEX: ", sex.shape)
ethnic = np.stack(ethnic)
emotion = np.stack(emotion)

	
# Normalize data
images /= 255.0
imagesResized /= 255.0
# Save to disk
f = h5py.File("train.h5", "w")
# Create dataset to store images
X_dset = f.create_dataset('data', images.shape, dtype='f')
X_dset[:] = images
X_dset = f.create_dataset('dataResized', imagesResized.shape, dtype='f')
X_dset[:] = imagesResized

# Create dataset to store labels
y_dset = f.create_dataset('sex', sex.shape, dtype='i')
y_dset[:] = sex
y_dset = f.create_dataset('ethnic', ethnic.shape, dtype='i')
y_dset[:] = ethnic
y_dset = f.create_dataset('emotion', emotion.shape, dtype='i')
y_dset[:] = emotion
    
f.close()

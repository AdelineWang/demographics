from PIL import Image

im_path= r'./demo8.jpg'
im = Image.open(im_path)
im2 = im.resize((160,160))
im2.save(r'./demo6.jpg')

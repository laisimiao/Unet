from unet import *
from data import *
import matplotlib.pyplot as plt
import numpy as np

mydata = dataProcess(512,512)
imgs_test = mydata.load_test_data()

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('my_unet.hdf5')
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
print("save imgs_mask_test.npy")
np.save('./results/imgs_mask_test.npy', imgs_mask_test)


print("array to image")
imgs = np.load('./results/imgs_mask_test.npy')
imgs_name = sorted(glob.glob("./raw/test" + "/*." + "tif"))
for i in range(imgs.shape[0]):
    img = imgs[i]
    imgname = imgs_name[i]
    midname = imgname[imgname.rindex("/") + 1:]
    img_order = midname[:-4]
    img = array_to_img(img)
    img.save("./results/%s.jpg" % (img_order))



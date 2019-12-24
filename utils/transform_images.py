import os
from PIL import Image

# select dataset folder to check and destination folder to put output images in
path = '../datasets/beauty_dataset/img/beauty_dataset'
dest_path = '../datasets/beauty_dataset/img/beauty_dataset_scaled'

# destination resolution
dest_res = 2 ** 8

for i, file in enumerate(os.listdir(path)):
    
    # open image using PIL to detect resolution.
    img = Image.open(os.path.join(path,file))
    width, height = img.size
    
    # pad image if necessary
    if width != height:
        # create a new black picture in size of (max(height,width), max(height,width))
        padded_size = (max(height,width), max(height,width))
        black_img = Image.new("RGB", padded_size)  # 
        # define origin to paste the image on the newly created image
        location_x = int((padded_size[0] - width) / 2)
        location_y = int((padded_size[1] - height) / 2)
        # paste the image
        black_img.paste(img, (location_x,location_y))
        img = black_img
    
    # resize image to destination resolution and save in dest folder
    img = img.resize((dest_res,dest_res),Image.ANTIALIAS)
    img.save(os.path.join(dest_path,file),quality=95)

    if i % 100 == 0:
        print("saved {}/{} images".format(i,len(os.listdir(path))))

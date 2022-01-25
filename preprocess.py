import os 
import glob
import shutil
from PIL import Image
import cv2
from tifffile import imread, imwrite
import imageio
import skimage.util
import skimage.io
from skimage.transform import resize
import fileinput
from natsort import natsorted
import numpy as np
import tifffile
from osgeo import gdal
from natsort import natsorted
import natsort
Image.MAX_IMAGE_PIXELS = 100000000000
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import rasterio as rio
from skimage.exposure import rescale_intensity

#Tiffからpngへ変換
def convert_tif_to_png(input_folder,output_folder):
    print("start converting tif to png.")
    # print((input_folder+"/*.tif"))
    src = glob.glob(input_folder+"/*.tif")
    print(src)
    os.makedirs(output_folder, exist_ok=True)
    for file in src:
        file_basename = os.path.basename(file)
        file_output_name = file_basename[:-4]+".png"
        print("convert ",file_basename,"to ", file_output_name)
        gdal.Translate(
            os.path.join(output_folder,file_output_name),
            file,
            format="PNG"
        )
    print("converting tiff to png is complete.")

#マルチバンド画像からRGB画像の作成
def make_mul_rgb(img_path,dst):
    os.makedirs(dst,exist_ok=True)
    image_name = os.path.basename(img_path)
    print(image_name)
    with rio.open(img_path) as src:
        bands = src.read()
    print(bands.shape) #bands = [bands_num, height, width]
    print(bands[1].shape)
    bgr = np.dstack((bands[1,:,:], bands[2,:,:], bands[4,:,:]))
    p_low, p_high = np.percentile(bgr, (0, 99.5))
    bgr_8bit = rescale_intensity(bgr,in_range=(p_low, p_high), out_range=(0,255)).astype(np.uint8)
    bgr_8bit_resize = cv2.resize(bgr_8bit, (256, 256))
    print(os.path.join(dst,image_name[:-3]+"png"))
    cv2.imwrite(os.path.join(dst,image_name[:-3]+"png"), bgr_8bit_resize)

    print("making mul-rgb image is complete.")

#Tiff画像のサイズ変更
def resize_tiff(src,dst,size):
    os.makedirs(dst,exist_ok=True)
    files_original_list = glob.glob(src+"/*.tif*")
    print(len(files_original_list))
    for file in files_original_list:
        basename = os.path.basename(file)
        image = gdal.Open(file)
        resize = (size/650)*100
        output_file_name = os.path.join(dst, basename)
        gdal.Translate(output_file_name,image ,widthPct=resize,heightPct=resize)
    print("resizing tiff img is complete.")
    
#画像サイズ変更
def resize_img(copied_data,dst,width_resize, height_resize):
    copied_file_list = glob.glob(os.path.join(copied_data,"*"))
    os.makedirs(dst,exist_ok=True)
    print(len(copied_file_list))
    for file in copied_file_list:
        basename= os.path.basename(file)
        image = cv2.imread(file)
        image_resize = cv2.resize(image, (width_resize, height_resize))
        save_name = os.path.join(dst, basename)
        cv2.imwrite(save_name,image_resize)
    print("resizing image is complete.")

#画像の色深度確認
def check_color_bit(img_path):
    img = gdal.Open(img_path)
    print(img.ReadAsArray())

#縦方向に画像を結合
def v_concat(src, dst, save_name, width_num, height_num):
    os.makedirs(dst, exist_ok=True)
    img_list = glob.glob(src + "/*")
    print(len(img_list))
    img_list = natsorted(img_list)
    for i in range(width_num-1):
        img = cv2.imread(img_list[i*(height_num-1)])
        for j in range(1,height_num):
            img_added = cv2.imread(img_list[i*(height_num-1)+j])
            img = cv2.vconcat([img, img_added])
            cv2.imwrite(os.path.join(dst,str(i)+"_"+save_name), img)
    print("concatting images vertically is complete.")

#横方向に画像を結合
def h_concat(src, dst, save_name):
    os.makedirs(dst, exist_ok=True)
    img_list = glob.glob(src + "/*")
    img_list = natsorted(img_list)
    print(len(img_list))
    img = cv2.imread(img_list[0])
    for i in range(len(img_list)-2):
        img_added = cv2.imread(img_list[i+1])  
        img = cv2.hconcat([img, img_added])
        cv2.imwrite(os.path.join(dst, save_name), img)
    print("concatting images horizontally is complete.")

#グリッド作成
def concat_img_grid(src, dst, save_name):
    os.makedirs(dst, exist_ok=True)
    img_all_list = natsorted(glob.glob(src + "/*"))
    for i in range(5):
        img_list = img_all_list[i*5 : (i+1)+5-1]
        img_list = [skimage.io.imread(img) for img in img_list]
        img_c = skimage.util.montage(img_list, grid_shape = (1,5), padding_width =10, fill = (0,0,0), multichannel=True)
        print(img_c.shape)
        skimage.io.imsave(os.path.join(dst,str(i)+"_"+save_name), img_c)
    print("making grid image is complete.")

#画像名変更
def rename_img(img_name,a,b):
    img_name_new = img_name.replace(a,b)
    os.rename(img_name, img_name_new)
    print("renaming images is complete.")

#.txt内の画像名変更
def rename_img_in_txt(txt_file):
    with fileinput.FileInput(txt_file, inplace=True, backup = "_back") as f:
        for line in f:
            print(line.replace("_out", "").rstrip("\n"))
    print("renaming images in txt is complete.")

#画像のサイズ確認
def check_size(img):
    img = cv2.imread(img)
    print(img.shape)

#画像のコピ
def copy_img(img,dst):
    shutil.copy(img,dst)
    print("copy {} to {}".format(img, dst))

#不要なモデルの削除
def remove_model(model):
    os.remove(model)
    print("remove {}".format(model))


if __name__=="__main__":
    print("All tasks are complete.")

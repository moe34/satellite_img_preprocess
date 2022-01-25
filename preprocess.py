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

def resize_img(copied_data,dst,width_resize, height_resize):
    copied_file_list = glob.glob(os.path.join(copied_data,"*"))
    os.makedirs(dst,exist_ok=True)
    print(len(copied_file_list))
    for file in copied_file_list:
        # print("resize ", file)
        basename= os.path.basename(file)
        image = cv2.imread(file)
        # print(image.shape)#width,height,colorchannel
        # width_resize = image.shape[0]//size
        # height_resize = image.shape[1]//size
        image_resize = cv2.resize(image, (width_resize, height_resize))
        
        
        # image_resize = resize(image,(size,size))
        # print(image_resize.shape)
        save_name = os.path.join(dst, basename)
        cv2.imwrite(save_name,image_resize)
        
    print("resize finish.")
def convert(input_folder,output_folder):
    print("convert tif to png")
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

    print("convert finish.")
    
def resize_tiff(src,dst,size):
    os.makedirs(dst,exist_ok=True)
    files_original_list = glob.glob(src+"/*.tif*")
    print(len(files_original_list))
    for file in files_original_list:
        basename = os.path.basename(file)
        # print(basename)
        image = gdal.Open(file)
        # print(image.GetProjection)
        resize = (size/650)*100
        output_file_name = os.path.join(dst, basename)
        # print("output name : ",output_file_name)
        gdal.Translate(output_file_name,image ,widthPct=resize,heightPct=resize)

    print("resize_tiff finish.")
    """
    with Image.open(filename) as img:
        print(img.width, img.height)
        print(img.info)
    for key, value in img.tag.items():
        taginfo = TiffTags.lookup(key)

        #switching key and value
        tagdict = {(y,) : x for x, y in taginfo.enum.items()}

        print(taginfo.name, ':', tagdict.get(value, value))
    """
def concat_img_grid(src, dst, save_name):
    os.makedirs(dst, exist_ok=True)
    img_all_list = natsorted(glob.glob(src + "/*"))
    for i in range(5):
        img_list = img_all_list[i*5 : (i+1)+5-1]
        # img = skimage.io.imread(img_list[i])
        # print(img.shape)
        img_list = [skimage.io.imread(img) for img in img_list]
        
        img_c = skimage.util.montage(img_list, grid_shape = (1,5), padding_width =10, fill = (0,0,0), multichannel=True)
        print(img_c.shape)
        skimage.io.imsave(os.path.join(dst,str(i)+"_"+save_name), img_c)

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
    # cv2.imwrite(os.path.join(dst,image_name[:-3]+"png"), bgr_8bit)
    cv2.imwrite(os.path.join(dst,image_name[:-3]+"png"), bgr_8bit_resize)

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

def rename_img(img_name,a,b):
    img_name_new = img_name.replace(a,b)
    os.rename(img_name, img_name_new)

def rename_img_in_txt(txt_file):
    with fileinput.FileInput(txt_file, inplace=True, backup = "_back") as f:
        for line in f:

            print(line.replace("_out", "").rstrip("\n"))

def check_color_bit(img_path):
    img = gdal.Open(img_path)
    print(img.ReadAsArray())

def check_size(img):
    img = cv2.imread(img)
    print(img.shape)

def copy_img(img,dst):
    shutil.copy(img,dst)

def remove_model(model):
    os.remove(model)


if __name__=="__main__":
   
    # convert("../alos3-pseudo_ortho_1030010040c36f00", "../alos3_png")
    # resize_img("../alos3_png/pan_ortho_16bit_1030010040c36f00_1.png","../resize",500)
    # check_size("../pan_ortho_16bit_1030010040c36f00_1.png")
    # files = glob.glob("/workspace/ubuntu/somatsu/mul_1_256_0.3_overlay_pred/*")
    # print(len(files))
    # # resize_img("/workspace/ubuntu/somatsu/hconcat/*2.png", "/workspace/ubuntu/somatsu/tmp_0113",10)
    # img_orig = Image.open("/workspace/ubuntu/somatsu/hconcat/concat_2.png")
    # width, height = img_orig.size
    # img_resize = img_orig.resize((width//50, height//50))
    # img_resize.save("/workspace/ubuntu/somatsu/hconcat/concat_2_resize_3.png")

    # v_concat(src="/workspace/ubuntu/somatsu/mul_1_256_0.3_overlay_pred", dst="/workspace/ubuntu/somatsu/vconcat_256_0.3_ver2", save_name="concat.png", width_num=159, height_num=322)
    # h_concat(src="/workspace/ubuntu/somatsu/vconcat_256_0.3_ver2", dst="/workspace/ubuntu/somatsu/hconcat", save_name="concat_2.png")
    # h_concat(src="/workspace/ubuntu/somatsu/vconcat", dst="/workspace/ubuntu/somatsu/hconcat",save_name="concat.png")

    # img_list_Vegas = glob.glob("/workspace/nagao/remotesensing/scripts/Real-ESRGAN/results/Vegas_SR_by_Shanghai/*") 
    # img_list_Shanghai = glob.glob("/workspace/nagao/remotesensing/scripts/Real-ESRGAN/results/Vegas_SR_by_Vegas/*") 
    # print(len(img_list_Shanghai))
    # print(len(img_list_Vegas))
    # # model_list = glob.glob("/home/ubuntu/somatsu/spacenet_building_detection/model2/logs/snapshot*")
    # # model_list = natsorted(model_list)
    # for img in img_list_Shanghai:
    #     rename_img(img, "_out", "")
    # for img in img_list_Vegas:
    #     rename_img(img, "_out", "")
    # print(model_list)
    # print(len(model_list))
    # shutil.copy("/home/ubuntu/somatsu/spacenet_building_detection/models/model_512/snapshot_iter_8422","/home/ubuntu/somatsu/spacenet_building_detection/models/models_save/model_512/")
    # for i in range(len(model_list)-5):
        # model = model_list[i]
        # remove_model(model)
    # check_size("/home/ubuntu/nagao/remotesensing/data/spacenet/Vegas/val_LR/RGB-PanSharpen/_AOI_2_Vegas_img1.png")
    # input_folder = "/workspace/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256"
    # output_folder = "/workspace/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256_png"
    # convert(input_folder, output_folder)
    # aux_files = glob.glob("/workspace/spacenet_data/processed_data/AOI_4_Shanghai_Train/RGB-PanSharpen_256_8bit_png/*aux.xml")
    # for file in aux_files:
    #     remove_model(file)
    # mask = natsorted(glob.glob("/workspace/spacenet_data/processed_data/AOI_4_Shanghai_Train/buildingMaskImages_256/*"))
    # mask_png = natsorted(glob.glob("/workspace/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256_png/*"))
    # print(len(mask))
    # print(len(mask_png))
    # for i in range(len(mask_png)):
    #     mask_img = np.array(Image.open(mask[i]))
    #     mask_img_png = np.array(Image.open(mask_png[i]))
    #     if (i==0):
    #         print(mask_img)
    #         print(mask_img_png)

    #     if (mask_img[i] == mask_img_png[i]).all():
    #         pass
    #     else:
    #         print(i)
    # images_train = glob.glob("../../spacenet_data/processed_data/AOI_2_Vegas_Train/RGB-PanSharpen_256_8bit/*")
    # images_test = glob.glob("../../spacenet_data/processed_data/AOI_2_Vegas_Test_public/RGB-PanSharpen_256_8bit/*")
    # images_all = glob.glob("tmp/*")
    # images_all = natsorted(images_all)
    # print(len(images_all))
    # print(images_all[-2:-1])
    # mul_images = glob.glob("../../spacenet_data/raw_data/AOI_2_Vegas_Train/MUL/*")
    # print(len(mul_images))
    # dst = "../../spacenet_data/processed_data/AOI_2_Vegas_Train/RGB-MUL_8bit_png"
    # for img in mul_images:
    #     make_mul_rgb(img, dst)
    # make_mul_rgb("../alos3-pseudo_ortho_1030010040c36f00/mul_ortho_16bit_1030010040c36f00_1-6band.tif", "mul_rgb_sample")
    # img_list = glob.glob("../../spacenet_data/raw_data/AOI_2_Vegas_Train/MUL/*")
    # print(len(img_list))
    # for img in img_list:
    #     make_mul_rgb(img, "../../spacenet_data/processed_data/AOI_2_Vegas_Train/RGB-MUL_256_8bit_png")

    # resize_img("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256_MUL_png", \
    #         "../../spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_163_png", 163, 163)
    img_list = glob.glob("../../spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_163_png/*")
    for img in img_list:
        img_basename = os.path.basename(img)
        img_color = cv2.imread(img)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        print(img_gray.shape)
        print(os.path.join("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_163_gray_png",img_basename))
        cv2.imwrite(os.path.join("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_163_gray_png",img_basename), img_gray)
    # img_list = glob.glob("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256_MUL_png/*")
    # for img in img_list:
    #     rename_img(img, "RGB-PanSharpen_", "")
    # img1 = cv2.imread("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_163_png/AOI_2_Vegas_img1.png")
    # img2 = cv2.imread("/home/ubuntu/spacenet_data/processed_data/AOI_2_Vegas_Train/buildingMaskImages_256_MUL_png/AOI_2_Vegas_img1.png")
    # print(img1.shape)
    # print(img2.shape)

    

    print("finish.")
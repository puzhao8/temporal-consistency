
import os, imageio, cv2
import numpy as np
from imageio import imread, imsave
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont



imageFolder = Path("snic_results\elephant_s2_ConvLSTM_tc_10_dft_20210508T094653\outputs\elephant_s2_ConvLSTM_tc_10_dft_20210508T094653\maskArr")
saveFolder = Path(os.path.split(imageFolder)[0])


images = []
scale = 0.2 
for filename in sorted(os.listdir(imageFolder)):
    blank_image = Image.open(imageFolder / filename)
    w, h = np.array(blank_image).shape

    # im = Image.new('RGB', (width, width), color_1)
    

    blank_image = blank_image.resize((int(h*scale), int(w*scale)))
    img_draw = ImageDraw.Draw(blank_image)
    w_, h_ = np.array(blank_image).shape
    

    font = ImageFont.truetype("calibrib.ttf", 50) # timesbd.ttf
    txt = filename[:-4]
    img_draw.text((np.floor(h_*0.01), np.floor(w_*0.01)), txt, fill="white", font=font)

    # blank_image.save(saveFolder / filename)
    images.append(blank_image)

# images[0].save(saveFolder / "pillow_imagedraw_.gif",
#             save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)


# from images2gif import writeGif
# writeGif(saveFolder / "pillow_imagedraw_.gif", images, duration=0.8)


## Methods: imageio + geemap
if False:
    scale = 0.2
    save_gif_file = saveFolder / "pillow_imagedraw.gif"
    fileList = list(os.listdir(imageFolder))#.sort(key=len)

    ### sort by eopch
    def sort_by_epoch(name): return eval(name.split(".")[0].split("_")[1][2:])
    fileList.sort(reverse=False, key=sort_by_epoch)

    """ Write as .gif file """
    with imageio.get_writer(save_gif_file, mode='I') as writer:
        for filename in fileList:
            image = imread(imageFolder / filename)
            image_scaled = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
            writer.append_data(image_scaled)

    """ Add text to gif """
    from geemap import add_text_to_gif
    # texted_gif_file = savePath / 'out_texted.gif'
    fileNameList = [filename[:-4] for filename in fileList]
    add_text_to_gif(save_gif_file, save_gif_file, xy=("1%", "5%"), text_sequence=fileNameList,
                    font_type="arial.ttf", font_size=50, font_color="#ff0000",
                    add_progress_bar=True, progress_bar_color='green', progress_bar_height=5,
                    duration=800, loop=0)
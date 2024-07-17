import glob
from PIL import Image
from pathlib import Path
from os.path import join, exists, basename
from concurrent.futures import ThreadPoolExecutor

img_dir = join('.', 'FIVES-dataset', 'train', 'Original')
lbl_dir = join('.', 'FIVES-dataset', 'train', 'Ground truth')

# 构造图片列表
img_list = glob.glob(join(img_dir, '*.png'))
lbl_list = []
for img_path in img_list:
    img_name = basename(img_path)
    lbl_path = join(lbl_dir, img_name)
    if exists(lbl_path):
        lbl_list.append(lbl_path)
    else:
        img_list.remove(img_path)

# 定义输出目录
output_img_dir = join('.', 'FIVES-dataset', 'train', 'Images')
output_lbl_dir = join('.', 'FIVES-dataset', 'train', 'Labels')

# 创建输出目录
Path(output_img_dir).mkdir(parents=True, exist_ok=True)
Path(output_lbl_dir).mkdir(parents=True, exist_ok=True)

# 图像增强函数
def augment_image(image_path, label_path, output_img_dir, output_lbl_dir, overwrite=False):
    # 读取图像和标签
    print(f'Handling {basename(image_path)}')
    img = Image.open(image_path)
    lbl = Image.open(label_path)

    # 原始图像和标签
    img.save(join(output_img_dir, basename(image_path)))
    lbl.save(join(output_lbl_dir, basename(label_path)))

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(join(output_img_dir, f"fr_{basename(image_path)}"))
    flipped_lbl.save(join(output_lbl_dir, f"fr_{basename(label_path)}"))

    # 旋转和镜像
    for angle in [90, 180, 270]:
        rotated_img = img.rotate(angle, expand=True)
        rotated_lbl = lbl.rotate(angle, expand=True)
        if overwrite or not (
            exists(join(output_img_dir, f"r{angle}_{basename(image_path)}")) and 
            exists(join(output_lbl_dir, f"r{angle}_{basename(label_path)}"))
        ): 
            rotated_img.save(join(output_img_dir, f"r{angle}_{basename(image_path)}"))
            rotated_lbl.save(join(output_lbl_dir, f"r{angle}_{basename(label_path)}"))

        flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_lbl = rotated_lbl.transpose(Image.FLIP_LEFT_RIGHT)
        if overwrite or not (
            exists(join(output_img_dir, f"fr{angle}_{basename(image_path)}")) and 
            exists(join(output_lbl_dir, f"fr{angle}_{basename(label_path)}"))
        ): 
            flipped_img.save(join(output_img_dir, f"fr{angle}_{basename(image_path)}"))
            flipped_lbl.save(join(output_lbl_dir, f"fr{angle}_{basename(label_path)}"))

# 使用多线程执行图像增强
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(
        augment_image, 
        img_list, 
        lbl_list, 
        [output_img_dir]*len(img_list), 
        [output_lbl_dir]*len(lbl_list)
    )
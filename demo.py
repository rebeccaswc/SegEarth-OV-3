from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample
from segearthov3_segmentor import SegEarthOV3Segmentation

img_path = 'resources/0000002.jpg'

# name_list = ['background', 'grass', 'road',
#              'tree,forest', 'water,river', 'cropland', 'building,roof,house']

name_list = [ 'grass', 'road', 'sidewalk;pavement',
             'tree,forest',  'earth,ground', 'building,roof,house']

# 定義每個類別對應的顏色（R,G,B in [0,1]）
palette = [
    # (0.71, 0.47, 0.47),  # background (B47878)
    # (0.016, 0.784, 0.012),  # bareland/barren -> #04C803
    (0.016, 0.980, 0.027),  # grass -> #04FA07
    (0.549, 0.549, 0.549),  # road/route -> #8C8C8C
    (0.922, 1, 0.028),  # sidewalk/pavement -> #EBFF07
    (0.016, 0.784, 0.012),  # tree/forest -> #04C803
    # (0.239, 0.902, 0.980),  # water/river -> #3DE6FA
    (0.471, 0.471, 0.275),  # earth / ground (#787846)
    (0.706, 0.471, 0.471),  # building/roof/house (FF3D06)
]

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img = Image.open(img_path)
img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') # This variable is only a placeholder; the actual data is read within the model. (To be optimized)

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size[::-1] # H, W
}
data_sample.set_metainfo(img_meta)


model = SegEarthOV3Segmentation(
    type='SegEarthOV3Segmentation',
    model_type='SAM3',
    classname_path='./configs/my_name.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

seg_pred = model.predict(img_tensor, data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)

from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
# fig, ax = plt.subplots(figsize=(6, 6))

cmap = ListedColormap(palette)
ax[1].imshow(seg_pred, cmap=cmap, vmin=0, vmax=len(name_list) - 1)
ax[1].axis('off')
# ax.imshow(seg_pred, cmap=cmap, vmin=0, vmax=len(name_list) - 1)
# ax.axis('off')

# # 將顏色對應表輸出到 console
# print('=== Segmentation color map ===')
# for idx, (name, col) in enumerate(zip(name_list, palette)):
#     print(f'{idx:02d}: {name} -> rgb{tuple(int(c*255) for c in col)}')

# legend for the plot
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(name_list))]
ax[1].legend(handles, name_list, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# ax.legend(handles, name_list, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
# plt.show()
plt.savefig(f'seg.png', bbox_inches='tight')

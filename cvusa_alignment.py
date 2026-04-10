import os
import time
import traceback
import numpy as np
import cv2
import pyproj
import shapely.ops
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

try:
    import osmnx as ox
    # Configure osmnx for robust network requests to prevent server timeouts
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 180  # Increased timeout
    ox.settings.retry_count = 5         # Retry up to 5 times
    ox.settings.retry_pause = 10        # Wait 10 seconds between retries
    ox.settings.overpass_settings = "[out:json][timeout:180]"
    OSM_ENABLED = True
except ImportError:
    OSM_ENABLED = False
    print("警告: OSMnx 未安裝，將跳過 OSM 資料處理。")

try:
    import torch
    from torchvision import transforms
    from mmseg.structures import SegDataSample
    from segearthov3_segmentor import SegEarthOV3Segmentation
    SEGMENTATION_ENABLED = True
except ImportError:
    SEGMENTATION_ENABLED = False
    print("警告: SegEarth-3 或相關依賴未安裝，將跳過預測階段。")

# ================= Configuration =================
IMG_DIR = "bingmap/18"
CSV_PATH = "all.csv"
OUTPUT_DIR = "output_aligned"
NUM_IMAGES_TO_PROCESS = 4  # MVP 設定為 2
ZOOM = 18
TARGET_SIZE = 224

# SegEarth-3 Categories and Palette
CLASS_NAMES = ['grass', 'road', 'sidewalk;pavement', 'tree,forest', 'earth,ground', 'building,roof,house']
SEG_PALETTE = [
    (0.016, 0.980, 0.027),  # grass -> #04FA07
    (0.549, 0.549, 0.549),  # road/route -> #8C8C8C
    (0.922, 1.000, 0.028),  # sidewalk/pavement -> #EBFF07
    (0.016, 0.784, 0.012),  # tree/forest -> #04C803
    (0.471, 0.471, 0.275),  # earth / ground (#787846)
    (0.706, 0.471, 0.471),  # building/roof/house (FF3D06)
]

# OSM Rasterization Colors
SURFACE_COLORS = {
    'asphalt': (128, 128, 128, 200),
    'paved': (128, 128, 128, 200),
    'unpaved': (139, 69, 19, 200),
    'dirt': (160, 82, 45, 200),
    'concrete': (211, 211, 211, 200),
    'paving_stones': (105, 105, 105, 200),
    'default': (255, 0, 0, 150)
}

# ================= Workflow Functions =================

def calculate_bbox_epsg3857(lat, lon, zoom=ZOOM, size=TARGET_SIZE):
    """
    1. 空間範圍計算 (BBox Calculation)
    使用投影座標系統 (EPSG:3857) 計算精準物理邊界，再轉回 WGS84 給 OSMnx。
    """
    transformer_to_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_to_4326 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    # EPSG:3857 在給定 Zoom Level 下的解析度 (meters/pixel)
    resolution = 156543.03392 / (2 ** zoom)
    
    # 中心點轉為 Web Mercator (meters)
    cx, cy = transformer_to_3857.transform(lon, lat)
    
    # 計算半寬度/半高度 (meters)
    half_dist = (size / 2.0) * resolution
    
    minx, miny = cx - half_dist, cy - half_dist
    maxx, maxy = cx + half_dist, cy + half_dist
    
    # 轉回 WGS84
    lon_west, lat_south = transformer_to_4326.transform(minx, miny)
    lon_east, lat_north = transformer_to_4326.transform(maxx, maxy)
    
    return {
        'north': lat_north, 'south': lat_south,
        'east': lon_east, 'west': lon_west,
        'minx_3857': minx, 'maxy_3857': maxy,
        'resolution': resolution
    }

def fetch_and_rasterize_osm(bbox, size=TARGET_SIZE):
    """
    2. OSM 資料獲取與柵格化 (OSM Retrieval & Rasterization)
    """
    if not OSM_ENABLED:
        return np.zeros((size, size, 3), dtype=np.uint8), np.zeros((size, size), dtype=np.uint8)
    
    tags = {'highway': True, 'surface': True, 'building': True, 'natural': True, 'landuse': True}
    
    print(f"   => 從 OSM 獲取地理特徵 (BBox: {bbox['south']:.5f}, {bbox['west']:.5f} to {bbox['north']:.5f}, {bbox['east']:.5f})")
    try:
        gdf = ox.features_from_bbox(bbox=(bbox['west'], bbox['south'], bbox['east'], bbox['north']), tags=tags)
           
    except Exception as e:
        print(f"   => OSMnx 獲取失敗或無資料: {e}")
        return np.zeros((size, size, 3), dtype=np.uint8), np.zeros((size, size), dtype=np.uint8)
        # try:
        #     center_lat = (bbox['north'] + bbox['south']) / 2.0
        #     center_lon = (bbox['east'] + bbox['west']) / 2.0
        #     lat_diff = (bbox['north'] - bbox['south']) * 111000.0
        #     lon_diff = (bbox['east'] - bbox['west']) * (111000.0 * np.cos(np.radians(center_lat)))
        #     dist = int(max(50, np.hypot(lat_diff, lon_diff) / 2.0))
        #     gdf = ox.features_from_point((center_lat, center_lon), tags=tags, dist=dist) if hasattr(ox, 'features_from_point') else ox.geometries_from_point((center_lat, center_lon), tags=tags, dist=dist)
        # except Exception as e2:
        #     print(f"   => OSMnx 中心點獲取失敗或無資料: {e2}")
        #     return np.zeros((size, size, 3), dtype=np.uint8), np.zeros((size, size), dtype=np.uint8)
        
    # 準備柵格化的影像
    osm_img = Image.new('RGB', (size, size), (0, 0, 0))
    osm_road_mask = Image.new('L', (size, size), 0)
    
    draw = ImageDraw.Draw(osm_img, "RGBA")
    draw_road = ImageDraw.Draw(osm_road_mask)
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    def coords_to_pixels(coords):
        pixels = []
        for lon, lat in coords:
            if pd.isna(lon) or pd.isna(lat):
                continue
            mx, my = transformer.transform(lon, lat)
            if pd.isna(mx) or pd.isna(my) or np.isinf(mx) or np.isinf(my):
                continue
            px = int((mx - bbox['minx_3857']) / bbox['resolution'])
            py = int((bbox['maxy_3857'] - my) / bbox['resolution'])
            pixels.append((px, py))
        return pixels

    if gdf is not None and not gdf.empty:
        for idx, row in gdf.iterrows():
            geom = row.geometry
                
            # 繪製道路 (線段)
            if geom.geom_type in ['LineString', 'MultiLineString'] and 'highway' in row and pd.notna(row['highway']):
                surface = str(row.get('surface', 'default')).lower()
                color = SURFACE_COLORS.get(surface, SURFACE_COLORS['default'])
                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
                
                for line in lines:
                    px_coords = coords_to_pixels(line.coords)
                    if len(px_coords) > 1:
                        draw.line(px_coords, fill=color, width=3)
                        draw_road.line(px_coords, fill=255, width=3) # 純道路 Mask，對齊用
                        
            # 繪製建築物或綠地 (多邊形)
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for poly in polys:
                    px_coords = coords_to_pixels(poly.exterior.coords)
                    if len(px_coords) >= 3:
                        if 'building' in row and pd.notna(row['building']):
                            draw.polygon(px_coords, fill=(255, 100, 100, 180)) # 建築紅色
                        elif 'natural' in row or 'landuse' in row:
                            draw.polygon(px_coords, fill=(100, 255, 100, 150)) # 綠地綠色
                        
    return np.array(osm_img), np.array(osm_road_mask)

def perform_segearth_classification(image, seg_model, img_path):
    """
    3. SegEarth-3 分類
    """
    if not SEGMENTATION_ENABLED or seg_model is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        
    print(f"   => 執行 SegEarth-3 預測...")
    img_tensor = transforms.Compose([transforms.ToTensor()])(image).unsqueeze(0).to(seg_model.device)
    data_sample = SegDataSample()
    data_sample.set_metainfo({'ori_shape': (TARGET_SIZE, TARGET_SIZE), 'img_path': img_path})
    
    seg_pred = seg_model.predict(img_tensor, data_samples=[data_sample])
    seg_idx = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0).astype(np.uint8)
    
    return seg_idx

def homography_correction(osm_rgb, osm_road_mask, seg_idx):
    """
    4. 特徵對齊修正 (Homography Correction / ECC Alignment)
    使用 cv2.findTransformECC 計算道路特徵位移，將 OSM 對齊至衛星圖。
    """
    # 提取 SegEarth 中的道路 (類別索引 1 是 road)
    seg_road_mask = (seg_idx == 1).astype(np.uint8) * 255
    
    # 如果兩張圖任一沒有道路特徵，退回單位矩陣
    if np.sum(osm_road_mask) == 0 or np.sum(seg_road_mask) == 0:
        print("   => 缺乏足夠道路特徵進行對齊，使用原始設定。")
        return osm_rgb, np.eye(2, 3, dtype=np.float32)
        
    # 初始化與高斯模糊 (幫助梯度下降最佳化)
    osm_float = cv2.GaussianBlur(osm_road_mask.astype(np.float32), (15, 15), 0)
    seg_float = cv2.GaussianBlur(seg_road_mask.astype(np.float32), (15, 15), 0)
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # MOTION_TRANSLATION 通常對 CVUSA 中心的微小平移最穩定
    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    
    success = False
    try:
        # 尋找從 OSM -> SEG 的轉換矩陣
        _, warp_matrix = cv2.findTransformECC(osm_float, seg_float, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        success = True
        print(f"   => ECC 對齊成功！平移: X={warp_matrix[0,2]:.2f}, Y={warp_matrix[1,2]:.2f}")
    except cv2.error as e:
        # 若 ECC 失敗 (例如陷入局部極小值或重疊度過低)，回退至相位相關 (Phase Correlate)
        shift, _ = cv2.phaseCorrelate(osm_float, seg_float)
        warp_matrix[0, 2] = shift[0]
        warp_matrix[1, 2] = shift[1]
        print(f"   => ECC 對齊失敗，回退至 Phase Correlate，平移: X={warp_matrix[0,2]:.2f}, Y={warp_matrix[1,2]:.2f}")
        
    # Warp OSM RGB 以對齊 Satellite
    aligned_osm_rgb = cv2.warpAffine(osm_rgb, warp_matrix, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_LINEAR)
    
    return aligned_osm_rgb, warp_matrix

def init_segmentation_model():
    if not SEGMENTATION_ENABLED:
        return None
    os.makedirs('configs', exist_ok=True)
    with open('./configs/my_name.txt', 'w') as f:
        f.write('\n'.join(CLASS_NAMES))
        
    print("初始化 SegEarthOV3 模型...")
    model = SegEarthOV3Segmentation(
        type='SegEarthOV3Segmentation', model_type='SAM3',
        classname_path='./configs/my_name.txt',
        prob_thd=0.01, confidence_threshold=0.01,
        slide_stride=512, slide_crop=512,
        use_presence_score=False,
    )
    return model

# ================= Main Execution =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 讀取 CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH, header=None, names=['lat', 'lon', 'c', 'd', 'e'])
    images = sorted([os.path.join(IMG_DIR, fp) for fp in os.listdir(IMG_DIR) if fp.lower().endswith(('.png', '.jpg'))])
    
    max_proc = min(NUM_IMAGES_TO_PROCESS, len(images), len(df))
    print(f"開始處理 {max_proc} 張影像...")
    
    seg_model = init_segmentation_model()
    palette_rgb = (np.array(SEG_PALETTE) * 255).round().astype(np.uint8)
    
    for i in range(max_proc):
        lat = float(df.iloc[i]['lat'])
        lon = float(df.iloc[i]['lon'])
        img_path = images[i]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        print(f"\n[{i+1}/{max_proc}] 處理: {base_name} @ ({lat:.5f}, {lon:.5f})")
        
        try:
            # 讀取並 Crop/Resize
            sat_img = Image.open(img_path).convert('RGB')
            if sat_img.size != (TARGET_SIZE, TARGET_SIZE):
                sat_img = sat_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                
            # 1. 空間範圍計算
            bbox = calculate_bbox_epsg3857(lat, lon, ZOOM, TARGET_SIZE)
            
            # 2. 獲取 OSM 與柵格化
            osm_rgb, osm_road_mask = fetch_and_rasterize_osm(bbox, TARGET_SIZE)
            
            # 3. SegEarth 分類
            seg_idx = perform_segearth_classification(sat_img, seg_model, img_path)
            
            # 將分類 Index 轉為 RGB 預覽圖
            seg_vis_np = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            for cls_idx in range(len(CLASS_NAMES)):
                seg_vis_np[seg_idx == cls_idx] = palette_rgb[cls_idx]
                
            # 4. 對齊修正
            aligned_osm_rgb, warp_matrix = homography_correction(osm_rgb, osm_road_mask, seg_idx)
            
            # 5. 輸出結果與視覺化
            # 生成 Fusion (例如：衛星底圖 + 對齊後的 OSM)
            sat_np = np.array(sat_img)
            fusion_vis = cv2.addWeighted(sat_np, 0.6, aligned_osm_rgb, 0.8, 0)
            
            # 新增: 語義分割 (Segmentation) 融合對齊後的 OSM 標籤 (OSM Tag)
            seg_osm_fusion = cv2.addWeighted(seg_vis_np, 0.6, aligned_osm_rgb, 0.8, 0)
            
            # 保存結果
            Image.fromarray(sat_np).save(os.path.join(OUTPUT_DIR, f"{base_name}_1_sat.png"))
            Image.fromarray(seg_vis_np).save(os.path.join(OUTPUT_DIR, f"{base_name}_2_seg.png"))
            Image.fromarray(osm_rgb).save(os.path.join(OUTPUT_DIR, f"{base_name}_3_osm_raw.png"))
            Image.fromarray(aligned_osm_rgb).save(os.path.join(OUTPUT_DIR, f"{base_name}_4_osm_aligned.png"))
            Image.fromarray(fusion_vis).save(os.path.join(OUTPUT_DIR, f"{base_name}_5_fusion.png"))
            # 將新的 Seg+OSM 存出
            Image.fromarray(seg_osm_fusion).save(os.path.join(OUTPUT_DIR, f"{base_name}_6_seg_osm_fusion.png"))
            
            # 建立一個整合預覽圖 (Matplotlib) - 擴充至 5 個子圖
            fig, axs = plt.subplots(1, 5, figsize=(25, 5))
            axs[0].imshow(sat_np)
            axs[0].set_title("1. Original Satellite")
            axs[0].axis('off')
            
            axs[1].imshow(seg_vis_np)
            axs[1].set_title("2. SegEarth Mask")
            axs[1].axis('off')
            
            axs[2].imshow(osm_road_mask, cmap='gray')
            axs[2].set_title("3. OSM Road Target")
            axs[2].axis('off')
            
            axs[3].imshow(fusion_vis)
            axs[3].set_title("4. Aligned Fusion")
            axs[3].axis('off')
            
            axs[4].imshow(seg_osm_fusion)
            axs[4].set_title("5. Seg + OSM")
            axs[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_summary.png"))
            plt.close()
            
            print(f"   => 完成並保存至 {OUTPUT_DIR}/")
            
        except Exception as e:
            print(f"處理失敗 {base_name}: {e}")
            traceback.print_exc()
            
    print("\n✅ 所有流程處理完畢。")

if __name__ == "__main__":
    main()
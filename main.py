# main.py
import carla
import open3d as o3d
import os
import shutil
import numpy as np
from dataset import generate_simulation_data, satellite_capture_images
from localization import filter_and_cluster_building_tops, generate_simulated_2d_features, test
from util import visualize_point_cloud, scale_and_convert_to_int, calculate_top_10_percent_z, analyze_point_cloud_range, extract_roof_features_with_image, extract_roof_features_from_point_cloud
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def main():
    trigger_for_good_looking = True

    # -------------------- 以下是生成數值用的code------------------------------
    for i in range (1):
        if trigger_for_good_looking:
    # # Carla 連線
    # client = carla.Client('localhost', 2000)
    # client.set_timeout(10.0)
    # world = client.get_world()
    # world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # # 衛星影像保存目錄
    # map_image_dir = "data/map_image"
    # # if os.path.exists(map_image_dir):
    # #     shutil.rmtree(map_image_dir)
    # # os.makedirs(map_image_dir, exist_ok=True)

    # RADIUS = 700  # 圓形路徑的半徑（以公尺為單位，根據需要調整）
    # ALTITUDE = 1000  # 相機的高度（以公尺為單位，視需要調整）
    # ANGLE_INCREMENT = 5  # 捕捉影像的角度增量（以度為單位）

    # # 拍攝衛星影像
    # print("開始捕捉衛星影像...")
    # try:
    #     satellite_capture_images(client, RADIUS, ANGLE_INCREMENT, map_image_dir)
    #     print(f"影像已經儲存到 {map_image_dir}")
    # except Exception as e:
    #     print(f"發生錯誤: {e}")
    # print("衛星影像拍攝完成!")
    
    # # # 開始模擬實驗
    # # num_experiments = 2     # 重新部署人車分佈次數
    # # duration = 10            # 每輪的持續秒數
    # # vehicle_num = 40        # 設定城市裡有多少台車
    # # select_num = 20          # 每次監控多少台車並取得資料
    # # walker_num = 100        # 設定城市裡有多少行人

    # # for i in range(num_experiments):
    # #     print(f"正在執行實驗 {i}...")
    # #     experiment_dir = f"data/{i}"
    # #     if os.path.exists(experiment_dir):
    # #         shutil.rmtree(experiment_dir)
    # #     os.makedirs(experiment_dir, exist_ok=True)
        
    # #     # 生成車輛和行人數據
    # #     print("Generating simulation data (vehicles and walkers)...")
    # #     generate_simulation_data(world, i, experiment_dir, duration, vehicle_num, walker_num, select_num)

    #     # 執行
    # # os.system()
    # print("所有實驗已完成！")
            break
    
    # -------------------- 以下是A_match用的code------------------------------
    data_dir = "data" 
    ciry_pcd = "Town01.pcd"                                              # 原始資料夾路徑
    city_point_cloud_path = "data/city_point_cloud_int.ply"         # 全局點雲                           # 存放2D點雲的路徑（目前還是從3D點雲抓取的）
    global_point_cloud_path = "data/city_point_cloud_int.ply"       # 存放3D點雲的路徑 (已將座標全部*10，並過濾到剩下z軸高於30公尺的點）
    raw_global_point_cloud_path = "data/city_point_cloud_raw.ply"   # 存放3D點雲的路徑 (已將座標全部*10，但保持完整結構）
    output_dir = "match_results"                                    # 輸出結果資料夾
    real_dataset_for_p1_and_p2 = "A_match_results"                  # 真正從影像中擷取的相對位置特徵點

    for i in range (1):
        if trigger_for_good_looking:
    # 如果 output_dir 已存在，刪除後重建
    # if os.path.exists(output_dir):
    #     print(f"清空舊的輸出資料夾: {output_dir}")
    #     shutil.rmtree(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
            break

    # 真正從影像中提取特徵點的工作（因為時間有限所以暫時放置）
    for i in range (1):
        if trigger_for_good_looking:
    # # 記錄被刪除的影像路徑
    # deleted_img_path = os.path.join(output_dir, "deleted_img.txt")
    # with open(deleted_img_path, "w") as deleted_file:
    #     for root, _, files in os.walk(data_dir):
    #         for file in files:
    #             if file.startswith("rgb_") and file.endswith(".png"):
    #                 rgb_path = os.path.join(root, file)
    #                 depth_path = rgb_path.replace("rgb_", "depth_")
    #                 if not os.path.exists(depth_path):
    #                     print(f"缺少深度影像，跳過: {rgb_path}")
    #                     continue

    #                 # 提取相對輸出路徑
    #                 vehicle_dir = os.path.relpath(root, data_dir)
    #                 output_vehicle_dir = os.path.join(output_dir, vehicle_dir)
    #                 os.makedirs(output_vehicle_dir, exist_ok=True)

    #                 marked_image_path = os.path.join(output_vehicle_dir, f"marked_{file}")
    #                 feature_npy_path = os.path.join(output_vehicle_dir, f"features_{file.replace('.png', '.npy')}")

    #                 print(f"處理影像: {rgb_path}")

    #                 # 呼叫水平掃描法提取特徵點
    #                 image_features = extract_roof_features_with_image(
    #                     rgb_path, depth_path, marked_image_path
    #                 )

    #                 # 判斷是否為有效影像
    #                 # if len(image_features) == 0:
    #                 #     print(f"影像不符合條件，移除: {rgb_path}")
    #                 #     deleted_file.write(f"{rgb_path}\n")
    #                 #     os.remove(rgb_path)
    #                 #     os.remove(depth_path)
    #                 #     continue

    #                 # 保存特徵點
    #                 np.save(feature_npy_path, image_features)
    #                 print(f"保存特徵點至: {feature_npy_path}")

    # # 處理全局點雲（如有需要）
    # print(f"處理全局點雲: {city_point_cloud_path}")
    # point_cloud_features, roof_ply_path = extract_roof_features_from_point_cloud(city_point_cloud_path, output_dir)
    # print(f"全局點雲特徵點已保存至: {roof_ply_path}")
            break
    
    # 步驟 1: 篩選建築物頂部點並聚類
    clusters = filter_and_cluster_building_tops(city_point_cloud_path, z_threshold=0, eps=20.0, min_points=10)
    # print(f"clusters: {clusters}")

    # 步驟 2: 生成 2D 特徵點模擬數據
    # generate_simulated_2d_features(clusters, data_dir, sample_count=100, error_range=5.0)
    sample_count=100
    generate_simulated_2d_features(clusters, data_dir, sample_count, error_range=5.0)

    # 實測匹配算法及生出可視化的結果
    test(data_dir,output_dir, global_point_cloud_path, raw_global_point_cloud_path, sample_count)
    visualize_point_cloud(ciry_pcd)

if __name__ == "__main__":
    main()
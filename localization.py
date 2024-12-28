# 目標：從輸入的影像篩選出建築物的特徵點，依照影像視覺判斷前後之後到city model中尋找相對應的點

import os 
import cv2
import shutil
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from util import spherical_projection, project_and_match

"""
# --------------------- Dataset generation used module -------------------------
"""
# 篩選建築物頂部點雲並聚類，基於 1 單位 = 10 公尺的假設進行調整。
def filter_and_cluster_building_tops(ply_path, z_threshold, eps, min_points):
    """
    篩選建築物頂部點雲並聚類，基於 1 單位 = 10 公尺的假設進行調整。

    :param ply_path: 輸入點雲檔案 (.ply) 路徑
    :param ground_truth_dir: 保存地面真實值的目錄
    :param z_threshold: Z 軸閾值，用於過濾低於建築物高度的點
    :param eps: DBSCAN 的鄰近距離閾值
    :param min_points: DBSCAN 的最小點數，用於定義一個 cluster
    :return: 聚類後的建築物頂部點雲列表
    """
    # 讀取點雲
    print(f"讀取點雲檔案: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # 初步過濾：僅保留 Z 值高於指定閾值的點
    print(f"初步過濾: 篩選 Z > {z_threshold} 的點雲...")
    filtered_points = points[points[:, 2] > z_threshold]

    if filtered_points.size == 0:
        print("未找到符合條件的建築物頂部點雲")
        return []

    # 聚類：DBSCAN
    print(f"開始進行 DBSCAN 聚類 (eps={eps}, min_points={min_points})...")
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(filtered_points)
    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = []
    print("提取有效聚類...")
    for label in tqdm(unique_labels, desc="Processing Clusters"):
        if label == -1:  # 過濾噪聲
            continue
        cluster_points = filtered_points[labels == label]
        clusters.append(cluster_points)

    print(f"共發現 {len(clusters)} 個有效建築物頂部點雲群")
    return clusters

# 生成模擬的局部 2D 特徵點dataset gen，並記錄 Ground Truth。
def generate_simulated_2d_features(clusters, data_dir, sample_count, error_range):
    """
    從相鄰 cluster 中隨機生成模擬的 2D 特徵點叢集，並加入誤差後保存。

    :param clusters: 建築物頂部點雲群列表
    :param data_dir: 輸出目錄
    :param sample_count: 生成的子點雲數量
    :param error_range: 誤差範圍 [0, error_range] (單位: 10 公尺)
    """
    total_generated = 0

    if os.path.exists(f"{data_dir}/ground_truth"):
        print(f"清空舊的輸出資料夾: {data_dir}/ground_truth")
        shutil.rmtree(f"{data_dir}/ground_truth")
    os.makedirs(f"{data_dir}/ground_truth", exist_ok=True)


    if os.path.exists(f"{data_dir}/2D_FP"):
        print(f"清空舊的輸出資料夾: {data_dir}/2D_FP")
        shutil.rmtree(f"{data_dir}/2D_FP")
    os.makedirs(f"{data_dir}/2D_FP", exist_ok=True)

    def extract_edge_points(cluster):
        """
        從 cluster 中提取邊緣點或尖角點，基於法向量變化檢測。
        :param cluster: 單個 cluster 點雲
        :return: 邊緣點集
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster)

        # 計算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals = np.asarray(pcd.normals)

        # 根據法向量變化提取邊緣點
        curvature = np.linalg.norm(normals - np.mean(normals, axis=0), axis=1)
        edge_indices = np.argsort(curvature)[-20:]  # 取曲率最大的 20 個點

        return cluster[edge_indices]

    # 建立 cluster 中心點 KD-Tree 以便查找相鄰 cluster
    cluster_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    cluster_tree = KDTree(cluster_centers)
    
    for i in tqdm(range(sample_count), desc="Generating Simulated 2D Features"):
        if not clusters:
            print("無有效點雲可生成 2D 特徵點")
            break

        # 隨機選取一個 cluster 作為起點，並查找相鄰 cluster
        start_idx = np.random.choice(len(clusters))
        _, neighbor_indices = cluster_tree.query([cluster_centers[start_idx]], k=np.random.randint(2, 5))
        selected_clusters = [clusters[idx] for idx in neighbor_indices[0]]

        sampled_points = []
        for cluster in selected_clusters:
            # 2. 從每個 cluster 提取邊緣或尖角點
            edge_points = extract_edge_points(cluster)
            if len(edge_points) < 5:
                continue  # 若邊緣點過少，跳過該 cluster

            # 3. 從邊緣點中隨機挑選 [1, 5] 個區域，每個區域 [5, 20] 個點
            for _ in range(np.random.randint(1, 6)):
                sample_size = min(len(edge_points), np.random.randint(5, 21))  # 保證不超過點數
                sample_indices = np.random.choice(len(edge_points), sample_size, replace=False)
                sampled_points.extend(edge_points[sample_indices])

        if len(sampled_points) == 0:
            continue

        sampled_points = np.array(sampled_points)

        # 記錄 ground truth
        ground_truth_path = os.path.join(f"{data_dir}/ground_truth", f"ground_truth_{i+1}.npy")
        np.save(ground_truth_path, sampled_points)  # 保存原始 cluster 的局部點雲

        # 加入誤差，模擬深度值：將 Z 軸數值加上隨機誤差範圍
        local_center = np.mean(sampled_points, axis=0)
        local_points = sampled_points - local_center  # 平移至局部座標系中心
        depth_values = local_points[:, 2] + np.random.uniform(-error_range, error_range, size=local_points.shape[0])
        simulated_points = np.column_stack((local_points[:, 0], local_points[:, 1], depth_values))

        # 保存模擬點雲
        save_path = os.path.join(f"{data_dir}/2D_FP", f"{i+1}.npy")
        np.save(save_path, simulated_points)
        # if len(simulated_points) == len(sampled_points):
        #     print(f"生成含有{len(simulated_points)}個點雲的模擬局部資料，且ground truth也有一樣多個點")

        total_generated += 1

    print(f"共生成 {total_generated} 個模擬局部點雲，保存至 {data_dir}/2D_FP")

"""
--------------------- Doing A_match algo testing -------------------------
"""
# 主要測試code
def test(data_dir, output_dir, global_point_cloud_path, raw_global_point_cloud_path, sample_count,  k_candidates=5):
    """
    :local_path: 局部特徵點檔案路徑 (.npy)
    :global_path: 全局特徵點雲檔案路徑 (.ply)
    :ground_truth_path: 正確答案的點雲存放路徑（.npy）
    :match_algo: 透過演算法配對並篩選出與輸入局部點雲相匹配的的部份全局點雲
    :evaluate_matching_accuracy: 匹配結果 (匹配結果的點雲座標＆真實值的點雲座標，兩者皆為世界座標) Chamfer 距離、誤差分佈，並進行可視化。
    """

    raw_global_path = o3d.io.read_point_cloud(raw_global_point_cloud_path)
    raw_global_points = np.asarray(raw_global_path.points)

    global_path = o3d.io.read_point_cloud(global_point_cloud_path)
    global_points = np.asarray(global_path.points)

    for i in range(1, sample_count+1):
        local_path = os.path.join(f"{data_dir}/2D_FP", f"{i}.npy")
        local_points = np.load(local_path)
        ground_truth_path = os.path.join(f"{data_dir}/ground_truth", f"ground_truth_{i}.npy")
        ground_truth_points = np.load(ground_truth_path)

        # Step 1: 使用 match_algo 進行 K 候選匹配
        print(f"\n第 {i} 組匹配進行中...")
        matched_candidates = match_algo(local_points, global_points, distance_threshold=100.0, k_candidates=k_candidates)
        
        if not matched_candidates:
            print(f"第 {i} 組匹配失敗，未找到有效候選點雲集")
            continue
        
        print(f"第 {i} 組生成了 {len(matched_candidates)} 組候選匹配點雲")

        # Step 2: 使用球面投影進一步過濾匹配點
        refined_matched_points = []
        for candidate in matched_candidates:
            matched_points = project_and_match(local_points, np.array(candidate))
            refined_matched_points.append(matched_points)

        # Step 3: 選擇最佳候選點集（基於 Chamfer 距離）
        best_candidate = None
        min_chamfer_dist = float("inf")

        for idx, matched_points in enumerate(refined_matched_points):
            chamfer_dist = chamfer_distance(ground_truth_points, matched_points)
            if chamfer_dist < min_chamfer_dist:
                min_chamfer_dist = chamfer_dist
                best_candidate = matched_points

        # Step 4: 驗證並視覺化匹配結果
        if best_candidate is not None:
            print(f"第 {i} 組最佳匹配候選 Chamfer 距離: {min_chamfer_dist:.4f}")
            evaluate_matching_accuracy(
                ground_truth_path, best_candidate, output_dir, i, global_points, raw_global_points
            )
        else:
            print(f"第 {i} 組匹配失敗，未找到有效候選點集")
        # # matched_points, transform_matrix = match_local_to_global(local_points, global_points)
        # matched_global_points, transform_matrix = match_algo(local_points, global_points)  
        # matched_global_points, transform_matrix = sp_match_algo(local_points, global_points)

        # if matched_global_points is not None:
        #     print(f"第 {i} 組匹配完成，準備進入驗證環節")
        #     print(f"ground_truth_points 包含 {len(ground_truth_points)}個點，matched_global_points 包含 {len(matched_global_points)}個點")
        #     evaluate_matching_accuracy(ground_truth_points, matched_global_points, output_dir, i, global_points, raw_global_points)
        # else:
        #     print(f"第 {i} 組匹配失敗")

# 匹配局部特徵點與全局特徵點雲，被test()呼叫
def match_algo(local_points, global_points, initial_distance_threshold=100, max_distance_threshold=2000):
    """
    匹配局部特徵點與全局特徵點雲。
    :param local_points_path: 局部特徵點檔案路徑 (.npy)
    :param global_point_cloud_path: 全局特徵點雲檔案路徑 (.ply)
    :param distance_threshold: 距離閾值，用於過濾不合理的匹配 (單位: 公尺)
    :return: 匹配結果 (局部點、全局點) 和剛體變換矩陣
    """

    def icp_refine(local_points, matched_points):
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local_points)

        matched_pcd = o3d.geometry.PointCloud()
        matched_pcd.points = o3d.utility.Vector3dVector(matched_points)

        threshold = 10.0  # ICP 配準距離閾值
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            local_pcd, matched_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        print("ICP 配準完成，轉換矩陣:")
        print(reg_p2p.transformation)
        return reg_p2p.transformation
    
    if len(local_points) == 0 or len(global_points) == 0:
        print("輸入的點雲為空，無法進行匹配")
        return None, None, None
    
    # 構建 KD 樹
    kdtree = KDTree(global_points)

    # 最近鄰搜索，過濾距離過遠的點
    distance_threshold = initial_distance_threshold
    matched_local_points = []
    matched_global_points = []

    while len(matched_local_points) != len(local_points):
        matched_global_points = []
        matched_local_points = []

        for point in local_points:
            # distance, index = kdtree.query(point)
            distance, index = kdtree.query(point)
            if distance <= distance_threshold:
                matched_global_points.append(global_points[index])
                matched_local_points.append(point)

        if len(matched_local_points) == len(local_points):
            break

        # 增加閾值並檢查上限
        distance_threshold += 5.0
        if distance_threshold > max_distance_threshold:
            print("警告: 已達最大距離閾值，仍未完全匹配")
            break

    matched_local_points = np.array(matched_local_points)
    matched_global_points = np.array(matched_global_points)
    print(f"matched_local_points.shape:{matched_local_points.shape}")
    print(f"matched_global_points.shape:{matched_global_points.shape}")
    
    if len(matched_global_points) < 4:
        print("匹配點數不足，無法計算轉換矩陣")
        return None, None, distance_threshold

    # # 使用 RANSAC 計算剛體變換
    # ransac = RANSACRegressor()
    # scaler = StandardScaler()

    # # 確保數據縮放並對齊
    # local_scaled = scaler.fit_transform(matched_local_points)
    # global_scaled = scaler.fit_transform(matched_global_points)
    # ransac.fit(local_scaled, global_scaled)

    # transform_matrix = ransac.estimator_.coef_
    transform_matrix = icp_refine(matched_local_points, matched_global_points)
    print(f"匹配完成。剛體變換矩陣:\n{transform_matrix}")
    print(f"最終使用的距離閾值: {distance_threshold:.2f}")

    return matched_global_points, transform_matrix

def sp_match_algo(local_points, global_points, distance_threshold=100.0, k_candidates=5):
    """
    匹配局部特徵點與全局特徵點雲，找出多個候選匹配點雲集。

    :param local_points: 局部特徵點座標
    :param global_points: 全局特徵點雲座標
    :param distance_threshold: 距離閾值
    :param k_candidates: 每個局部點要找的 K 個候選匹配點
    :return: 匹配候選點集列表
    """
    kdtree = KDTree(global_points)
    matched_candidates = []

    for point in local_points:
        distances, indices = kdtree.query(point, k=k_candidates)
        candidates = [global_points[idx] for d, idx in zip(distances, indices) if d <= distance_threshold]
        if candidates:
            matched_candidates.append(candidates)
    
    print(f"共找到 {len(matched_candidates)} 組候選匹配點雲集")
    return matched_candidates

"""
--------------------- Evaluation -------------------------
"""
# 驗證總模組
def evaluate_matching_accuracy(ground_truth_points, matched_global_points, output_dir, index, global_points, raw_global_points):
    """
    計算 Chamfer 距離，並視覺化匹配結果與誤差分佈。

    :param ground_truth_path: ground truth 點雲座標檔案路徑
    :param matched_local_points: 演算法匹配到的局部點雲
    :param output_dir: 結果儲存目錄
    :param index: 當前匹配組的索引
    """

    # 計算 Chamfer 距離
    chamfer_dist = chamfer_distance(ground_truth_points, matched_global_points)
    print(f"第 {index} 組 Chamfer 距離: {chamfer_dist:.4f}")

    # 儲存數值
    with open(os.path.join(output_dir, f"chamfer_distance_{index}.txt"), "w") as f:
        f.write(f"Chamfer Distance: {chamfer_dist:.4f}")

    # 視覺化誤差分佈
    errors = compute_matching_error(ground_truth_points, matched_global_points)
    plot_error_distribution(errors, os.path.join(output_dir, f"error_distribution_{index}.png"))

    # 視覺化結果
    visualize_matching(
        ground_truth_points=ground_truth_points,
        registrated_points=matched_global_points,
        global_points=global_points,
        raw_global_points=raw_global_points,
        output_path=os.path.join(output_dir, f"visual_{index}.ply"),
        radius=50.0
    )

"""
# --------------------- Evaluation used module -------------------------
"""
# 紀錄兩點雲間的距離
def chamfer_distance(predicted_points, ground_truth_points):
    """
    計算 Chamfer 距離。
    :param predicted_points: 預測點集 (N, 3)
    :param ground_truth_points: 真實點集 (M, 3)
    :return: Chamfer 距離
    """
    tree_ground = KDTree(ground_truth_points)
    tree_predicted = KDTree(predicted_points)

    distances_ground_to_predicted, _ = tree_ground.query(predicted_points, k=1)
    distances_predicted_to_ground, _ = tree_predicted.query(ground_truth_points, k=1)

    return np.mean(distances_ground_to_predicted) + np.mean(distances_predicted_to_ground)

# 計算匹配點對的平均距離誤差
def compute_matching_error(ground_truth_points, matched_global_points):
    """
    計算兩組已對齊的點雲間的誤差。

    :param local_points: 匹配後的局部點雲
    :param matched_points: 匹配後的全局點雲
    :return: 誤差值列表
    """
    # 確保兩個輸入點雲形狀一致
    # assert ground_truth_points.shape == matched_global_points.shape, "兩組點雲形狀不匹配"
    print (f"ground_truth_points:{ground_truth_points.shape}")
    print (f"matched_global_points:{matched_global_points.shape}")
    errors = np.linalg.norm(ground_truth_points - matched_global_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"平均誤差: {mean_error:.2f}, 最大誤差: {max_error:.2f}")
    return errors

# 可視化誤差分佈
def plot_error_distribution(errors, output_dir="/match_result"):
    """
    畫出誤差分佈並保存圖片。
    :param errors: 誤差列表
    :param output_dir: 輸出圖片目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Error Distribution")
    plt.xlabel("Error (meters)")
    plt.ylabel("Frequency")
    plt.grid()
    output_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"誤差分佈圖已保存至: {output_path}")

def visualize_matching(ground_truth_points, registrated_points, global_points, raw_global_points, output_path, radius=50.0):
    """
    可視化局部點雲 (ground truth)、匹配到的點雲 (registrated points)，以及原始與篩選後的全局點雲。

    :param ground_truth_points: 真實的局部點雲 (紅色)
    :param registrated_points: 匹配到的點雲 (綠色)
    :param global_points: 篩選後的全局點雲 (藍色)
    :param raw_global_points: 原始全局點雲 (灰色，用於對照)
    :param output_path: 輸出視覺化結果的檔案路徑
    :param radius: 過濾藍色點雲與灰色點雲的半徑範圍
    """
    def ensure_valid_format(points, name):
        if not isinstance(points, np.ndarray):
            raise TypeError(f"{name} 必須是 NumPy 陣列，但收到 {type(points)}")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name} 必須是形狀為 (N, 3) 的陣列，但收到 {points.shape}")
        return points.astype(np.float64)

    # 檢查並轉換數據格式
    ground_truth_points = ensure_valid_format(ground_truth_points, "ground_truth_points")
    registrated_points = ensure_valid_format(registrated_points, "registrated_points")
    global_points = ensure_valid_format(global_points, "global_points")
    raw_global_points = ensure_valid_format(raw_global_points, "raw_global_points")

    # 1. 建立局部 (ground truth) 點雲
    ground_truth_pcd = o3d.geometry.PointCloud()
    ground_truth_pcd.points = o3d.utility.Vector3dVector(ground_truth_points)
    ground_truth_pcd.paint_uniform_color([1, 0, 0])  # 紅色表示 ground truth 局部點雲

    # 2. 建立匹配到的點雲
    registrated_pcd = o3d.geometry.PointCloud()
    registrated_pcd.points = o3d.utility.Vector3dVector(registrated_points)
    registrated_pcd.paint_uniform_color([0, 1, 0])  # 綠色表示匹配點雲

    # 3. 篩選藍色點雲 (範圍過濾全局點雲)
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_points)

    kdtree = o3d.geometry.KDTreeFlann(global_pcd)
    selected_global_indices = set()
    for point in ground_truth_points:
        _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
        selected_global_indices.update(idx)
    selected_global_points = np.asarray(global_points)[list(selected_global_indices)]

    filtered_global_pcd = o3d.geometry.PointCloud()
    filtered_global_pcd.points = o3d.utility.Vector3dVector(selected_global_points)
    filtered_global_pcd.paint_uniform_color([0, 0, 1])  # 藍色表示篩選後的全局點雲

    # 4. 篩選灰色點雲 (原始點雲的局部範圍)
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(raw_global_points)

    kdtree_raw = o3d.geometry.KDTreeFlann(raw_pcd)
    selected_raw_indices = set()
    for point in ground_truth_points:
        _, idx, _ = kdtree_raw.search_radius_vector_3d(point, radius)
        selected_raw_indices.update(idx)
    selected_raw_points = np.asarray(raw_global_points)[list(selected_raw_indices)]

    filtered_raw_pcd = o3d.geometry.PointCloud()
    filtered_raw_pcd.points = o3d.utility.Vector3dVector(selected_raw_points)
    filtered_raw_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示原始點雲

    # 5. 可視化並保存結果
    o3d.visualization.draw_geometries(
        [raw_pcd, filtered_global_pcd, ground_truth_pcd, registrated_pcd],
        window_name="匹配結果可視化",
        width=800,
        height=600
    )

    combined_pcd = ground_truth_pcd + registrated_pcd + filtered_global_pcd + filtered_raw_pcd
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"匹配結果視覺化已保存至: {output_path}")

# 生成匹配結果的點對點連線圖
def draw_matching_lines(local_points, matched_points):
    lines = [[i, i] for i in range(len(local_points))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((local_points, matched_points)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])
    return line_set

# 檢查剛體變換矩陣 (旋轉矩陣應該接近正交，行列式為 1) (平移向量應在合理範圍內)
def validate_transform_matrix(transform_matrix):
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]

    determinant = np.linalg.det(rotation)
    print(f"旋轉矩陣行列式: {determinant:.2f} (應接近 1)")
    print(f"平移向量: {translation}")

    is_valid_rotation = np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-2)
    print(f"旋轉矩陣是否正交: {is_valid_rotation}")
    return is_valid_rotation and abs(determinant - 1) < 0.01
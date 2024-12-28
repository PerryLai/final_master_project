import os 
import cv2
import numpy as np
import time
from tqdm import tqdm
import open3d as o3d


def clear_existing_actors(world):
    """
    清除所有已存在的車輛和行人。
    """
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.*')
    controllers = actors.filter('controller.*')

    for vehicle in vehicles:
        vehicle.destroy()
    for walker in walkers:
        walker.destroy()
    for controller in controllers:
        controller.destroy()

    print(f"清理完成：共移除 {len(vehicles)} 輛車輛，{len(walkers)} 名行人，及 {len(controllers)} 個控制器。")

def save_pointcloud_to_ply(points, filename):
    """
    保存點雲到 .ply 文件。
    """
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"點雲已保存至 {filename}")

def attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth'):
    """
    附加相機到車輛，支援 RGB 或深度相機。
    """
    try:
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", "1920")
        camera_bp.set_attribute("image_size_y", "1080")
        camera_bp.set_attribute("fov", "90")
        if camera_type == 'sensor.camera.depth':
            camera_bp.set_attribute("PostProcessing", "Depth")  # 確保使用深度後處理模式
        transform = carla.Transform(location=carla.Location(x=2.5, z=1.5))
        # 確保 transform 和 vehicle 合法
        assert camera_bp is not None, "Camera blueprint 不存在！"
        assert vehicle is not None, "Vehicle 不存在！"
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        return camera
    except RuntimeError as e:
        print(f"附加 {camera_type} 失敗：{e}")
        return None

def attach_lidar_to_vehicle(world, vehicle, lidar_range=50.0, channels=32, points_per_second=100000, rotation_frequency=10):
    """
    在指定車輛上附加 LiDAR 感測器。
    """
    print("Attach lidar start!")
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(lidar_range))
    lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('points_per_second', str(points_per_second))
    lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))

    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))  # LiDAR 放在車頂
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    print("Attach lidar success!")
    return lidar

def get_camera_intrinsic(camera):
    """
    獲取相機的內參矩陣。
    """
    fov = float(camera.attributes['fov'])
    width = int(camera.attributes['image_size_x'])
    height = int(camera.attributes['image_size_y'])
    focal_length = width / (2.0 * np.tan(np.radians(fov / 2.0)))
    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

def detect_feature_points(image):
    """
    使用 ORB（Oriented FAST and Rotated BRIEF）檢測影像中的特徵點。

    Args:
        image: 用於特徵檢測的輸入影像。

    Returns:
        以 (x, y) 元組形式返回 2D 特徵點列表。
    """
    # print("--------檢測特徵點------------")
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)
    return [kp.pt for kp in keypoints]

def estimate_camera_pose_2d_3d(image_points, world_points):
    """
    使用 2D-3D 對應關係估計相機的姿態（旋轉和平移向量）。

    Args:
        image_points (np.ndarray): 圖像中檢測到的 2D 特徵點。
        world_points (np.ndarray): 對應的 3D 世界座標點。

    Returns:
        rvec (np.ndarray): 旋轉向量。
        tvec (np.ndarray): 平移向量。
    """
    assert len(image_points) >= 4, "相機姿態估計需要至少 4 個圖像點。"
    assert len(world_points) >= 4, "相機姿態估計需要至少 4 個世界座標點。"
    
    # 轉換為所需的格式，並取整數位
    image_points = np.array([[int(x), int(y)] for x, y in image_points], dtype=np.float32)
    world_points = np.array([[int(x), int(y), int(z)] for x, y, z in world_points], dtype=np.float32)
    
    # 相機內參矩陣（根據設置進行修改）
    camera_matrix = np.array([
        [1000, 0, 640],  # fx, 0, cx
        [0, 1000, 360],  # 0, fy, cy
        [0, 0, 1]        # 0, 0, 1
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # 假設沒有鏡頭畸變

    # 使用 EPnP 演算法求解 PnP 問題
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=world_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    print(f"--------------------成功估計相機姿態: {success}---------------------")
    if not success:
        raise RuntimeError("---------------EPnP 未能成功估計相機姿態。----------------------")
    return rvec, tvec

def save_ply_file(depth_image, intrinsic, transform, ply_filename):
    """
    從深度影像生成 .ply 文件。

    Args:
        depth_image: CARLA 深度影像。
        intrinsic: 相機內參矩陣。
        transform: 相機外部變換矩陣。
        ply_filename: 保存 .ply 文件的路徑。
    """
    depth_array = np.array(depth_image.raw_data, dtype=np.float32).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    depth_array = depth_array * 1000.0 / 255.0  # 轉換為公尺
    points = []

    height, width = depth_array.shape
    for y in range(height):
        for x in range(width):
            depth = depth_array[y, x]
            if depth <= 0:
                continue  # 跳過無效深度點
            pixel_coords = np.array([x, y, 1.0])
            camera_coords = np.dot(np.linalg.inv(intrinsic), pixel_coords) * depth
            world_coords = transform.transform(carla.Location(
                x=int(camera_coords[0]),  # 取整數位
                y=int(camera_coords[1]),  # 取整數位
                z=int(camera_coords[2])   # 取整數位
            ))
            points.append(f"{world_coords.x} {world_coords.y} {world_coords.z} 255 255 255\n")

    with open(ply_filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        f.writelines(points)

    print(f"點雲數據已保存為 .ply 文件：{ply_filename}")

def validate_epnp(vehicle, matched_2d_points, matched_3d_points, timestamp, intrinsic, experiment_id, t, vehicle_id):
    """
    驗證 EPnP 演算法是否正確，並統一世界坐標系（單位設定為米）。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print(f"Timestamp: {timestamp}")
    print(f"Experiment ID: {experiment_id}, Vehicle ID: {vehicle_id}")
    print(f"Matched 2D Points (pixels):\n{matched_2d_points}")
    print(f"Matched 3D Points (meters):\n{matched_3d_points}")
    print(f"Intrinsic Matrix (pixels):\n{intrinsic}\n")

    # 內參轉換至米單位
    intrinsic_in_meters = intrinsic.copy()
    pixel_to_meter_scale = 1.0 / intrinsic[0, 0]
    intrinsic_in_meters[:2, 2] *= pixel_to_meter_scale
    intrinsic_in_meters[:2, :3] *= pixel_to_meter_scale
    print(f"Pixel-to-Meter Scale: {pixel_to_meter_scale}")
    print(f"Intrinsic Matrix (meters):\n{intrinsic_in_meters}\n")

    # 過濾與正規化 3D 點
    world_points = np.array(matched_3d_points, dtype=np.float32)
    image_points = np.array(matched_2d_points, dtype=np.float32)

    if np.max(np.abs(world_points)) > 100:  # 假設超過 100 為非米單位
        print("警告：world_points 單位可能不是米，正在進行轉換...")
        world_points /= 100.0
        print("已將 world_points 單位從厘米轉為米")

    print(f"Filtered 2D Points (N x 2):\n{image_points}")
    print(f"Filtered 3D Points (N x 3):\n{world_points}\n")

    # 畫出 2D 點與 3D 點分佈
    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], label="Image Points", alpha=0.7)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.title("2D Points Distribution")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], label="World Points", alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.legend()
    plt.title("3D Points Distribution")
    plt.show()

    # 當前車輛的真實位置與方向
    location = vehicle.get_location()
    rotation = vehicle.get_transform().rotation
    vehicle_position = np.array([location.x, location.y, location.z])
    print(f"Vehicle Position (ground truth): {tuple(round(x, 3) for x in vehicle_position)}\n")

    try:
        # 計算 EPnP
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=world_points,
            imagePoints=image_points,
            cameraMatrix=intrinsic_in_meters,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise ValueError("EPnP 解算失敗")

        projected_points, _ = cv2.projectPoints(world_points, rvec, tvec, intrinsic_in_meters, distCoeffs=None)
        plt.scatter(image_points[:, 0], image_points[:, 1], label="Original Image Points", alpha=0.7)
        plt.scatter(projected_points[:, 0, 0], projected_points[:, 0, 1], label="Projected Points", alpha=0.7)
        plt.legend()
        plt.title("2D Points and Projected Points")
        plt.show()

        R_cam_to_world, _ = cv2.Rodrigues(rvec)
        t_cam_to_world = tvec.ravel()
        print(f"Rotation Matrix (R_cam_to_world):\n{R_cam_to_world}")
        print(f"Translation Vector (t_cam_to_world): {t_cam_to_world}\n")

        epnp_position_camera = -np.dot(R_cam_to_world.T, t_cam_to_world)
        epnp_position_world = np.dot(R_cam_to_world, epnp_position_camera) + t_cam_to_world
        position_error = np.linalg.norm(epnp_position_world - vehicle_position)
        print(f"EPnP Position (World Coordinates): {tuple(round(x, 3) for x in epnp_position_world)}")
        print(f"EPnP Position Error: {position_error:.3f} meters\n")

    except Exception as e:
        print(f"EPnP Verification Failed: {e}")

def capture_lidar_pointcloud(lidar, output_ply_path, timeout=5):
    """
    使用 LiDAR 感測器生成點雲並保存為 PLY 文件，添加進度條顯示和超時邏輯。
    """
    lidar_data = None

    def lidar_callback(data):
        nonlocal lidar_data
        lidar_data = data

    # 開始監聽 LiDAR 數據
    lidar.listen(lidar_callback)

    try:
        print("等待 LiDAR 數據...")
        start_time = time.time()

        # 等待 LiDAR 數據的非阻塞處理
        while lidar_data is None:
            if time.time() - start_time > timeout:
                print(f"start_time: {start_time}")
                print(f"time.time(): {time.time()}")
                print("等待 LiDAR 數據超時。")
                lidar.stop()
                return
            time.sleep(0.1)

        # 提取 LiDAR 數據大小
        print("LiDAR 數據捕獲完成，開始解析...")
        raw_data = lidar_data.raw_data
        total_points = len(raw_data) // (4 * 4)  # 每點由 4 個 float32 組成
        points = []

        print(f"解析 {total_points} 個點的 LiDAR 數據...")

        # 使用進度條逐步處理數據
        for i in tqdm(range(0, len(raw_data), 4 * 4), desc="Processing LiDAR Data"):
            chunk = raw_data[i:i + 4 * 4]
            if len(chunk) < 16:  # 確保分段長度足夠
                break
            point = np.frombuffer(chunk, dtype=np.float32)[:3]
            points.append(point)

        # 保存為 PLY 文件
        with open(output_ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"LiDAR 點雲已保存至 {output_ply_path}")

    finally:
        lidar.stop()

def attach_cameras_to_vehicles(world, vehicles):
    """
    為每輛車輛附加 RGB 和深度相機。
    """
    cameras = {}
    for vehicle, vehicle_id in vehicles:
        # 附加 RGB 相機
        rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
        depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

        cameras[vehicle_id] = {
            "rgb_camera": rgb_camera,
            "depth_camera": depth_camera,
            "rgb_data": [],
            "depth_data": []
        }

        # 定義數據捕捉回調
        def save_rgb(image, vehicle_id=vehicle_id):
            cameras[vehicle_id]["rgb_data"].append(image)

        def save_depth(image, vehicle_id=vehicle_id):
            cameras[vehicle_id]["depth_data"].append(image)

        # 註冊監聽器
        rgb_camera.listen(save_rgb)
        depth_camera.listen(save_depth)

    return cameras

def clean_up_cameras(cameras):
    """
    銷毀所有附加的感測器。
    """
    for camera_data in cameras.values():
        if camera_data["rgb_camera"].is_alive:
            camera_data["rgb_camera"].destroy()
        if camera_data["depth_camera"].is_alive:
            camera_data["depth_camera"].destroy()

def save_pointcloud_to_ply(points, ply_filename):
    """
    保存點雲到 .ply 文件。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"點雲已保存到 {ply_filename}")

def record_vehicle_states(vehicles):
    """
    記錄所有車輛的當前位置和速度。
    """
    vehicle_states = {}
    for vehicle, vehicle_id in vehicles:
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        vehicle_states[vehicle_id] = {
            'location': location,
            'velocity': velocity
        }
    return vehicle_states

def restore_vehicle_states(vehicles, vehicle_states, delta_time=0.1):
    """
    將車輛恢復到上一次的狀態，模擬倒退 delta_time 秒。
    """
    for vehicle, vehicle_id in vehicles:
        if vehicle_id in vehicle_states:
            state = vehicle_states[vehicle_id]
            location = state['location']
            velocity = state['velocity']

            # 模擬倒退位置
            backward_location = carla.Location(
                x=location.x - velocity.x * delta_time,
                y=location.y - velocity.y * delta_time,
                z=location.z - velocity.z * delta_time
            )

            # 更新車輛位置
            transform = vehicle.get_transform()
            transform.location = backward_location
            vehicle.set_transform(transform)

            # 更新車輛速度（保持靜止）
            vehicle.set_velocity(carla.Vector3D(0, 0, 0))

def adjust_vehicle_to_ground_level(world, vehicle):
    """
    調整車輛高度，確保其與地面水平。
    """
    location = vehicle.get_location()
    waypoint = world.get_map().get_waypoint(location)
    if waypoint:
        transform = vehicle.get_transform()
        transform.location.z = waypoint.transform.location.z + 0.5  # 加一點高度偏移
        vehicle.set_transform(transform)

def capture_images_for_all_vehicles(world, cameras, timestamp, output_dir):
    """
    觸發所有車輛的感測器捕捉數據，並保存影像。
    """
    # 推進模擬步驟，觸發感測器
    world.tick()

    # 保存影像數據
    for vehicle_id, camera_data in cameras.items():
        rgb_images = camera_data["rgb_data"]
        depth_images = camera_data["depth_data"]

        if rgb_images and depth_images:
            # 保存 RGB 影像
            rgb_image = rgb_images.pop(0)
            rgb_path = f"{output_dir}/{vehicle_id}_rgb_{timestamp}.png"
            rgb_image.save_to_disk(rgb_path)
            print(f"RGB 影像已保存至 {rgb_path}")

            # 保存深度影像
            depth_image = depth_images.pop(0)
            depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
            depth_array = depth_array.reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
            depth_path = f"{output_dir}/{vehicle_id}_depth_{timestamp}.png"
            cv2.imwrite(depth_path, depth_array)
            print(f"深度影像已保存至 {depth_path}")
        else:
            print(f"車輛 {vehicle_id} 未捕捉到影像。")

def save_actor_locations(t, vehicles, walkers, output_dir):
    """
    保存當前所有車輛和行人的座標到檔案。
    """
    vehicle_file = os.path.join(output_dir, "vehicle_location", f"{t}_ms.txt")
    walker_file = os.path.join(output_dir, "walkers_location", f"{t}_ms.txt")

    os.makedirs(os.path.dirname(vehicle_file), exist_ok=True)
    os.makedirs(os.path.dirname(walker_file), exist_ok=True)

    # 保存車輛座標
    with open(vehicle_file, "w") as vf:
        for vehicle, vehicle_id in vehicles:
            location = vehicle.get_location()
            vf.write(f"{vehicle_id}: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})\n")

    # 保存行人座標
    with open(walker_file, "w") as wf:
        for walker, _, walker_id in walkers:
            location = walker.get_location()
            wf.write(f"{walker_id}: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})\n")

    print(f"時間 {t} 毫秒的車輛與行人座標已保存。")

def smoothly_stop_vehicle(vehicle):
    """
    平穩停止車輛。
    """
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

def check_and_correct_vehicle_orientation(vehicle):
    """
    檢查車輛的旋轉角度，過大時進行糾正。
    """
    transform = vehicle.get_transform()
    if abs(transform.rotation.pitch) > 20 or abs(transform.rotation.roll) > 20:
        print(f"車輛 {vehicle.id} 的角度異常，正在糾正...")
        transform.rotation.pitch = 0
        transform.rotation.roll = 0
        vehicle.set_transform(transform)

def capture_images(world, vehicle, dirname, timestamp):
    """
    捕捉 RGB 和深度影像，確保其他車輛和行人在拍攝期間保持靜止。
    """
    # 附加相機
    rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
    depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

    if not rgb_camera or not depth_camera:
        print(f"跳過 {dirname.split('/')[-1]} 在時間 {timestamp} 的捕捉：未附加感測器。")
        if rgb_camera:
            rgb_camera.destroy()
        if depth_camera:
            depth_camera.destroy()
        return

    # 捕捉數據的容器
    rgb_image_data = []
    depth_image_data = []

    def save_rgb_image(image):
        rgb_image_data.append(image)

    def save_depth_image(image):
        depth_image_data.append(image)

    rgb_camera.listen(save_rgb_image)
    depth_camera.listen(save_depth_image)

    # 使用 world.tick() 進行感測器觸發
    for _ in range(10):  # 最多等待 10 次 tick
        world.tick()
        if rgb_image_data and depth_image_data:
            break

    # 停止監聽感測器
    rgb_camera.stop()
    depth_camera.stop()

    if not (rgb_image_data and depth_image_data):
        print(f"未能捕捉到 {dirname.split('/')[-1]} 在時間 {timestamp} 的影像。")
    else:
        # 保存 RGB 影像
        rgb_image = rgb_image_data[0]
        rgb_path = f"{dirname}/rgb_{timestamp}.png"
        rgb_image.save_to_disk(rgb_path)
        print(f"RGB 影像已保存至 {rgb_path}")

        # 保存深度影像
        depth_image = depth_image_data[0]
        depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth_array = depth_array.reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
        depth_path = f"{dirname}/depth_{timestamp}.png"
        cv2.imwrite(depth_path, depth_array)
        print(f"深度影像已保存至 {depth_path}")

    # 銷毀感測器
    if rgb_camera.is_alive:
        rgb_camera.destroy()
    if depth_camera.is_alive:
        depth_camera.destroy()

def scale_and_convert_to_int(ply_path, output_ply_path):
    """
    將點雲中的座標乘以 10 並轉換為整數，然後保存為新的 .ply 文件。

    :param ply_path: 輸入點雲檔案 (.ply) 路徑
    :param output_ply_path: 輸出點雲檔案 (.ply) 路徑
    """
    # 讀取點雲
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # 將座標乘以 10 並轉換為整數
    scaled_points = (points * 10).astype(int)

    # 更新點雲並保存
    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
    o3d.io.write_point_cloud(output_ply_path, scaled_pcd)

    print(f"已將座標縮放並保存至 {output_ply_path}")

def calculate_top_10_percent_z(ply_file):
    """
    計算點雲中 z 座標位於從大到小第 10% 的臨界值。
    :param ply_file: 點雲文件路徑
    :return: 第 10% 的 z 座標值
    """
    # 讀取點雲
    point_cloud = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(point_cloud.points)

    # 確保數據存在
    if points.shape[0] == 0:
        raise ValueError("點雲文件中沒有有效的數據。")

    # 提取 z 座標
    z_coords = points[:, 2]

    # 計算第 10% 的 z 值
    top_10_percent_z = np.percentile(z_coords, 90)  # 90% 百分位
    return top_10_percent_z

def analyze_point_cloud_range(ply_path):
    """
    分析點雲的範圍值，包括 x, y, z 的最小值與最大值。
    :param ply_path: 點雲文件的路徑 (.ply)
    """
    # 讀取點雲
    point_cloud = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(point_cloud.points)

    if points.size == 0:
        print("點雲文件中沒有數據！")
        return

    # 計算範圍
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    print(f"x 範圍: {x_min:.2f} 到 {x_max:.2f}")
    print(f"y 範圍: {y_min:.2f} 到 {y_max:.2f}")
    print(f"z 範圍: {z_min:.2f} 到 {z_max:.2f}")

def extract_roof_features_with_image(rgb_path, depth_path, marked_image_path, sky_threshold=(180, 180, 240),
                                     edge_margin=20, depth_threshold=50, smooth_kernel_size=15):
    """
    改進版：使用雙向水平掃描檢測建築物與天空交界線，提取有效特徵點並整合深度資訊。
    """
    # 讀取影像
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb_image is None or depth_image is None:
        print(f"無法讀取影像: {rgb_path} 或 {depth_path}")
        return []

    height, width = rgb_image.shape[:2]

    # 天空過濾
    lower_sky = np.array([sky_threshold[0], sky_threshold[0], sky_threshold[1]])
    upper_sky = np.array([255, 255, 255])
    sky_mask = cv2.inRange(rgb_image, lower_sky, upper_sky)
    non_sky_mask = cv2.bitwise_not(sky_mask)
    cv2.imwrite("debug_sky_mask.png", sky_mask)

    # 雙向掃描
    edge_line_top = np.full(width, height, dtype=int)
    for col in range(width):
        non_zero_indices = np.where(non_sky_mask[:, col] > 0)[0]
        if len(non_zero_indices) > 0:
            edge_line_top[col] = non_zero_indices[0]

    # 平滑邊界
    edge_line_top_smoothed = cv2.GaussianBlur(edge_line_top.reshape(-1, 1).astype(np.float32),
                                              (smooth_kernel_size, 1), 0).flatten().astype(int)

    # 生成邊界遮罩
    boundary_mask = np.zeros_like(sky_mask, dtype=np.uint8)
    for col in range(width):
        top = max(0, edge_line_top_smoothed[col] - edge_margin)
        bottom = min(height, edge_line_top_smoothed[col] + edge_margin)
        boundary_mask[top:bottom, col] = 255
    cv2.imwrite("debug_boundary_mask.png", boundary_mask)

    # 提取特徵點
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray_image, mask=boundary_mask)
    pixel_coords_with_depth = [(int(kp.pt[0]), int(kp.pt[1]), depth_image[int(kp.pt[1]), int(kp.pt[0])])
                               for kp in keypoints if depth_image[int(kp.pt[1]), int(kp.pt[0])] > depth_threshold]

    # 繪製標記
    marked_image = rgb_image.copy()
    for col in range(width):
        cv2.circle(marked_image, (col, edge_line_top_smoothed[col]), 1, (0, 0, 255), -1)
    for x, y, _ in pixel_coords_with_depth:
        cv2.circle(marked_image, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite(marked_image_path, marked_image)
    print(f"標記影像已保存: {marked_image_path}")
    print(f"有效特徵點數量: {len(pixel_coords_with_depth)}")

    return pixel_coords_with_depth

def extract_roof_features_from_point_cloud(pcd_path, output_dir, height_threshold=10.0):
    """
    從3D點雲中提取建築物屋頂的特徵點。
    :param pcd_path: 點雲文件路徑 (.ply)
    :param output_dir: 輸出文件夾路徑
    :param height_threshold: 高度閾值
    :return: 特徵點3D座標 (list of tuples), 屋頂點雲文件路徑 (.ply)
    """
    # 讀取點雲
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 過濾高度
    roof_points = points[points[:, 2] > height_threshold]
    roof_pcd = o3d.geometry.PointCloud()
    roof_pcd.points = o3d.utility.Vector3dVector(roof_points)

    # 保存屋頂點雲
    roof_ply_path = os.path.join(output_dir, "roof_point_cloud.ply")
    o3d.io.write_point_cloud(roof_ply_path, roof_pcd)

    return roof_points, roof_ply_path

def spherical_projection(points):
    """
    將 3D 點雲投影到球面坐標系。

    :param points: 3D 點雲座標 (N, 3)
    :return: 投影點的 2D 球面坐標 (azimuth, elevation)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)  # 方位角
    elevation = np.arcsin(z / r)  # 高度角

    return np.column_stack((azimuth, elevation))

def project_and_match(local_points, global_points):
    """
    使用球面投影進行 2D 匹配。

    :param local_points: 局部點雲
    :param global_points: 全局點雲
    :return: 匹配點對
    """
    local_proj = spherical_projection(local_points)
    global_proj = spherical_projection(global_points)

    # 使用 KDTree 在 2D 投影空間進行匹配
    kdtree = KDTree(global_proj)
    matched_indices = []
    for proj in local_proj:
        _, idx = kdtree.query(proj)
        matched_indices.append(global_points[idx])

    matched_points = np.array(matched_indices)
    return matched_points

def visualize_point_cloud(ply_path):
    """
    讀取並顯示 3D 點雲檔案 (.ply)。

    :param ply_path: 輸入的點雲檔案路徑 (.ply)
    """
    # 讀取點雲
    print("正在讀取點雲檔案...")
    pcd = o3d.io.read_point_cloud(ply_path)

    # 檢查點雲資訊
    print("點雲資訊:")
    print(pcd)
    print(f"點雲包含 {len(pcd.points)} 個點")

    # 可視化點雲
    print("顯示點雲...")
    o3d.visualization.draw_geometries([pcd], window_name="3D 點雲視覺化", width=800, height=600)

    # 保存點雲圖片（選擇性功能）
    save_image = input("是否保存點雲視覺化結果為圖片？ (y/n): ").strip().lower()
    if save_image == 'y':
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 開啟可視化視窗
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("point_cloud_visualization.png")
        vis.destroy_window()
        print("點雲視覺化圖片已保存為: point_cloud_visualization.png")
    












































import os
import random
import time
import numpy as np
import carla
import sys
import cv2
from util import clear_existing_actors, clean_up_cameras, save_pointcloud_to_ply, attach_camera_to_vehicle
from util import attach_cameras_to_vehicles, get_camera_intrinsic, validate_epnp, capture_images
from util import record_vehicle_states, restore_vehicle_states, adjust_vehicle_to_ground_level, smoothly_stop_vehicle, check_and_correct_vehicle_orientation
from util import capture_images_for_all_vehicles, save_actor_locations
from tqdm import tqdm
import matplotlib.pyplot as plt

# 在衛星上設置相機
def setup_camera(world, location, target):
    """Setup a camera sensor pointing accurately towards the city center."""
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '90')

    # Calculate direction vector
    direction = carla.Location(
        x=target.x - location.x,
        y=target.y - location.y,
        z=target.z - location.z
    )

    # Calculate rotation
    pitch = np.degrees(np.arctan2(direction.z, np.sqrt(direction.x**2 + direction.y**2)))
    yaw = np.degrees(np.arctan2(direction.y, direction.x))
    roll = 0
    rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

    # Log details
    print(f"Camera setup: Location=({int(location.x)}, {int(location.y)}, {int(location.z)}), "
          f"Target=({int(target.x)}, {int(target.y)}, {int(target.z)}), "
          f"Rotation=({rotation.pitch:.2f}, {rotation.yaw:.2f}, {rotation.roll:.2f})")
    print(f"Direction Vector: x={direction.x:.2f}, y={direction.y:.2f}, z={direction.z:.2f}")

    # Set camera transform
    camera_transform = carla.Transform(location, rotation)
    camera = world.spawn_actor(camera_bp, camera_transform)
    return camera

# 衛星拍攝視角模擬（每十度拍攝一次）
def satellite_capture_images(client, radius, angle_increment, output_dir):
    """Capture satellite images with correct target alignment."""
    world = client.get_world()

    # City center location
    city_center = carla.Location(x=90, y=0, z=0) #Town01
    # city_center = carla.Location(x=2500, y=4600, z=370) #Town12

    # Generate satellite motion trajectory
    angles = np.arange(45, 135, angle_increment)
    for i, angle in enumerate(angles):
        # Compute camera position
        x = radius * np.cos(np.radians(angle))
        y = 0  # Fixed on XZ plane
        z = radius * np.sin(np.radians(angle))
        location = carla.Location(x=x, y=y, z=z)

        print(f"Angle {int(angle)}: ")

        # Setup the camera pointing at the city center
        camera = setup_camera(world, location, city_center)

        # Save image
        def save_image(image, idx=i+18):
            output_path = f"{output_dir}/satellite_{idx:03d}.png"
            # print(f"Saving image {idx} to {output_path}")
            image.save_to_disk(output_path)

        camera.listen(lambda image: save_image(image))

        # Pause to ensure image capture completes
        time.sleep(1)
        world.tick()

        # Stop and destroy camera
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
                print("")
        except Exception as e:
            print(f"Error during camera cleanup at angle {int(angle)}: {e}")

# 連線carla後設置行人與車子，並設定相機拍攝影像
def generate_simulation_data(world, experiment_id, dirname, duration, vehicle_num, walker_num, select_num):

    # 設置同步模擬模式
    sys.stderr = open(os.devnull, 'w')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    sys.stderr = sys.__stderr__

    # 生成車輛和行人數據，並記錄它們的數據。
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    walker_bp_library = world.get_blueprint_library().filter('*walker*')
    spawn_points = world.get_map().get_spawn_points()

    # 建立目錄
    vehicle_dir = os.path.join(dirname, "vehicle")
    walker_dir = os.path.join(dirname, "walkers")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(walker_dir, exist_ok=True)

    destroyed_vehicles = set()
    destroyed_walkers = set()

    vehicles, walkers = [], []

    # 清除已存在的角色
    clear_existing_actors(world)

    # 生成所有車輛
    print("開始生成車輛...")
    for j in range(vehicle_num):  # 總共生成 vehicle_num 輛車
        if f"vehicle_{j}" in destroyed_vehicles:
            continue
        for attempt in range(10):  # Carla有點容易出錯，所以給他最多重試 10 次的機會
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(vehicle_bp_library)
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                vehicle_id = f"vehicle_{j}"
                vehicles.append((vehicle, vehicle_id))
                os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                print(f"車輛 {vehicle_id} 成功生成於實驗 {experiment_id} 中。")
                break
            else:
                print(f"嘗試第 {attempt + 1} 次生成車輛 {j} 失敗，重試中...")
                time.sleep(2)

        # 生成所有行人
    
    print("開始生成行人...")
    for j in range(walker_num):  # 總共生成 200 名行人
        if f"walker_{j}" in destroyed_walkers:
            continue
        for attempt in range(3):
            spawn_point = world.get_random_location_from_navigation()
            if not spawn_point:
                print(f"行人 {j} 的生成點無效，跳過。")
                break
            walker_bp = random.choice(walker_bp_library)
            walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_point))
            if walker:
                try:
                    walker_control_bp = world.get_blueprint_library().find('controller.ai.walker')
                    walker_controller = world.spawn_actor(walker_control_bp, carla.Transform(), walker)
                    walker_speed = random.uniform(1.0, 3.0)
                    walker_controller.start()
                    walker_controller.set_max_speed(walker_speed)
                    walker_controller.go_to_location(world.get_random_location_from_navigation())
                    walker_id = f"walker_{j}"
                    walkers.append((walker, walker_controller, walker_id))
                    os.makedirs(f"{walker_dir}/{walker_id}", exist_ok=True)
                    print(f"行人 {walker_id} 生成，速度為 {walker_speed:.2f} 米/秒。")
                    break
                except RuntimeError as e:
                    print(f"初始化行人 {walker_id} 失敗：{e}")
                    if walker.is_alive:
                        walker.destroy()

    print(f"---------車輛生成完成：共生成 {len(vehicles)} 輛車輛。------------")
    print(f"---------行人生成完成：共生成 {len(walkers)} 名行人。------------")

    # 開始模擬並記錄數據
    print("---------開始模擬...---------")
    record_dynamic_objects(world, experiment_id, dirname, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, select_num)

    # 刪除空目錄
    print("---------開始刪除空目錄...---------")
    for root, dirs, files in os.walk(dirname, topdown=False):  # 自底向上遍歷
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):  # 如果目錄是空的
                os.rmdir(dir_path)
                # print(f"已刪除空目錄：{dir_path}")
    print("空目錄刪除完成！")

    # process_all_ply_files(dirname)

    # 恢復設置
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("模擬完成！")

# 開始模擬並記錄數據
def record_dynamic_objects(world, experiment_id, dirname, vehicles, walkers, vehicle_dir, walker_dir, duration, destroyed_vehicles, destroyed_walkers, select_num):
    """
    結合同步模式與完整功能的模擬與數據記錄函數。
    確保模擬中車輛和行人保持同步，每秒記錄數據並自動補充缺失對象。
    """
    vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    print("")
    print(f"---------實驗 {experiment_id} 開始---------")
    print("")

    # 隨機選取 5 台車輛進行檢測
    if len(vehicles) > select_num:
        selected_vehicles = random.sample(vehicles, select_num)  # 隨機選擇 5 台
    else:
        selected_vehicles = vehicles  # 如果車輛數量少於或等於 5，則全部檢測

    # test
    # cameras = attach_cameras_to_vehicles(world, vehicles)

    for t in range(int(duration * 10)):  # 持續模擬 `duration` 秒，每100ms紀錄一次
        
        print(f"---------模擬時間 {t * 100} 毫秒--------")

        # 同時捕捉所有車輛的影像
        capture_images_all_at_once(world, selected_vehicles, t, vehicle_dir)
        # 保存車輛與行人座標
        save_actor_locations(t * 100, vehicles, walkers, dirname)
        
        # 記錄車輛當前狀態
        world.tick()  # 同步模式下執行模擬步驟

        # 檢查車輛是否存活，若有損失則補充
        for i, (vehicle, vehicle_id) in enumerate(vehicles):
            if not vehicle.is_alive:
                print(f"車輛 {vehicle_id} 已不再存活，重新生成中...")
                destroyed_vehicles.add(vehicle_id)

                # 嘗試生成新車輛並替代損失車輛
                for attempt in range(10):
                    try:
                        spawn_point = random.choice(spawn_points)
                        blueprint = random.choice(vehicle_bp_library)
                        new_vehicle = world.try_spawn_actor(blueprint, spawn_point)
                        if new_vehicle:
                            new_vehicle.set_autopilot(True)
                            print(f"新的車輛 {vehicle_id} 已於生成點 {spawn_point} 生成。")
                            vehicles[i] = (new_vehicle, vehicle_id)
                            os.makedirs(f"{vehicle_dir}/{vehicle_id}", exist_ok=True)
                            break
                        else:
                            print(f"嘗試生成車輛 {vehicle_id} 失敗，重試中...")
                            time.sleep(2)
                    except RuntimeError as e:
                        print(f"生成車輛 {vehicle_id} 時發生錯誤：{e}")
        
        # 嘗試設置導航目標
        target_location = None
        
        # 檢查並更新行人導航
        # for walker, walker_controller, walker_id in walkers:
        #     if walker.is_alive:
        #         target_location = world.get_random_location_from_navigation()
        #         if target_location:
        #             walker_controller.go_to_location(target_location)
        #             print(f"行人 {walker_id} 正在導航到: {target_location}")

        # 捕捉車輛數據並記錄
        for vehicle, vehicle_id in selected_vehicles:
            if vehicle_id in destroyed_vehicles:
                continue
            try:
                if not vehicle.is_alive:
                    print(f"車輛 {vehicle_id} 已被摧毀，跳過處理。")
                    destroyed_vehicles.add(vehicle_id)
                    continue

                # 記錄車輛位置、速度與旋轉數據，取整數值
                location = vehicle.get_location()
                print("")
                with open(f"{vehicle_dir}/{vehicle_id}/info.txt", "a") as f:
                    f.write(f"時間 {t * 100} 毫秒: 位置: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})\n")
                with open(f"{vehicle_dir}/{vehicle_id}/info_int.txt", "a") as f:
                    f.write(f"時間 {t * 100} 毫秒: 位置: ({location.x:.0f}, {location.y:.0f}, {location.z:.0f})\n")

            except RuntimeError as e:
                print(f"捕捉車輛 {vehicle_id} 數據時發生錯誤：{e}")
                destroyed_vehicles.add(vehicle_id)

        # 捕捉行人數據並記錄
        # for walker, walker_controller, walker_id in walkers:
        #     if walker_id in destroyed_walkers:
        #         continue
        #     try:
        #         if not walker.is_alive:
        #             print(f"行人 {walker_id} 已被摧毀，跳過處理。")
        #             destroyed_walkers.add(walker_id)
        #             continue

        #         # 記錄行人位置數據，取整數值
        #         location = walker.get_location()
        #         with open(f"{walker_dir}/{walker_id}/info.txt", "a") as f:
        #             f.write(f"時間 {t * 100} 毫秒: 位置: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})\n")

        #     except RuntimeError as e:
        #         print(f"捕捉行人 {walker_id} 數據時發生錯誤：{e}")
        #         destroyed_walkers.add(walker_id)    
        
    print("所有時間步驟的模擬已完成。")

def capture_images_all_at_once(world, vehicles, timestamp, output_dir):
    """
    同時捕捉所有車輛的 RGB 和深度影像。
    """
    # 為所有車輛附加相機，並初始化數據容器
    cameras = {}
    for vehicle, vehicle_id in vehicles:
        # 附加 RGB 和深度相機
        rgb_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.rgb')
        depth_camera = attach_camera_to_vehicle(world, vehicle, camera_type='sensor.camera.depth')

        # 初始化數據存儲
        cameras[vehicle_id] = {
            "rgb_camera": rgb_camera,
            "depth_camera": depth_camera,
            "rgb_data": [],
            "depth_data": []
        }

        # 定義數據回調函數
        def save_rgb(image, vehicle_id=vehicle_id):
            cameras[vehicle_id]["rgb_data"].append(image)

        def save_depth(image, vehicle_id=vehicle_id):
            cameras[vehicle_id]["depth_data"].append(image)

        # 啟動監聽
        rgb_camera.listen(save_rgb)
        depth_camera.listen(save_depth)

    # 推進模擬步驟，觸發感測器數據
    for _ in range(20):  # 最多等待 20 次 tick
        world.tick()
        time.sleep(0.1)  # 適度等待數據生成
        # 檢查所有車輛是否捕捉到影像
        if all(len(camera["rgb_data"]) > 0 and len(camera["depth_data"]) > 0 for camera in cameras.values()):
            break

    # 保存影像數據並停止相機
    for vehicle_id, camera_data in cameras.items():
        rgb_images = camera_data["rgb_data"]
        depth_images = camera_data["depth_data"]

        if rgb_images and depth_images:
            # 保存 RGB 影像
            rgb_image = rgb_images.pop(0)
            rgb_path = f"{output_dir}/{vehicle_id}/rgb_{timestamp}.png"
            rgb_image.save_to_disk(rgb_path)
            print(f"車輛 {vehicle_id} 的 RGB 影像已保存至 {rgb_path}")

            # 保存深度影像
            depth_image = depth_images.pop(0)
            depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
            depth_array = depth_array.reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
            depth_path = f"{output_dir}/{vehicle_id}/depth_{timestamp}.png"
            cv2.imwrite(depth_path, depth_array)
            print(f"車輛 {vehicle_id} 的深度影像已保存至 {depth_path}")
        else:
            print(f"車輛 {vehicle_id} 未捕捉到影像。")

        # 停止相機監聽
        if camera_data["rgb_camera"].is_alive:
            camera_data["rgb_camera"].destroy()
        if camera_data["depth_camera"].is_alive:
            camera_data["depth_camera"].destroy()
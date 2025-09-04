import carla
import os
import time
import numpy as np
import cv2
import random

# ==================== 全局配置参数 ====================
FPS = 10                   # 模拟精度
SAVE_INTERVAL = 2          # 每2帧保存一次
IMAGE_RESOLUTION = (1280, 720)  # 720p分辨率
IMAGE_FORMAT = "jpg"       # 保存格式
JPEG_QUALITY = 90          # 提高JPG质量
BASE_DIR = ""
TRAFFIC_MANAGER_PORT = 8000
NUM_MAIN_VEHICLES = 20     # 增加20辆主车（每辆一个摄像头）
NUM_NPC_VEHICLES = 130     # 调整NPC车辆数量（总车辆数150）
NUM_PEDESTRIANS = 0        # 行人数量
# ======================================================

def set_rainy_weather(world):
    """ 设置白天雨天天气 """
    weather = carla.WeatherParameters(
        cloudiness=60.0,            # 云量
        precipitation=90.0,         # 降雨量
        precipitation_deposits=70.0,# 地面积水（形成水洼）
        wind_intensity=30.0,        # 风力
        sun_altitude_angle=65.0     # 太阳高度角（稍微降低，模拟阴天）
    )
    world.set_weather(weather)
    print("天气已设置为：雨天")

def save_image_cv(image, path, frame):
    """ 使用OpenCV保存图像（支持格式/质量控制） """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]  # 移除Alpha通道
    array = array[:, :, ::-1]  # RGB -> BGR
    
    params = []
    if IMAGE_FORMAT.lower() == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    
    filename = f"{path}/{frame:08d}.{IMAGE_FORMAT}"
    cv2.imwrite(filename, array, params)

class ImageSaver:
    """ 智能图像保存器，控制保存频率 """
    def __init__(self, path, save_interval=5):
        self.path = path
        self.save_interval = save_interval
        self.counter = 0
        
    def __call__(self, image):
        if self.counter % self.save_interval == 0:
            save_image_cv(image, self.path, image.frame)
        self.counter += 1

def get_valid_roadside_location(world, waypoint, distance=2.5, height=2.8):
    """
    获取道路边缘的有效位置
    :param world: CARLA世界对象
    :param waypoint: 参考路点
    :param distance: 距离道路中心的距离（正数表示右侧，负数表示左侧）
    :param height: 摄像头高度
    :return: 有效的道路边缘位置
    """
    # 获取道路边缘的路点
    if distance > 0:
        # 右侧边缘
        for _ in range(3):
            if waypoint.right_lane_marking.type == carla.LaneMarkingType.Solid:
                break
            waypoint = waypoint.get_right_lane()
            if not waypoint:
                break
    else:
        # 左侧边缘
        for _ in range(3):
            if waypoint.left_lane_marking.type == carla.LaneMarkingType.Solid:
                break
            waypoint = waypoint.get_left_lane()
            if not waypoint:
                break
    
    if not waypoint:
        return None
    
    # 获取道路边缘位置
    location = waypoint.transform.location
    
    # 沿着道路法线方向移动一定距离
    right_vec = waypoint.transform.get_right_vector()
    location.x += right_vec.x * abs(distance)
    location.y += right_vec.y * abs(distance)
    location.z = height  # 设置适当高度
    
    return location, waypoint

def spawn_npc_vehicles(world, traffic_manager, num_vehicles=10, excluded_spawn_points=[]):
    """ 生成并设置NPC车辆 """
    print(f"正在生成 {num_vehicles} 辆包含汽车和摩托车的NPC载具...")
    
    traffic_vehicles = []
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    
    # vehicle_blueprints = x for x in vehicle_blueprints 
    #                      if int(x.get_attribute('number_of_wheels')) == 4]
    
    # 生成指定数量的车辆
    i = 0
    while len(traffic_vehicles) < num_vehicles and i < len(spawn_points):
        # 跳过已排除的生成点
        if i in excluded_spawn_points:
            i += 1
            continue
            
        spawn_point = spawn_points[i]
        i += 1
        
        # 随机选择车辆类型
        bp = random.choice(vehicle_blueprints)
        
        # 设置车辆颜色
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # 设置不无敌（可以被碰撞）
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        
        # 尝试生成车辆
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is not None:
            # 设置为自动驾驶模式
            vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)
            
            # 设置随机速度（基于道路限速）
            try:
                current_waypoint = world.get_map().get_waypoint(vehicle.get_location())
                speed_limit = current_waypoint.lane_speed_limit
                target_speed = speed_limit * (0.8 + random.random() * 0.4)  # 80%-120%限速
                traffic_manager.set_desired_speed(vehicle, target_speed)
            except:
                traffic_manager.set_desired_speed(vehicle, 20.0)
            
            traffic_vehicles.append(vehicle)
        else:
            print(f"警告: 无法在生成点 {i} 生成车辆")
    
    print(f"成功生成 {len(traffic_vehicles)} 辆NPC载具")
    return traffic_vehicles

def get_vehicle_blueprint(blueprint_lib, vehicle_id):
    """
    安全获取车辆蓝图，如果指定模型不存在则使用替代模型
    """
    # 
    vehicle_models = [
        'vehicle.nissan.leaf',          
        'vehicle.audi.tt',
        'vehicle.bmw.grandtourer',
        'vehicle.chevrolet.impala',
        'vehicle.citroen.c3',
        'vehicle.dodge.charger_2020',
        'vehicle.jeep.wrangler_rubicon',
        'vehicle.lincoln.mkz_2017',
        'vehicle.mercedes-benz.coupe',
        'vehicle.mini.cooperst',
        'vehicle.nissan.micra',
        'vehicle.nissan.patrol',
        'vehicle.seat.leon',
        'vehicle.tesla.model3',         
        'vehicle.toyota.prius',
        'vehicle.volkswagen.t2'
    ]
    
    # 尝试获取指定模型
    model = vehicle_models[vehicle_id % len(vehicle_models)]
    vehicles = blueprint_lib.filter(model)
    if len(vehicles) > 0:
        return vehicles[0]
    
    # 如果指定模型不存在，使用随机车辆 (只选四轮车作为主车)
    four_wheeled_vehicles = [
        x for x in blueprint_lib.filter('vehicle.*') 
        if int(x.get_attribute('number_of_wheels')) == 4
    ]
    
    if four_wheeled_vehicles:
        return random.choice(four_wheeled_vehicles)
    
    # 最后手段：使用model3
    return blueprint_lib.filter('vehicle.tesla.model3')[0]

def main():
    # 初始化变量
    cameras = []
    main_vehicles = []  # 现在有20辆主车
    traffic_manager = None
    traffic_vehicles = []
    frame_count = 0

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        
        # 切换到Town03地图
        print("正在加载Town03地图...")
        client.load_world('Town03')
        time.sleep(3)  # 等待地图加载完成
        
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / FPS  # 精确匹配10Hz
        settings.no_rendering_mode = False  # 确保渲染以支持Traffic Manager
        world.apply_settings(settings)
        set_rainy_weather(world)

        # 初始化Traffic Manager (CARLA 0.9.14兼容)
        print("正在配置交通管理系统...")
        traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_boundaries_respawn_dormant_vehicles(2, 50)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.set_random_device_seed(42)  # 固定随机种子确保可重复性

        blueprint_lib = world.get_blueprint_library()
        
        # 选择20辆主车的生成点（确保在不同路段）
        spawn_points = world.get_map().get_spawn_points()
        main_vehicle_spawn_points = []
        main_vehicles = []
        
        print("正在选择主车辆生成点...")
        for i, spawn_point in enumerate(spawn_points):
            # 检查该位置是否在相对直的路段
            waypoint = world.get_map().get_waypoint(spawn_point.location)
            if waypoint.is_junction:
                continue
            # 检查前方是否有足够空间
            forward_waypoint = waypoint.next(10.0)
            if forward_waypoint and not forward_waypoint[0].is_junction:
                main_vehicle_spawn_points.append(i)
                print(f"找到主车辆生成点 #{i}")
                if len(main_vehicle_spawn_points) >= NUM_MAIN_VEHICLES:
                    break
        
        # 如果没有找到足够的生成点，使用默认点
        if len(main_vehicle_spawn_points) < NUM_MAIN_VEHICLES:
            print(f"警告: 只找到 {len(main_vehicle_spawn_points)} 个合适的主车辆生成点，将使用默认点补足")
            for i in range(NUM_MAIN_VEHICLES - len(main_vehicle_spawn_points)):
                main_vehicle_spawn_points.append(i)
        
        # 20种不同颜色
        colors = [
            '255,0,0', '0,0,255', '0,255,0', '255,255,0', '255,0,255',
            '0,255,255', '128,0,0', '0,128,0', '0,0,128', '128,128,0',
            '128,0,128', '0,128,128', '192,192,192', '128,128,128', '64,64,64',
            '255,128,0', '128,64,0', '255,64,128', '64,128,255', '128,255,64'
        ]
        
        print("正在生成主车辆...")
        for i in range(NUM_MAIN_VEHICLES):
            # 使用安全方法获取车辆蓝图
            vehicle_bp = get_vehicle_blueprint(blueprint_lib, i)
            
            # 设置不同颜色
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', colors[i])
            else:
                print(f"警告: 车辆 {i} 不支持颜色属性")
            
            spawn_point = spawn_points[main_vehicle_spawn_points[i]]
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            main_vehicles.append(vehicle)
            print(f"成功生成主车辆 #{i+1} ({vehicle_bp.id}) 在生成点 {main_vehicle_spawn_points[i]}")
        
        # 生成NPC车辆（排除主车占用的生成点）
        traffic_vehicles = spawn_npc_vehicles(
            world, 
            traffic_manager, 
            NUM_NPC_VEHICLES,
            excluded_spawn_points=main_vehicle_spawn_points
        )
        
        # 等待车辆稳定在道路上
        print("等待车辆稳定...")
        for _ in range(20):  # 等待2秒
            world.tick()
            time.sleep(0.1)
        
        # 为每辆主车设置自动驾驶
        for i, vehicle in enumerate(main_vehicles):
            vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)
            print(f"主车辆 #{i+1} 已设置为自动驾驶模式 (Traffic Manager端口: {TRAFFIC_MANAGER_PORT})")
            
            # 设置目标速度
            try:
                current_waypoint = world.get_map().get_waypoint(vehicle.get_location())
                speed_limit = current_waypoint.lane_speed_limit
                target_speed = speed_limit * 0.9
                print(f"主车辆 #{i+1} 检测到道路限速: {speed_limit} km/h, 设置目标速度: {target_speed:.1f} km/h")
            except:
                target_speed = 20.0
                print(f"主车辆 #{i+1} 无法获取道路限速，使用默认速度: {target_speed} km/h")
            
            traffic_manager.set_desired_speed(vehicle, target_speed)

        # 创建存储目录
        os.makedirs(BASE_DIR, exist_ok=True)
        camera_dirs = []
        # 添加20个主车摄像头目录
        for i in range(1, NUM_MAIN_VEHICLES + 1):
            camera_dirs.append(f"main_vehicle_{i:02d}")
            
        # 添加18个固定摄像头目录
        for i in range(1, 19):
            camera_dirs.append(f"fixed_{i:02d}")
            
        for dir_name in camera_dirs:
            os.makedirs(os.path.join(BASE_DIR, dir_name), exist_ok=True)

        # 为每辆主车创建前向摄像头
        for i, vehicle in enumerate(main_vehicles):
            front_bp = blueprint_lib.find('sensor.camera.rgb')
            front_bp.set_attribute('image_size_x', str(IMAGE_RESOLUTION[0]))
            front_bp.set_attribute('image_size_y', str(IMAGE_RESOLUTION[1]))
            front_bp.set_attribute('fov', '90')
            front_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            front_camera = world.spawn_actor(
                front_bp, front_transform, attach_to=vehicle
            )
            # 使用ImageSaver控制保存频率
            front_camera.listen(ImageSaver(
                os.path.join(BASE_DIR, f"main_vehicle_{i+1:02d}"), 
                SAVE_INTERVAL
            ))
            cameras.append(front_camera)
            print(f"为主车辆 #{i+1} 创建前向摄像头 (每0.2秒保存1张)")

        # 获取地图
        map = world.get_map()
        
        # 重新设计18个道路关键点 
        road_key_points = [
            # 1. 主要道路交叉口东北角
            {"x": 75.0, "y": 35.0, "z": 0.0, "distance": 2.5, "height": 2.8, "pitch": -10, "yaw": 45},
            # 2. 主要道路交叉口东南角
            {"x": 10.0, "y": 35.0, "z": 0.0, "distance": -2.5, "height": 3.2, "pitch": -10, "yaw": 135},
            # 3. 西部主干路北侧
            {"x": 55.0, "y": -15.0, "z": 0.0, "distance": 2.5, "height": 2.9, "pitch": -10, "yaw": 0},
            # 4. 西部主干路
            {"x": 100.0, "y": -20.0, "z": 0.0, "distance": -2.5, "height": 4.0, "pitch": -10, "yaw": 0},
            # 5. 西部主干路直路段
            {"x": 45.0, "y": -30.0, "z": 0.0, "distance": -2.5, "height": 3.3, "pitch": -15, "yaw": 45},
            # 6. 北部直道路段西侧
            {"x": 65.0, "y": 0.0, "z": 0.0, "distance": -2.5, "height": 3.0, "pitch": -10, "yaw": 270},
            # 7. 东部直道路段北侧
            {"x": 120.0, "y": 75.0, "z": 0.0, "distance": 2.5, "height": 2.9, "pitch": -12, "yaw": 330},
            # 8. 东部直道路段南侧
            {"x": 100.0, "y": 45.0, "z": 0.0, "distance": -2.5, "height": 2.9, "pitch": -12, "yaw": 0},
            # 9. 西部直道路段北侧
            {"x": 50.0, "y": -10.0, "z": 0.0, "distance": 2.5, "height": 3.1, "pitch": -10, "yaw": 0},
            # 10. 西部直道路段南侧
            {"x": 50.0, "y": -20.0, "z": 0.0, "distance": -2.5, "height": 3.1, "pitch": -10, "yaw": 180},
            # 11. 西部主干路交叉口
            {"x": 45.0, "y": -30.0, "z": 0.0, "distance": -2.5, "height": 3.3, "pitch": -15, "yaw": 135},
            # 12. 西部主干路
            {"x": 40.0, "y": 25.0, "z": 0.0, "distance": -2.5, "height": 3.4, "pitch": -12, "yaw": 0},
            # 13. 东北部区域
            {"x": 110.0, "y": 15.0, "z": 0.0, "distance": 2.5, "height": 2.7, "pitch": -10, "yaw": 180},
            # 14. 西北部区域
            {"x": 40.0, "y": 25.0, "z": 0.0, "distance": -2.5, "height": 3.4, "pitch": -12, "yaw": 180},
            # 15. 东南部区域
            {"x": 100.0, "y": -30.0, "z": 0.0, "distance": -2.5, "height": 3.7, "pitch": -8, "yaw": 90},
            # 16. 西南部区域
            {"x": 180.0, "y": 50.0, "z": 0.0, "distance": 2.5, "height": 3.5, "pitch": -10, "yaw": 45},
            # 17. 中心广场西侧 
            {"x": 48.5, "y": -19.5, "z": 0.0, "distance": 2.5, "height": 3.5, "pitch": -10, "yaw": 90},
            # 18. 西部主干路另一交叉口
            {"x": 80.0, "y": -15.0, "z": 0.0, "distance": 2.5, "height": 2.6, "pitch": -10, "yaw": 45},
        ]
        
        # 创建18个固定摄像头 (全部严格位于道路边缘)
        fixed_transforms = []
        map = world.get_map()
        
        print("正在确定18个道路边缘位置...")
        for i, point in enumerate(road_key_points):
            # 获取参考路点
            location = carla.Location(x=point["x"], y=point["y"], z=point["z"])
            waypoint = map.get_waypoint(location)
            
            if waypoint:
                # 获取道路边缘的有效位置
                result = get_valid_roadside_location(
                    world, 
                    waypoint, 
                    distance=point["distance"],
                    height=point["height"]
                )
                
                if result:
                    location, waypoint = result
                    transform = carla.Transform(
                        location,
                        carla.Rotation(pitch=point["pitch"], yaw=point["yaw"])
                    )
                    fixed_transforms.append(transform)
                    print(f"fixed_{i+1:02d} 位置: x={location.x:.1f}, y={location.y:.1f}, z={location.z:.1f} (道路边缘)")
                else:
                    print(f"警告: 无法为 fixed_{i+1:02d} 找到有效的道路边缘位置")
            else:
                print(f"警告: 无法为 fixed_{i+1:02d} 找到参考路点")
        
        # 确保我们有18个有效位置
        if len(fixed_transforms) < 18:
            print(f"警告: 只找到 {len(fixed_transforms)} 个有效的道路边缘位置，补充默认位置")
            # 补充默认位置
            default_heights = [2.5, 3.2, 3.5, 2.8, 3.2, 2.6, 3.8, 4.0, 3.3, 2.9, 3.1, 3.6, 2.7, 3.4, 3.7, 2.4, 3.0, 2.5]
            default_positions = [
                (72.0, 32.0, -10, 45), (10.0, 38.0, -15, 135), (55.0, -15.0, -10, 0), (100.0, -20.0, -10, 0),
                (45.0, -30.0, -15, 45), (68.0, 0.0, -10, 270), (120.0, 75.0, -12, 330), (100.0, 48.0, -12, 180),
                (50.0, -10.0, -10, 0), (50.0, -20.0, -10, 180), (45.0, -30.0, -15, 135), (40.0, 25.0, -12, 0),
                (115.0, 15.0, -10, 180), (40.0, 25.0, -12, 180), (105.0, -25.0, -8, 270), (50.0, -85.0, -10, 90),
                (-95.0, 20.0, -15, 135), (80.0, -15.0, -10, 45)
            ]
            
            while len(fixed_transforms) < 18:
                idx = len(fixed_transforms)
                x, y, pitch, yaw = default_positions[idx]
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=default_heights[idx]),
                    carla.Rotation(pitch=pitch, yaw=yaw)
                )
                fixed_transforms.append(transform)
                print(f"fixed_{idx+1:02d} 使用默认位置作为补充")
        
        # 创建18个固定摄像头
        for i, transform in enumerate(fixed_transforms):
            fixed_bp = blueprint_lib.find('sensor.camera.rgb')
            fixed_bp.set_attribute('image_size_x', str(IMAGE_RESOLUTION[0]))
            fixed_bp.set_attribute('image_size_y', str(IMAGE_RESOLUTION[1]))
            fixed_bp.set_attribute('fov', '110')
            
            fixed_camera = world.spawn_actor(
                fixed_bp, transform, attach_to=None
            )
            # 使用ImageSaver控制保存频率
            fixed_camera.listen(ImageSaver(
                os.path.join(BASE_DIR, f"fixed_{i+1:02d}"), 
                SAVE_INTERVAL
            ))
            cameras.append(fixed_camera)
            print(f"创建固定摄像头 fixed_{i+1:02d} (每0.2秒保存1张)")

        print(f"已部署 {len(cameras)} 个摄像头 (20主车 + 18固定)")
        print(f"采样率: {1/(SAVE_INTERVAL/FPS):.1f} Hz | 分辨率: {IMAGE_RESOLUTION[0]}x{IMAGE_RESOLUTION[1]}")
        print(f"场景包含: {len(main_vehicles)}辆主车辆 + {len(traffic_vehicles)}辆NPC载具 (含摩托车)")
        print("主车辆已启动，正在遵守交通规则行驶...")

        frame_count = 0
        while True:
            world.tick()
            frame_count += 1
            
            # 每秒打印状态
            if frame_count % FPS == 0:
                elapsed = frame_count / FPS
                status = f"[{time.strftime('%H:%M:%S')}] 运行 {elapsed:.1f}秒 | 总帧数: {frame_count} | 保存帧: {frame_count//SAVE_INTERVAL}"
                # 只显示前5辆车的速度（太多会拥挤）
                for i in range(min(5, len(main_vehicles))):
                    try:
                        speed = main_vehicles[i].get_velocity().length() * 3.6
                        status += f" | 车{i+1}速: {speed:.1f}"
                    except:
                        status += f" | 车{i+1}速: N/A"
                print(status)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n正在清理资源...")
        # 销毁所有摄像头
        for camera in cameras:
            try:
                camera.destroy()
            except:
                pass
        
        # 销毁所有车辆
        for vehicle in traffic_vehicles + main_vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        print("资源清理完成")

if __name__ == '__main__':
    main()

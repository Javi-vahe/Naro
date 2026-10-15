# Naro · Nano Relocalization

![Naro 无人机重定位](docs/naro.png)

轻量级全局激光重定位，离线建立地图索引，在线搜索并验证位置。结合局部结构和激光射线区分相似地点，可配合 FAST-LIO2 等 LIO 前端使用。

- 静止开机定位，无需手动给定初始位姿。
- 使用 CPU 运行，支持 RK3588。
- 提供 ROS 2 接口和三个仿真回放用例。

## 下载源代码

环境：Ubuntu 22.04、ROS 2 Humble。

1. 获取源码：

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Javi-vahe/Naro.git naro
```

2. 安装依赖：

```bash
source /opt/ros/humble/setup.bash
sudo apt install python3-pip python3-colcon-common-extensions python3-rosdep
cd ~/ros2_ws
python3 -m pip install -r src/naro/requirements.txt
test -f /etc/ros/rosdep/sources.list.d/20-default.list || sudo rosdep init
rosdep update
rosdep install --from-paths src/naro --ignore-src -r -y
```

3. 编译：

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select naro
source install/setup.bash
```

## 仿真 Demo

1. 运行庭院示例：

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch naro demo.launch.py case:=courtyard
```

示例使用 Gazebo 采集的地图和静止点云，启动后自动定位并打开 RViz。灰色为地图，绿色为重定位后的扫描，终端输出位置和航向误差。

2. 切换场景，将 `case:=courtyard` 改为 `case:=office` 或 `case:=warehouse`。批量回放和场景文件见 [仿真用例](docs/examples.md)。

## 激光重定位

1. 用 FAST-LIO2 等前端保存 PCD 地图，建立定位索引：

```bash
ros2 run naro prepare_map --cloud map.pcd --output ~/maps/site
```

2. 复制配置文件，修改点云话题、里程计话题和雷达安装位置：

```bash
cp ~/ros2_ws/src/naro/config/reloc.yaml ~/reloc.yaml
```

默认订阅 `/cloud_registered` 和 `/Odometry`，里程计坐标系为 `camera_init`，机体坐标系为 `body`。`lidar_translation` 填写雷达原点在机体坐标系中的 x、y、z，单位为米。

3. 启动节点，保持飞机静止：

```bash
ros2 launch naro reloc.launch.py \
  map_dir:=$HOME/maps/site config:=$HOME/reloc.yaml
```

节点采集三秒点云后进行定位，成功时发布 `map → camera_init` 的 TF。更多参数和 Python 调用见 [使用说明](docs/usage.md)。

## 离线回放

```bash
cd ~/ros2_ws/src/naro
python3 -m pip install .
naro-replay --case courtyard
```

## 开源许可

[Apache-2.0](LICENSE)。配准使用 [small_gicp](https://github.com/koide3/small_gicp)。

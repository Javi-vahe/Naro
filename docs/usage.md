# 使用说明

## 接入 LIO

1. 启动 FAST-LIO2 等 LIO 前端，确认点云和里程计正常发布。

点云使用 PointCloud2，里程计使用 Odometry。点云需要经过畸变补偿，并转换到里程计坐标系中。地图与里程计的 Z 轴均朝上，长度单位为米。

2. 将地图导出为 PCD，运行地图导入工具：

```bash
source ~/ros2_ws/install/setup.bash
ros2 run naro prepare_map --cloud map.pcd --output ~/maps/site
```

支持 ASCII、binary 格式的 PCD，也可以读取 NPY，以及包含 `points` 数组的 NPZ。导入工具会生成定位点云和检索索引，输出目录需要为空。

3. 复制配置文件到用户目录：

```bash
cp ~/ros2_ws/src/naro/config/reloc.yaml ~/reloc.yaml
```

打开 `~/reloc.yaml`，修改对应参数：

```yaml
naro:
  ros__parameters:
    cloud_topic: /cloud_registered
    odom_topic: /Odometry
    odom_frame: camera_init
    base_frame: body
    lidar_translation: [0.0, 0.0, 0.0]
```

`cloud_topic` 对应的点云帧和里程计父帧都使用 `odom_frame`，里程计子帧使用 `base_frame`。`lidar_translation` 按 x、y、z 顺序填写雷达原点在机体坐标系中的位置。

4. 启动定位节点：

```bash
ros2 launch naro reloc.launch.py \
  map_dir:=$HOME/maps/site config:=$HOME/reloc.yaml
```

保持飞机静止，节点会自动采集点云并完成一次定位。采集过程中发生明显移动或转动，会重新开始累计。

## 读取结果

| 话题 | 内容 |
|---|---|
| /naro/result | JSON 结果，包含 accepted、reason 和 T_map_odom |
| /naro/transform | 地图到里程计的 TransformStamped |
| /naro/pose | 定位时的机体 PoseStamped |
| /naro/map | 先验地图点云 |
| /naro/aligned | 定位后的扫描点云 |

`accepted` 为 true 时，可通过 `T_map_odom` 将里程计坐标转换为地图坐标。节点同时发布这两个坐标系之间的 TF。已有其他节点发布该 TF 时，将配置中的 `publish_tf` 设为 false。

再次采集并定位：

```bash
ros2 service call /naro/relocalize std_srvs/srv/Trigger
```

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| collect_seconds | 3.0 | 静止点云采集时长，单位 s |
| threads | 4 | 配准线程数 |
| auto_start | true | 启动后自动执行一次定位 |
| publish_tf | true | 发布地图到里程计的 TF |

地图导入默认搜索地面以上 0.2 至 2 m 的位置，需要调整时，在导入命令中加入 `--min-height` 和 `--max-height`。

## Python 调用

```python
import numpy as np
from naro import localize

result = localize(
    np.load('scan.npy'),
    'maps/site',
    here=[0.0, 0.0, 0.0],
    sensor_origin=[0.0543, 0.0, 0.0418],
)
if result['accepted']:
    transform = np.asarray(result['T_map_odom'])
```

`here` 填写里程计坐标系中的机体位置，`sensor_origin` 填写同一时刻的雷达原点。使用 ROS 2 节点时，雷达原点会根据里程计姿态和安装位置自动计算。

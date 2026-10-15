# 仿真回放

功能包中带有三组 Gazebo 数据，每组都包含地图和静止扫描。地图通过 LIO 建图获得，扫描在另一处开机位置采集。

| 用例 | 场景 |
|---|---|
| office | 办公室 |
| courtyard | 庭院 |
| warehouse | 仓库 |

## 在 RViz 中查看

1. 编译功能包后，启动回放：

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch naro demo.launch.py case:=courtyard
```

2. 等待定位完成，查看绿色扫描是否与灰色地图重合。终端会输出位置误差 `position_cm` 和航向误差 `yaw_deg`。

将 `case` 换成 `office` 或 `warehouse` 可以切换场景。只查看终端结果时，在命令末尾加上 `rviz:=false`。

## 批量运行

```bash
ros2 run naro replay --all --output results/replay.json
```

结果保存在 `results/replay.json`，包含每组用例的变换矩阵、误差、耗时和定位状态。

加入 `--baseline` 可以同时运行基础版本，比较同一扫描的定位结果：

```bash
ros2 run naro replay --all --baseline --output results/compare.json
```

## 使用 Gazebo 场景

进入功能包目录，打开对应的场地文件：

```bash
cd ~/ros2_ws/src/naro
gz sim examples/worlds/courtyard.sdf
```

办公室和仓库分别对应 `office.sdf`、`warehouse.sdf`。场景中保留了建筑、家具和障碍物，可以加入自己的无人机与传感器，再将 LIO 输出接入定位节点。

原用例的雷达配置放在 `examples/lidar.json` 中，可用于设置扫描范围、线数和安装位置。

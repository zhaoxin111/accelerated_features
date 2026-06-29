# XFeat 外部调用 API

本文档说明如何在别的 repo 中调用本仓库的关键点检测和匹配接口，以及如何用命令行调试同一套 API。

## 1. 环境准备

在外部 repo 中，把本仓库加入 `PYTHONPATH`：

```bash
export XFEAT_REPO=/home/maicro/accelerated_features
export PYTHONPATH="${XFEAT_REPO}:${PYTHONPATH}"
```

确认依赖已安装：

```bash
pip install torch opencv-contrib-python numpy tqdm
```

默认权重路径是：

```text
/home/maicro/accelerated_features/weights/xfeat.pt
```

## 2. Python 调用

### 2.1 关键点检测

```python
import cv2
from xfeat_api import XFeatMatchingAPI

api = XFeatMatchingAPI(top_k=4096)

image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
features = api.detect(image)

keypoints = features["keypoints"]       # float32, shape: [N, 2], xy
descriptors = features["descriptors"]   # float32, shape: [N, 64]
scores = features["scores"]             # float32, shape: [N]
```

### 2.2 两图匹配

```python
import cv2
from xfeat_api import XFeatMatchingAPI

api = XFeatMatchingAPI(top_k=4096)

image0 = cv2.imread("prev.jpg", cv2.IMREAD_COLOR)
image1 = cv2.imread("curr.jpg", cv2.IMREAD_COLOR)

result = api.match(
    image0,
    image1,
    method="sparse",
    min_cossim=0.82,
    ransac=True,
    ransac_thr=4.0,
)

points0 = result.points0          # float32, shape: [M, 2], image0 xy
points1 = result.points1          # float32, shape: [M, 2], image1 xy
matches = result.matches          # int64, shape: [M, 2]
inlier_mask = result.inlier_mask  # bool[M] or None

print(result.num_matches, result.num_inliers)
```

`method` 可选：

- `sparse`: XFeat sparse keypoints + mutual nearest descriptor matching，默认推荐调试入口。
- `semidense`: XFeat* 半稠密匹配。
- `lighterglue`: XFeat sparse keypoints + LighterGlue，需要额外依赖 `kornia`。

### 2.3 从文件直接匹配

```python
from xfeat_api import XFeatMatchingAPI

api = XFeatMatchingAPI(top_k=4096)
result = api.match_files(
    "prev.jpg",
    "curr.jpg",
    method="sparse",
    min_cossim=0.82,
)
```

## 3. 命令行调试

`xfeat_api.py` 也可以直接作为调试 CLI 使用：

```bash
cd /home/maicro/accelerated_features

python xfeat_api.py \
  --image0 data/pair1/1776299438444_prev.jpg \
  --image1 data/pair1/1776299438444_curr.jpg \
  --method sparse \
  --top-k 4096 \
  --min-cossim 0.82 \
  --json outputs/api_debug/pair1_matches.json \
  --npz outputs/api_debug/pair1_matches.npz \
  --viz outputs/api_debug/pair1_matches.jpg
```

CLI 会在终端打印简要 JSON：

```json
{
  "image0": "...",
  "image1": "...",
  "method": "sparse",
  "num_matches": 123,
  "num_inliers": 100
}
```

输出文件说明：

- `--json`: 完整可读结果，包含 `points0`、`points1`、`matches`、`inlier_mask`。
- `--npz`: numpy 压缩结果，适合下游程序直接加载。
- `--viz`: OpenCV 匹配可视化图。

## 4. 接口约定

- 输入图像支持 OpenCV 常用的 `np.ndarray`，形状为 `H,W,C`，BGR/RGB 都可用于特征匹配。
- `uint8` 图像会在 API 内部归一化到 `[0, 1]` 后做关键点检测。
- 输出关键点坐标使用 `xy` 顺序，单位为原图像素。
- `result.matches[i] == [i, i]`，表示 `points0[i]` 与 `points1[i]` 是同一条匹配。当前 API 返回的是已经配对后的点数组，而不是原始 keypoint 索引。
- `ransac=True` 时会用 `cv2.findHomography(..., cv2.USAC_MAGSAC, ...)` 估计内点；匹配数少于 4 或估计失败时 `inlier_mask=None`。
- 外部 repo 长期集成时，建议只依赖 `XFeatMatchingAPI.detect()`、`XFeatMatchingAPI.match()`、`XFeatMatchingAPI.match_files()` 和 `MatchResult` 字段，不依赖 demo 脚本。

## 5. 常见问题

### 想固定 CPU 跑

在导入 API 前设置：

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### 想使用自定义权重

```python
api = XFeatMatchingAPI(weights="/path/to/xfeat.pt", top_k=4096)
```

### 外部 repo import 失败

检查：

```bash
echo "$PYTHONPATH"
python - <<'PY'
from xfeat_api import XFeatMatchingAPI
print(XFeatMatchingAPI)
PY
```

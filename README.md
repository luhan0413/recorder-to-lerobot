# Recorder → LeRobot 轉換器

將 Recorder raw v4／v5 錄製轉成 **LeRobot Dataset v3、10 FPS**。依 CSV 時間篩選品質、切分片段、重採樣，再由官方 writer 輸出。本機執行，保留原始錄製。

## 安裝

目前以 macOS、Python 3.12、LeRobot 0.6.1 驗證。在專案根目錄執行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

已有 `.venv` 時，只需啟用環境。以下指令均在專案根目錄執行。

## 使用

先檢查錄製並保存品質報告：

```bash
python converter.py \
  --input raw_data/my_recording \
  --validate-only \
  --report outputs/quality_check.json
```

轉換資料集：

```bash
python converter.py \
  --input raw_data/my_recording \
  --output outputs/my_dataset \
  --repo-id local/ur-drawer
```

`--input` 接受單一 episode 或包含 `episode_*` 的父資料夾。預設輸出 `outputs/dataset`，相機為 `left`；雙相機使用 `--cameras left wrist`。`--repo-id` 是資料集識別名稱，不會觸發上傳。

輸出路徑與報告檔案必須尚未存在。輸入格式不合格會拒絕整批；品質不合格的樣本會排除並記錄。正式轉換至少要有一段兩幀以上的有效片段。`--validate-only` 可用來查看沒有有效片段的原因。

## 輸入格式

```text
raw_data/my_recording/
├── episode_000000/
│   ├── episode.json
│   ├── frames.csv
│   └── episode_000000.mp4
└── episode_000001/
    ├── episode.json
    ├── frames.csv
    └── episode_000001.mp4
```

`episode.json` 至少包含：

```json
{
  "raw_schema_version": 4,
  "status": "completed",
  "complete": true,
  "task": "Open the drawer",
  "robot_type": "ur7e",
  "position_unit": "meter",
  "rotation_unit": "radian",
  "rotation_type": "axis_angle",
  "pose_frame": "robot_base",
  "joint_unit": "radian",
  "gripper_source": "commanded",
  "gripper_zero": "fully_open",
  "gripper_one": "fully_closed",
  "cameras": {"left": "episode_000000.mp4"}
}
```

v5 可使用 `"format": "raw", "version": 5` 代替 `raw_schema_version`。相機也接受 `{"video_file": "episode_000000.mp4"}`；影片須位於 episode 目錄內。`visual_training_eligible=false` 會拒絕轉換。

CSV 的前 15 欄必須依序為：

```csv
frame_index,timestamp,state_x,state_y,state_z,state_rx,state_ry,state_rz,gripper_command,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6
```

必要的品質診斷欄位接在後面：

```csv
rtde_age_seconds,camera_left_pairing_skew_seconds
```

每個輸出相機都需有自己的 `camera_<名稱>_pairing_skew_seconds`。診斷缺值、負值或超標會排除該筆。其他附加欄位必須是數值或空白。

- `frame_index` 從 0 連續遞增，影片第 i 幀對應 CSV 第 i 列。
- `timestamp` 是秒，從 0 開始且嚴格遞增，可不等距。若提供 `sample_host_monotonic`，每列都要有值，其相對時間須與 `timestamp` 一致。
- TCP 位置以公尺表示，旋轉是機器人 base 座標系的 axis-angle／rotvec，單位為弧度。
- 關節順序為 shoulder_pan、shoulder_lift、elbow、wrist_1、wrist_2、wrist_3。
- 夾爪是 0～1 的命令值，0 全開、1 全閉。
- 所有宣告影片的幀數須等於 CSV 列數，尺寸固定且寬高為偶數；各 episode 的 robot type 與輸出相機尺寸須一致。

## 品質與重採樣

| CLI 參數 | 預設值（秒） | 意義 |
| --- | --- | --- |
| `--max-sample-gap-seconds` | 0.150 | 原始相鄰樣本最大間隔 |
| `--max-image-alignment-error-seconds` | 0.050 | 影像樣本與輸出格點的最大時間差 |
| `--max-rtde-age-seconds` | 0.050 | RTDE 資料最大 age |
| `--max-camera-rtde-skew-seconds` | 0.050 | 相機與 RTDE 最大 host pairing skew |

門檻須為有限正數。排除樣本與過大間隔都會切開片段，插值與 action 不跨越切點。

每段從第一個有效 CSV timestamp 建立 0.1 秒格點，不外推。位置與關節採線性插值，旋轉採 SO(3) 最短路徑插值，夾爪保持格點之前最近的命令；影像採最近樣本。影像時間差超標的格點會再次切段，少於兩個輸出格點的片段捨棄。

Action 位置為 `p_next - p_current`，旋轉為 `rotvec(R_next @ R_current.T)`，夾爪為下一個格點的命令。末幀位置與旋轉 delta 為零、夾爪保持當前命令，`next.done=true`。

時間基準為 CSV `timestamp`。MP4 播放時間與 native clock 不用來建立重採樣格點；host 診斷通過也不表示感測器曝光已同步。

## 輸出

```text
outputs/my_dataset/
├── data/                      # 官方 writer 產生的 Parquet
├── videos/                    # 各相機 H.264 影片
└── meta/
    ├── info.json
    ├── tasks.parquet
    ├── stats.json
    ├── episodes/
    ├── modality.json          # 欄位語意
    ├── conversion.json        # 轉換設定與原始 metadata
    └── quality_report.json    # 排除原因、切點與對齊資訊
```

| 資料欄位 | 內容 |
| --- | --- |
| `observation.state` | 7 維：TCP pose＋夾爪命令 |
| `observation.joints` | 7 維：六關節＋夾爪命令 |
| `action` | 7 維：base pose delta＋下一筆夾爪命令 |
| `observation.images.<相機>` | RGB 影像 |
| `annotation.language.language_instruction` | 任務文字索引 |
| `next.done` / `next.reward` | 片段終止標記／固定 0 |

每個輸出片段的時間從 0 開始。影片使用 H.264 CRF18／YUV420 有損編碼，分檔與 episode 範圍由官方 writer 管理。完整寫入後才移至輸出路徑；一般失敗會清理暫存目錄，強制終止可能留下隱藏暫存目錄。

品質報告包含排除列、缺失診斷、被拒絕格點、短片段、P95／最大時間差、來源影格索引與重複影格比例。`quality_verified` 只表示必要診斷沒有缺值；超標樣本仍會排除。

## 選用時鐘診斷

預設 `--timing-policy sample-order`。提供 `--clock-maps clock_maps.json` 會啟用 `calibrated` 診斷，額外比較 native 感測器時間；重採樣仍使用 CSV timestamp。

映射以 episode 名稱區分，每段要提供 controller 與所有輸出相機的映射：

```json
{
  "episode_000000": {
    "rtde_controller_timestamp": {
      "scale": 1.0, "offset": 950.0, "provenance": "independent calibration events"
    },
    "camera_left_sensor_timestamp": {
      "scale": 0.001, "offset": 995.0, "provenance": "independent calibration events"
    }
  }
}
```

公式為 `common_time_seconds = native_timestamp × scale + offset`，以上數值僅供示意。對應 CSV 欄位須每列都有值。

校準工具可從至少三組獨立配對事件擬合映射：

```bash
python -m tools.calibrate_clocks --anchors anchors.json --output clock_maps.json
```

`anchors.json` 格式如下，每段都需列出 controller 與各相機：

```json
{
  "episode_000000": {
    "camera_left_sensor_timestamp": {
      "native": [5000, 5500, 6000],
      "common_host_monotonic": [1000, 1000.5, 1001],
      "provenance": "independently matched calibration events"
    }
  }
}
```

事件須涵蓋整次錄製，不能把 host 收件時間當曝光時間。工具預設最大擬合殘差 0.005 秒，可用 `--max-residual` 調整。產生的映射包含校準範圍，轉換器拒絕外推與時鐘倒退；可接受重複 native timestamp。此功能僅提供診斷，不會證明硬體同步。

## 範例與測試

建立兩段合成錄製並轉換、讀回：

```bash
python -m tools.make_example
python converter.py \
  --input raw_data/example \
  --output outputs/example \
  --repo-id local/ur-example \
  --clock-maps raw_data/example/clock_maps.json
python -m tests.verify_example --root outputs/example
```

合成資料只供功能驗證。執行自動測試：

```bash
python -m unittest discover -s tests -v
```

測試在臨時目錄執行，涵蓋品質切段、插值與 action、校準範圍、輸入拒絕、官方 writer／reader 往返、輸出保護與失敗清理。

## 程式結構

```text
converter.py          CLI、輸入驗證與轉換流程
quality.py            品質篩選、重採樣與 action
exporter.py           影片讀取與 LeRobot 資料集輸出
schema.py             共用欄位、關節名稱與預設門檻
requirements.txt      執行所需套件
tools/                合成資料與時鐘校準工具
tests/                自動測試與範例讀回驗證
```

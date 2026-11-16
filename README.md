# voice-doppler-engine
Python 游戏音频多普勒库，处理移动声源、距离延迟、速度连续性和重采样率

## 用法

```python
from doppler import DopplerEngine, constant_velocity

# 声源从 200 m 外以 50 m/s 接近位于原点的听者
engine = DopplerEngine(
    constant_velocity((200.0, 0.0, 0.0), (-50.0, 0.0, 0.0)),
    speed_of_sound=343.0,   # m/s
    sample_rate=48000,      # Hz
    max_mach=0.9,           # 径向速度钳制
)

for pcm_block in stream:            # float 序列，任意块长
    out_block, pitch = engine.process(pcm_block)
    # out_block 与输入等长，按听者时钟对齐；pitch 为当前音高系数
tail = engine.flush()               # 输入结束后排空仍在传播中的声音
```

轨迹是 `pos(t) -> (x, y, z)` 的可调用对象（SI 单位：米、秒、米/秒）。
`constant_velocity` 与 `piecewise_velocity` 提供匀速与分段匀速轨迹。
引擎按发射时刻（retarded time）方程 `tau + |L(t) - S(tau)| / c = t`
逐样本推进发射时钟，延迟平滑变化，不会按块跳样本；输出与分块方式无关。

运行测试：`python -m unittest`

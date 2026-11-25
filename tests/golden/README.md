# golden：内核黄金样本

这里只放一份东西：`kernel_ref.json` —— **2026-11-26 性能优化之前**用当时的
`simulate_seq` 冻结下来的输出。它存在的唯一理由是给"重构不改数值"提供证据。

## 里面存了什么

| 字段 | 含义 |
|---|---|
| `args` | 该算例的参数（n、m、R、dt、n_cycles、burn_frac、lam、beta、trace） |
| `pstd` / `mx` | 完整 float64（JSON 用 repr 存，round-trip 精确） |
| `p_shape` / `p_sha256` | Poincaré 点矩阵的形状与逐位哈希 |
| `t_shape` / `X_sha256` | `trace=True` 时的时间轴形状与轨迹哈希 |

## 怎么用

```bash
.venv\Scripts\python.exe -m pytest tests/test_kernel_golden.py -q
.venv\Scripts\python.exe tools\bench_speed.py        # 顺带也做一遍逐位校验
```

**过了就是过了**：哈希相等意味着优化/重构前后每一个 Poincaré 点都逐位相同。
用容差比对是不够的——混沌系统里 1 ulp 会被指数放大，`allclose` 只能证明
"量级没变"，证明不了"数值没动"。

## 什么时候该重做

只有在内核被**有意识地**升级时才重做，例如：换了离散化、改了步长口径、
修了截面积分的 bug、论文里的数也跟着改了。做法：

```bash
.venv\Scripts\python.exe tools\make_kernel_golden.py        # 先看差异，不落盘
.venv\Scripts\python.exe tools\make_kernel_golden.py --yes  # 确认后覆盖
```

然后在提交说明里写清**为什么数值该变**、哪些图/表跟着重算过。

**不要**为了让测试变绿而重做——测试红了就是在说"数值被动了"，这时候该去
查内核，不是查黄金样本。

算例定义在 `tools/kernel_cases.py`，测试、基准、重冻结三处共用同一份；
新增算例时记得同时重冻结，否则 `test_golden_covers_all_cases` 会拦下来。

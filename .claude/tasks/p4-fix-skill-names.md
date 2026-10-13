# P4-FIX · 热力图/推荐要显示真实知识点名

## 问题(已实测)

`app.py` 的 `/api/students/{uid}/state` 和 `/recommend` 返回的 `skill_name` 是
`f"skill_{skill_id}"`(如 `skill_42`),**不是数据集里的真实知识点名**。

数据集里 `skill_id -> skill_name` 映射完整(111 个),实测:
```
1  -> "Box and Whisker"
2  -> "Circle Graph"
42 -> "Perimeter of a Polygon"
```

热力图和推荐列表显示 `skill_42` 对现场演示不可读。

## 需求

1. **`app.py`**:加载数据时构建 `skill_id -> skill_name` 映射
   (同一 skill_id 可能有多个名字,取**出现次数最多**的那个);
   `/state` 与 `/recommend` 两个端点都返回真实 `skill_name`;
   映射里查不到时才回退 `f"skill_{id}"`。
2. **新增测试**:断言 `/api/students/{uid}/state` 返回的 knowledge point 名**不以 `skill_` 开头**
   (用最小模型注入即可,别跑全量拟合)。
3. `web/index.html` 无需大改(若名字较长导致排版问题,做最简处理即可)。

## DoD

- `uv run pytest -q` 全绿,**现有 83 个测试一个都不能红**。
- 起服务后 `GET /api/students/14/state` 的 `skills` 里能看到真实知识点名(如 `Perimeter of a Polygon`)。

## 约束

- 只改 `app.py`(+ 测试);不要动 `src/` 下的算法模块。
- 不引入新依赖。

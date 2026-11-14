# 五日优势代码说明

## 公开范围

`src/relative_strength_l/core.py`从原五日优势实验按函数提取，保留原始计算语义，独立依赖NumPy和pandas。包含15个函数，不包含其他选股策略、桌面界面、用户配置、行情缓存加载器或交易账户功能。

| 函数 | 对应计算 |
|---|---|
| distribution_scores | 剔除自身的收益排名P、横截面标准化Z及原始差值L |
| daily_score | 100×[0.7P+0.3×(1+tanh(Z/2))/2] |
| score_windows | 五日窗口，从最早到最新，缺失不补齐 |
| candidate_weights | 狄利克雷随机候选，每日至少5%，加两组参考 |
| top_indices | 前20排名，同分边界使用固定列顺序 |
| temporal_split | 最后五年，前四年训练，后一年测试，尾部保守隔离 |
| training_opportunities | 先截断训练价格，再生成峰值和触及标签 |
| search_training | 只在训练上按峰值相对改善优先搜索 |
| rank_trials / training_shortlist | 保留训练顺序及前15名 |
| peak_returns / outcome_labels | 完整窗口未来峰值、触及和涨跌标签 |
| _row_mean / forward_returns / training_inputs | 原计算流程的必要聚合与标签辅助函数 |

forward_returns为辅助开盘到开盘标签函数；公开示例的主要训练目标仍为五日峰值。outcome_labels的`return5`在传入`peak_fraction=0.7`时表示事后峰值情景，不表示到期收益。

## 输入CSV

必须包含以下列，按日期和股票代码唯一：

```csv
date,code,open,high,close,amount
```

使用一致的价格调整口径，股票代码作为字符串保留前导零；所有股票应按同一个交易日历对齐。缺失报价留空，不补零。成交额只检查是否为正，其金额倍率不影响评分。未来行情不得改写过去信号，历史退市池和公司行为需要由数据提供者处理。

原研究使用3,052只股票、至少500只参照股票，固定前20只。示例按代码排序以稳定处理边界并列。时间划分根据`--asof`截取后实际最后日期计算最近五个日历年，至少需要200个训练信号日和100个后段日期。

## 安装与运行

建议在独立环境使用Python 3.12或3.13。原提取函数已在Python 3.13.17、NumPy 2.2.6、pandas 2.3.3核验。

```bash
python -m venv .venv
# 激活环境后
python -m pip install -e .
python -B -m unittest discover -s tests -v
python examples/research_from_csv.py local_data/stocks.csv --asof 2026-11-11 --output runs/research
```

默认随机种子20261112、随机候选1000组、保留15组、每期20只。可使用`--random-count`缩小搜索，但这会改变实验配置，不能冒称重现论文全部搜索。

示例输出：

- frozen_weights.json：先保存训练选出的权重，再构造后段标签。
- training_trials.csv：该次运行的所有训练候选成绩。
- periods.csv：信号日、模型、日期内均值及各指标有效股票数。
- summary.csv：对日期等权平均的机会统计，训练顺序保存在冻结文件中。

示例没有重建桌面程序，也没有运行实盘成交回测。论文中的指数对照、完整资金账本和区块重采样不能仅凭此简洁入口全部复现；它用于公开五日优势的核心计算与训练流程。提供同版本原始数据、股票池及完整环境，才有条件检验与历史快照的逐项一致性。

## 测试与边界

测试覆盖剔除自身及并列排名、样本标准差、论文候选227的权重、五日顺序与缺失、排名边界、完整峰值窗口、训练排序和测试价格污染隔离。测试使用合成数据，不是新增的A股盈利验证。

未来最高价只允许用于标签。公开代码不提供自动交易、收益保证或可实现卖出点。高峰值可以与较低到期收益并存，实际执行还需要T+1、委托成交、停牌、价格限制、费用和持仓风险模型。

# 配套数据字典

收益率和比例在CSV中通常用小数保存，例如0.05代表5%。图中收益率乘100显示百分比，超额乘100显示百分点。CSV保留缺失空值，不填零。

| 文件 | 统计单位 | 用途 |
|---|---|---|
| daily_test.csv | 测试表现日期 | strategy_daily、index_daily为昨收至今收；excess为二者差；market_basket为另一路市场日结果 |
| daily_summary.csv | 阶段 | 训练／测试共同有效日期的均值、比例、标准差和样本数 |
| top15.csv | 训练入围方案 | rank为训练顺序；w1最早、w5最新；peak为日期等权峰值；test_peak_n是股票标签数 |
| original_summary.csv | 原研究模型×阶段 | 原始汇总表，保留全部15组与等权、市场的训练／测试字段 |
| opportunities_by_signal_date.csv | 信号日×模型 | 方案01、五日等权、市场的原始日期均值，用于机会对比 |
| scenario_summary.csv | 阶段×模型 | 已结算股票观察与事后峰值70%资金情景 |
| scenario_curves.csv | 表现日期×模型 | net归一化本金；strategy、market、peak:sh.000001为事后情景，price:sh.000001为原始指数价格 |
| timing_by_day.csv | 对齐的测试表现日期 | 215日的overnight、intraday、daily及index_intraday；每行20只 |
| monthly.csv | 月份 | 共同有效日期内的日均收益，不是月累计收益；valid与calendar描述覆盖 |
| peak_histogram.csv | 峰值区间 | lower_pct含下端、upper_pct不含上端，最后一档无上限；count合计4664 |
| statistics.json | 各类统计汇总 | 数据规模、分位数、条件重采样区间及权重分布 |

## 旧字段名称的语义

original_summary.csv中的“平均持有收益”“平均净收益”和opportunities_by_signal_date.csv中的return5、return、net_return，是原系统沿用的列名。在本次研究配置中，它们对应峰值70%的事后情景及扣费情景，**不是固定五日到期收益，也不是可实现收益**。峰值字段peak5、peak10对应真实历史窗口最高价空间；next_day、entry_day在原逐日汇总中分别表示上涨股票比例，不是连续收益率。

方案对比采用原研究的日期等权平均；当前情景summary的完整峰值采用已结算完整股票观察平均。二者样本边界和分母不同，不能要求11.089%与11.051%相等。股票观察可重复，不能等同独立样本。

## 图表与输入映射

| 图 | 输入 |
|---|---|
| 1 流程 | 固定计算逻辑示意，无行情数值 |
| 2 评分响应 | 固定评分公式的数学函数，无历史拟合 |
| 3 权重扇形与均值 | statistics.json、top15.csv |
| 4 前15名热力图 | top15.csv |
| 5 全部15名表现 | top15.csv |
| 6 全测试期单日曲线 | daily_test.csv |
| 7 末尾60日放大 | daily_test.csv固定尾段 |
| 8 月份与覆盖 | monthly.csv |
| 9 机会基准 | original_summary.csv |
| 10 峰值分布 | peak_histogram.csv、statistics.json |
| 11 隔夜与日内 | timing_by_day.csv |
| 12 事后情景 | scenario_curves.csv |
| 13 覆盖与区间 | statistics.json |

第6、7图连接缺失日前后有效点，叉号仅标识缺失位置。第12图上半部分故意展示两种不同口径以解释原结果，不能读成可实现超额；下半部分才是相同峰值假设的对照，但同样依赖未来信息。

统计区间是已选方案的条件区间，未消除全部模型搜索和人工选择的偏差。当前材料不能替代完整历史成分、退市、公司行为、复权和成交约束审计。

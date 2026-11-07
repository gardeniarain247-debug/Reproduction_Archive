# IJFS 论文返修：Model A 实证代码与结果复现说明（交稿版）

**论文：** *Distribution-Driven Linearization of Credibilistic Mean-RVaR Portfolio Optimization under Coherent Trapezoidal Fuzzy Returns*

本包包含模型 A（最小下行风险）的 Step 1–5 程序、补充实验和统计模块、原有输入/输出及测试文件。**必须区分**“利用既有结果完成核查”和“从本地 BaoStock 缓存独立重算”。文件存在、哈希核对或测试通过，均不能单独证明已完成全部独立复现或真实市场成交核验。供国际审稿人阅读的主要说明为同目录 `README.md`。

## 1. 目录及模块

以下命令均在包含 `step1.py` 的代码根目录执行（ZIP 中的 `IJFS/`；本地可能是 `D:\latex\code`）。

```text
IJFS/
├── README.md / README_zh.md       英文/中文交稿说明
├── requirements.txt               Python 依赖
├── CODE_SHA256_final.json               既有代码 SHA-256 清单（须按最终版本核对）
├── data/                          成分股、国债利率与指数收益
├── results1/                      既有 Stage 1 完整结果
│   ├── baostock_cache/            本地 BaoStock 行情缓存
│   ├── full_return_matrix.csv
│   ├── full_turnover_matrix.csv
│   ├── risk_free_rate.csv
│   ├── rolling_windows.csv
│   ├── windows_universe.csv
│   ├── fuzzy_parameters.csv
│   ├── endpoint_calibration.csv
│   ├── bootstrap_indices.npz
│   └── step1_run_metadata.json
├── results_execution/             冻结的实证结果，禁止随意覆盖
│   ├── results2/                  模型 A 优化及策略锁定
│   ├── results3/                  实际持仓递推和逐日回测
│   ├── results4/                  业绩、统计检验与状态结果
│   ├── results5/                  论文图形
│   ├── results5_0/                保留的其他历史图形输出
│   ├── results_remaining/         端点、替换诊断和条件 k_i
│   └── completion/                历史补全核验/执行敏感性
├── manuscript/                    论文 TeX、补充材料、参考文献及图片
├── step1.py – step5.py            五阶段核心程序
├── baseline_pipeline.py          正式 Step 2–5 与剩余实验入口
├── finish_all.py                 已有结果的补充统计及审查入口
├── run_all.py                    在 Step 1 齐全后调用主流程和 finish_all
├── empirical_statistics.py        HAC、区块 Bootstrap、Holm 等
├── remaining_experiments.py       同初始条件替换、端点重抽样
├── ki_calibration.py              条件资产特定 k_i 投影
├── review_completion.py           审稿导向的结果完整性检查
├── controlled_diagnostics.py      控制条件比较
├── execution_sensitivity.py      成交时点及失败回退敏感性
├── data_evidence_audit.py        本地缓存与输入数据审计
└── test_*.py、test_support*.py    数学、回归和合成数据测试
```

`manuscript/` 可能包含多个历史版本，投稿时应另行确认最终 `main.tex`、`supplement.tex`、PDF 和图片完全一致。修改任何代码之后，原有 `CODE_SHA256_final.json` 不应被当作新版代码的有效哈希证明。

## 2. 环境

```powershell
python -m pip install -r requirements.txt
python --version
python -m pip freeze > environment_installed.txt
```

依赖包括 NumPy、pandas、SciPy、PyArrow、CVXPY、OSQP、Matplotlib 和 openpyxl。`requirements.txt` 为最低版本约束，并非完整锁定环境。既有运行记录涉及 Python 3.9.23、SciPy 1.13.1；正式数值复现需要记录系统、求解器及依赖版本。

## 3. 先单独运行 Step 1：从 baostock_cache 生成完整 Stage 1

`step1.py` 使用 `results1/baostock_cache/` 的本地价格/换手率缓存，以及 `data/` 的成分股和国债利率输入，生成收益矩阵、40 个滚动窗口、端点校准、利率对齐审计及 `bootstrap_indices.npz`。默认 250 日估计、60 日持有、60 日滚动、至少 200 条有效估计收益；基线端点为 (0.05, 0.25, 0.75, 0.95)，5 日和 10 日移动区块均为 300 次抽样。未传 `--allow-network` 时使用本地缓存。

**情况一：全新解压包的 `results1/` 只有缓存，尚无冻结的核心 CSV。** 此时可以直接在同一目录构建完整 `results1/`：

```powershell
python step1.py --mode build --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\results1
```

**情况二：当前提供的 IJFS 包已包含完整 `results1/`。** `step1.py` 为保护冻结结果，会拒绝在已有核心 CSV 的 `results1/` 中再次 build，即使指定 `--overwrite` 也不应绕过。若只使用现有结果，可直接进入第 4.1 节，不必再造一个目录；若要从缓存独立验证 Step 1，则生成**一个新输出目录**：

```powershell
python step1.py --mode build --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\stage1_independent
```

`stage1_independent/` 是命令执行成功后**自动创建的输出目录名称示例**，当前嵌套结构中原本没有它，也不要求事先手工创建。它不能未经哈希和来源核验就与旧 `results_execution/` 混合使用。只审计现有 `results1/` 则使用：

```powershell
python step1.py --mode audit --data .\data --results .\results1 --cache .\results1\baostock_cache --out .\stage1_audit
```

`stage1_audit/` 也是新增审计输出，**不会**形成完整的 Stage 1 核心矩阵。正式重建时不要使用 `--skip-bootstrap`。

**Step 1 验收：** 检查 `step1_run_metadata.json` 的完成状态、40 个窗口和失败审计项，核查 `full_return_matrix.csv`、`full_turnover_matrix.csv`、`risk_free_rate.csv`、`rolling_windows.csv`、`windows_universe.csv`、`fuzzy_parameters.csv`、`endpoint_calibration.csv`、`bootstrap_indices.npz` 是否存在；结合 `input_data_audit.csv` 和 `step1_run.log` 判断。审计 review 标志不应自动删除；`failed_checks` 时不得进入正式后续流程。

## 4. 后续路线：已有结果补全或完整重算

### 4.1 使用随包已有的完整 `results1/` 和 `results_execution/`（不重算主实证）

```powershell
python finish_all.py --source .\results_execution --input .\results1 --data .\data --cache .\results1\baostock_cache --out .\completion_submission_check
```

`--source` 是包含 `results2/`、`results3/`、`results4/` 等子目录的**父目录**，不能指向 `results_execution/completion/`；`--out` 是新建或空目录，不能覆盖现有历史结果。这条路线只执行补充核查，**不能写成从缓存重新独立计算全部 40 个窗口**。

### 4.2 从缓存独立重算全部正式实验

先完成第 3 节的 Step 1 build。若输出为 `stage1_independent/`：

```powershell
python baseline_pipeline.py --input .\stage1_independent --data .\data --out-root .\modelA_independent
python finish_all.py --source .\modelA_independent --input .\stage1_independent --data .\data --cache .\results1\baostock_cache --out .\completion_independent
```

如果 Step 1 在一个尚无冻结结果的 `results1/` 中原地构建成功，则把这两条命令中的 `stage1_independent` 改为 `results1`。`baseline_pipeline.py` 顺序执行 **Step 2 → Step 3 → remaining_experiments → Step 4 → Step 5**，生成 `results2/`、`results3/`、`results_remaining/`、`results4/`、`results5/` 和对应审计记录。正式配对推断至少 2,000 次抽样，端点重抽样为 300 次。Pilot 不等于正式结果。

`run_all.py` 可以在 Stage 1 已齐备后作为替代统一入口，自动运行 `baseline_pipeline` 再运行 `finish_all`；它**不负责重新生成 Step 1**。例如：

```powershell
python run_all.py --input .\stage1_independent --data .\data --out-root .\modelA_unified
```

若只对现有结果补全，可用 `python run_all.py --source .\results_execution --input .\results1 --data .\data --cache .\results1\baostock_cache --out-root .\modelA_unified_existing`，输出目录需为空。

### 4.3 可选执行敏感性

若 Stage 1、完整 Stage 2、指数输入均齐备，在 `finish_all.py` 命令后增加 `--with-execution`，计算 formation-close/repair、next-close/hold、next-close/repair 等变体及下游统计。这些是理想化收盘价格和换手率代理下的敏感性分析，不是真实订单成交核验。`finish_all.py` **不支持**旧版 README 中的 `--verify-only`、`--overwrite`、`--endpoint-replicates` 参数。

## 5. 模型及数据口径

本研究固定采用 **Model A**：以净预期收益下限为约束最小化下行风险，目标函数不包含额外均值奖励；买入、卖出成本均为 **0.30%**，从现金预算支付，金额按调仓前净值计量。策略逐窗保留自身实际持仓与现金并计入价格漂移；持续不可行时维持持仓，不能把失败窗口当作已求得最优解。三项尾部风险为 CVaR .05、RVaR [.05,.20]、CVaR .20。同一收益对象的分位数恒等式不等于分别优化的策略实际业绩恒等式。

`results_execution/results3/` 为逐日收益、持仓和执行状态；`results4/` 为业绩、HAC/Bootstrap/Holm 和市场状态；`results5/` 为绘图；`results_remaining/` 为端点与 k_i 等补充诊断；`completion/` 为既有核验。主文 Figure/Table 应与实际提交的 `manuscript/figures/` 及统计 CSV 逐一核对。

## 6. 测试、溯源与局限

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```

测试以合成数据、独立数值算例和程序不变量验证公式、现金预算、停牌、缺失值及统计行为；即使全通过，也不意味着已从历史输入重新计算全部实证。若模块导入失败，应先检查完整代码和依赖，不得为通过测试擅自修改断言或数值。需要核对最终提交的代码 SHA-256 与结果元数据；修改后的代码不能沿用旧哈希作为新版复现证明。

本地缓存与收益矩阵核对不能独立证明原始下载请求、除权复权参数、完整公司行动台账、历史成分股公告时点和真实成交。换手率为可交易性代理，不等于逐笔成交证据；第三方行情数据必须符合源许可，不能默认原始缓存可公开再分发。论文 Code/Data availability 只能声明**实际已上传且审稿人可访问**的内容，不得虚构仓库链接、复现认证或显著性结果。

## 7. 交稿前检查

请确认：（1）源代码仅有一个确定最终版本；（2）Main/Supplement 的 TeX、PDF、BibTeX 和图一致；（3）正式环境与测试结果如实记录；（4）代码及归档结果的哈希、来源可对应；（5）正文点名的 Holm CSV 和允许共享的输出文件确已随包；（6）上传后再填写真实代码归档标识；（7）审稿回复信的章节与表图定位对应最终稿。

本 README 为复现流程说明，**不代表**已经完成全部独立重算、外部数据认证或真实市场成交验证。


### 最终测试与源码哈希（2026-11-06）

`CODE_SHA256_final.json` 记录本归档中 27 个正式 Python 源码及测试文件的 SHA-256。`validation_final/TEST_SOURCE_SHA256_final.json` 是当时 106/106 项测试通过时的较大范围历史快照，包含本次清理后不再发布的临时修复文件及备份，因此仅作为历史验证证据，不能要求其逐项对应精简压缩包。`validation_final/VALIDATION_final.json` 记录当时测试 106 项全通过、退出码 0，但并不表示重新运行了全部历史实证数据（`real_full_data_rerun: false`）。

本次清理未修改主程序、测试源码或科学结果，仅移除一次性安装脚本、旧备份、过期哈希表和缓存，同时更新 README 引用。如需在精简目录再次核验，请于项目根目录运行 `python -m unittest discover -s . -p "test_*.py" -v`。

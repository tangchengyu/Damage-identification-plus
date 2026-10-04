# 共用数据目录

2026-10-03 将原损伤识别模块内部的 `data/`、`extra_data/` 移到项目根目录，供损伤识别和寿命预测共用。使用当前已核对的元数据，不用备份中的旧 DATA.json 覆盖。

```text
sunshangshibie_tiaozhan/          项目根目录
├── data/                       82 张原始图片及 DATA.json
├── extra_data/                 16 张额外图片及 DATA.json
├── sunshangshibie_tiaozhan/     损伤识别代码、模型与输出
│   ├── src/
│   ├── scripts/
│   ├── configs/
│   └── output/
├── shoumingyuce/                寿命预测代码、模型与输出
│   ├── src/
│   ├── configs/
│   └── output/
├── project.py                  统一启动入口
├── docs/data_audit/             派生清单
└── archive/                    本地历史归档
```

两个模块都有自己的模型和输出，但没有各自复制一套活动数据。历史归档保留旧文件，仅供查阅。文件名中的 `data_`、`extra_data_` 是原记录编码，不是需要替换的绝对路径。

根入口和清单使用项目根目录；独立脚本根据自身文件位置寻找共用目录，不依赖终端当前目录。寿命配置中的 `../data` 相对于寿命模块目录解析；分类路径配置同样以分类模块目录为基准。

在根目录运行：

```powershell
& .\.venv\Scripts\python.exe project.py gui
& .\.venv\Scripts\python.exe project.py audit
& .\.venv\Scripts\python.exe project.py predict --hongguan .\data\boluo\hongguan\train\4052-1.jpg
& .\.venv\Scripts\python.exe project.py lifetime --image .\data\boluo\hongguan\train\4052-1.jpg --sample-id boluo_figure1
& .\.venv\Scripts\python.exe -m unittest scripts.test_audit_dataset scripts.test_shared_paths
& .\.venv\Scripts\python.exe project.py verify
```

迁移前后 98 张图片与两份元数据的 SHA-256 一致；数据清单的状态数量不变。2026-10-04 修复了寿命加载器忽略源视角的问题：分类读取 76 条记录，寿命宏观读取 38 条记录（此前错误映射得到 58 条）。剩余路径缺失、重复寿命与标签来源问题仍须复核。派生清单的 78 张分类可对应图片，不自动替代原加载器。具体修复、训练门槛与解释见 [MODEL_REVIEW.md](MODEL_REVIEW.md)。

历史实验输出中的旧路径在本机保留原文，用于追溯当时运行环境。公开仓库只包含代码与说明，图片、元数据、权重和历史私有输出需要本地配置；目录图描述的是配置完整后的本机结构。本次未重新训练。

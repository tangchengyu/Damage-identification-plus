[简体中文](#简体中文) | [English](#english)

# Damage identification plus

## 简体中文

齿轮表面损伤识别、双视角概率融合、损伤几何测量与图像/结构化属性寿命回归的研究原型。包含 ResNet34 分类、VGG 对比训练脚本及 Tkinter 桌面界面。

**当前状态：本机 CPU 推理与桌面集成已核验；历史准确率和寿命标签未获得独立实验验证。** 本次修复没有重新训练或替换历史模型权重。

### 快速体验

推荐 Python 3.11。Windows Python 通常带有 Tkinter；Linux 需另外配置 Tk 和图形显示环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe project.py gui
```

这是本机桌面窗口，没有网页地址。加载宏观或体视镜照片，再点击“智能识别”。分类无需标定；面积、长度等物理测量需要先标定，且依赖传统阈值与人工操作，不能当作学习型分割模型。

默认使用微调版本，窗口和结果会显示权重版本。要比较原始模型：

```powershell
python project.py gui --weights original
python project.py predict --hongguan path/to/macro.jpg
python project.py predict --hongguan path/to/macro.jpg --tishijing path/to/microscope.jpg --macro-weight 0.3
python project.py predict --hongguan path/to/macro.jpg --weights original
```

两张照片应来自同一被测齿轮和对应区域；代码不能仅凭文件名认证配对。输出类别概率未经校准，不能解释为保证正确的概率。

### 本机模型与数据配置

本仓库发布源代码与说明，**不发布原始图片、原始实验属性数据、训练权重及私有归档**。新克隆尚不能直接进行模型推理。请把可信的本地模型文件放到：

```text
sunshangshibie_tiaozhan/output/models/
  model_hongguan_updated.pth       # 默认宏观模型
  model_tishijing_updated.pth      # 默认体视镜模型
  model_hongguan.pth               # 可选：原始宏观模型
  model_tishijing.pth              # 可选：原始体视镜模型
shoumingyuce/output/models/
  best_lifetime_model.pth          # 可选：历史寿命模型
data/DATA.json                    # 本地元数据
data/<class>/<view>/<split>/*.jpg
extra_data/DATA.json              # 本地额外数据
```

选定版本需要完整的两个分类检查点；缺失会明确报错，不会悄悄改用原始版本。历史寿命检查点含 sklearn 预处理对象，只应加载可信文件。分类检查点哈希见 [checkpoint-audit.json](docs/checkpoint-audit.json)。

寿命可使用自己的属性 JSON，从标准输入传入：

```powershell
Get-Content attributes.json -Raw | python project.py lifetime --image path/to/gear.jpg --attributes-stdin
```

属性名称与单位见 [核查与学习说明](docs/MODEL_REVIEW.md)。本机保留的历史元数据也可通过 `--sample-id boluo_figure1` 演示，输出会标注 `label_provenance_verified=false`。

### 代码布局与验证

```text
project.py                       # 统一命令行/桌面入口
sunshangshibie_tiaozhan/src/      # 分类、加载、训练、评估
sunshangshibie_tiaozhan/utils.py  # 桌面、测量、寿命调用桥接
shoumingyuce/src/                 # 图像/属性寿命回归
scripts/                         # 回归测试、数据审计、检查点核查
docs/                            # 设计核查与验证范围
```

无需数据或权重的回归检查：

```powershell
python -m unittest scripts.test_model_regressions scripts.test_audit_dataset -v
```

有图形显示环境时可执行 `python -m unittest scripts.test_gui_regressions -v`。配置本机完整历史数据后，`python project.py audit` 重建清单；`python project.py verify` 检查恢复的 98 张图片、权重推理和桌面集成。该命令针对本项目历史数据规模，不是任意数据集通用评估器；首次使用需先运行 audit。

分类训练/微调/VGG 基线必须提供经审查的 `val` 分区，测试集只用于最后评估；当前历史数据没有独立验证分区，会明确停止。寿命再训练要求每条记录具备真实的 `specimen_id`、`lifetime_verified=true` 和可追溯的 `lifetime_source`，并按实物分组划分。不要用占位 ID 或随意标记补齐这些字段。训练依赖另外见 `requirements.txt`。

### 方法边界

- 图像与结构化属性融合无需 CLIP；当前三个属性分支分别嵌入并堆叠成 token。
- 图像经过全局池化只剩一个 memory token，交叉注意力无法选择图像区域。
- 图像编码器冻结参数和 BatchNorm 统计；属性分支、投影层与回归头仍训练。
- 双视角默认 `0.3/0.7` 是固定基线。可在独立成对验证集调优，尚未学习融合权重。
- 本机有 98 张图片，但只有 78 张原始图片通过文件/元数据一致性检查。寿命来源、实物分组、近重复及泛化仍需验证。

完整解释、修复证据和下一步实验方案见 [MODEL_REVIEW.md](docs/MODEL_REVIEW.md)；验证范围见 [validation-summary.json](docs/validation-summary.json)。

## English

A research prototype for gear surface damage classification, two-view probability fusion, calibrated geometry measurement, and image/tabular lifetime regression. Includes ResNet34 classifiers, VGG baseline scripts, and a native Tkinter desktop interface.

**Status: local CPU inference and desktop integration checked; historical accuracy and lifetime labels have not been independently validated.** This maintenance update does not retrain or replace historical checkpoints.

### Run

Use Python 3.11. Install `requirements-runtime.txt`, then run `python project.py gui`. Windows Python normally includes Tkinter; Linux needs Tk and a graphical display. This opens a native window, not a website. Load a macro or microscope image and choose intelligent classification. Classification does not require calibration; physical measurements do.

Fine-tuned checkpoints are the default. Use `python project.py gui --weights original` for comparison. Command-line inference supports `--hongguan`, `--tishijing`, `--weights original`, and `--macro-weight 0.3`. Supply paired views from the same inspected specimen; filenames cannot establish physical pairing. Class probabilities are uncalibrated.

### Local assets

Source code and documentation are public. Original images, experimental attribute data, weights, and private archives are excluded. **A fresh clone cannot run inference until checkpoints are provided.** Install the matching macro/microscope pair under `sunshangshibie_tiaozhan/output/models/`, using `model_hongguan_updated.pth` and `model_tishijing_updated.pth` for the default version. Optional original files omit `_updated`. Missing selected files fail explicitly.

Lifetime inference additionally needs `shoumingyuce/output/models/best_lifetime_model.pth`. Its historical serialized preprocessors require a trusted checkpoint. Pass an attribute JSON via standard input with `python project.py lifetime --image path/to/gear.jpg --attributes-stdin`, or use local metadata with `--sample-id`. Predictions explicitly mark lifetime label provenance as unverified. See the Chinese sections above for paths and commands, and [MODEL_REVIEW.md](docs/MODEL_REVIEW.md) for feature keys/units.

### Validation and training

Run `python -m unittest scripts.test_model_regressions scripts.test_audit_dataset -v` without external data or weights. GUI regressions need a local graphical display. `project.py audit` regenerates the local inventory; `project.py verify` is specific to the recovered 98-image dataset and checks both classifier revisions, lifetime inference and desktop integration. It is a smoke check, not an independent accuracy evaluation.

Classifier, fine-tuning and VGG training require an explicitly reviewed validation split. Historical data lack one, so training stops rather than selecting checkpoints on the test set. Lifetime retraining requires verified label sources and real specimen IDs, then separates specimen groups across train/validation/test. Do not invent provenance or group IDs. Additional training dependencies are in `requirements.txt`.

The current lifetime model uses three tabular tokens and one globally pooled image token. CLIP is not required, but cross-attention cannot select spatial regions with a single memory token. The frozen image encoder now keeps BatchNorm statistics fixed. Probability fusion remains a fixed `0.3/0.7` baseline; learned or validation-tuned weights are future experiments requiring reliable paired validation data. See [checkpoint identity checks](docs/checkpoint-audit.json), [validation scope](docs/validation-summary.json), and [method review](docs/MODEL_REVIEW.md).

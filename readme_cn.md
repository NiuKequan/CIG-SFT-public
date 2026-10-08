<div align="center">

# 监督微调中的反事实实例门控

Yuxuan Gu* &emsp; Kequan Niu* &emsp; Xiaocheng Feng &emsp; Yun Li<br>
Yiqiao Yang &emsp; Heyuan Huang &emsp; Bing Qin

</div>

本仓库是 CIG-SFT 论文的正式公开代码发布。作者顺序与论文及 `CITATION.cff` 一致。仓库包含训练实现、训练配方、准备好的数据和数据构建脚本；论文专用的基准评测驱动程序和生成的评测输出不包含在本训练发布中。

## 📰 新闻

* **[2026.10]** 我们已公开发布 CIG-SFT 的训练实现和复现材料。

## 方法

CIG-SFT 使用任务输入为监督微调分配 token 级信用。冻结的参考模型分别在原始任务输入和擦除任务输入的条件下，对相同 teacher-forced 解答进行评分；两种条件下的对数似然差异经过有界变换后得到 token 信用，再通过 KL 正则化投影产生标准交叉熵的标量权重。

实现位于 [`cig_sft_loss_func`](src/llamafactory/train/trainer_utils.py)。集成新增了 `use_cig_sft_loss`、`cig_lambda`、`cig_erased_prompt` 和 `cig_credit_cache` 参数；除非训练配方显式启用，否则这些参数均关闭。擦除输入路径需要冻结参考模型（`finetuning_type` 为 `full` 或 `freeze`），暂不支持 packing 和 streaming dataset。

![CIG-SFT 概览](example/figures/overview.png)

## 安装

测试环境使用 Python 3.11 和 PyTorch 2.6.0（CUDA 12.4）。创建并激活环境后运行：

```bash
git clone https://github.com/NiuKequan/CIG-SFT-public.git
cd CIG-SFT-public
bash example/scripts/setup_env.sh
pip install flash-attn --no-build-isolation
```

## 复现实验训练

已构建数据和清单位于 `example/data/`。如需重新构建数学数据混合，可以运行：

```bash
python example/scripts/build_dataset.py --task numina_cot --sizes 30k
```

运行公开的 1.5B 训练配方：

```bash
python src/train.py example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml
```

CPU 冒烟路径使用仓库中的小规模数据：

```bash
SMOKE_MODEL=/path/to/a/local/model bash example/scripts/smoke_train.sh
```

## 相关仓库

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：用于训练的代码库。
* [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)：用于评测的代码库。

## 引用

```bibtex
@misc{gu2027cigsft,
  title  = {What Should {SFT} Learn? Conditional Information Gain for Generalizable Post-Training},
  author = {Gu, Yuxuan and Niu, Kequan and Feng, Xiaocheng and Li, Yun and Yang, Yiqiao and Huang, Heyuan and Qin, Bing},
  year   = {2027},
  note   = {Preprint}
}
```

论文上传 arXiv 后，请将上面的条目更新为正式的 arXiv 编号。

## 许可证与发布声明

本公开代码以 Apache License 2.0 发布。代码快照用于复现论文中的训练过程；论文专用基准评测程序和生成的评测输出单独维护。软件来源和许可证说明见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

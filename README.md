<div align="center">

# Counterfactual Instance Gating<br>for Supervised Fine-Tuning

Yuxuan Gu* &emsp; Kequan Niu* &emsp; Xiaocheng Feng &emsp; Yun Li<br>
Yiqiao Yang &emsp; Heyuan Huang &emsp; Bing Qin

</div>

This repository is the public code release accompanying the CIG-SFT paper. It contains the training implementation,
recipes, prepared data, and data-building utilities. The paper-specific benchmark evaluation drivers are maintained
separately from this training release.

## 📰 News

* **[2026.10]** We have released the public training implementation and reproducibility materials for CIG-SFT.

## Method

CIG-SFT uses the task input to assign token-level credit during supervised fine-tuning. For each target token, a frozen
reference model scores the same teacher-forced solution with the original task input and with an erased input. The
log-likelihood contrast is converted into a bounded credit score. A KL-regularized projection then produces a scalar
weight for the ordinary cross-entropy term.

The implementation is in [`cig_sft_loss_func`](src/llamafactory/train/trainer_utils.py). The integration adds
`use_cig_sft_loss`, `cig_lambda`, `cig_erased_prompt`, and `cig_credit_cache`; all are disabled unless selected by a
recipe. A frozen reference model is required (`finetuning_type: full` or `freeze`). Packing and streaming datasets are
not supported for the erased-input path.

![CIG-SFT overview](example/figures/overview.png)

## Installation

The tested environment uses Python 3.11 and PyTorch 2.6.0 with CUDA 12.4. Create and activate an environment first,
then run:

```bash
git clone https://github.com/NiuKequan/CIG-SFT-public.git
cd CIG-SFT-public
bash example/scripts/setup_env.sh
pip install flash-attn --no-build-isolation
```

## Reproducing a training run

The built data and manifests are under `example/data/`. Rebuild a math mixture from its recorded public source with:

```bash
python example/scripts/build_dataset.py --task numina_cot --sizes 30k
```

The published 1.5B recipe is:

```bash
python src/train.py example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml
```

The recipe materializes the credit cache before optimization. The CPU smoke path uses the shipped data and a small
number of examples:

```bash
SMOKE_MODEL=/path/to/a/local/model bash example/scripts/smoke_train.sh
```

## Related Repositories

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Codebase used for training.
* [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math): Codebase used for evaluation.

## Citation

```bibtex
@misc{gu2027cigsft,
  title  = {What Should {SFT} Learn? Conditional Information Gain for Generalizable Post-Training},
  author = {Gu, Yuxuan and Niu, Kequan and Feng, Xiaocheng and Li, Yun and Yang, Yiqiao and Huang, Heyuan and Qin, Bing},
  year   = {2027},
  note   = {Preprint}
}
```

Please update the citation with the assigned arXiv identifier when the manuscript is uploaded.

## License and release statement

This public release is distributed under the Apache License 2.0. The code snapshot is intended to reproduce the training
procedure described in the paper; paper-specific benchmark evaluators and generated evaluation outputs are maintained
separately. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for software provenance and
license notices.

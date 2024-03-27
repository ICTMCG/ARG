# ARG

Official repository for ["**Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**"](https://arxiv.org/abs/2309.12247), which has been accepted by AAAI 2024.

## Dataset

The experimental datasets where can be seen in `data` folder. Note that you can download the datasets only after an ["Application to Use the Datasets from ARG for Fake News Detection"](https://forms.office.com/r/eZELqSycgn) has been submitted.

## Code

### Requirements

- python==3.10.13

- CUDA: 11.3

- Python Packages:

  ```
  pip install -r requirements.txt
  ```

### Pretrained Models

You can download pretrained models ([bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)) and change paths (`bert_path`) in the corresponding scripts.

### Run

You can run this model through `run_zh.sh` for chinese dataset and `run_en.sh` for english dataset. 

## How to Cite

```
@article{hu2023bad,
  title={Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection},
  author={Hu, Beizhe and Sheng, Qiang and Cao, Juan and Shi, Yuhui and Li, Yang and Wang, Danding and Qi, Peng},
  journal={arXiv preprint arXiv:2309.12247},
  year={2023}
}
```
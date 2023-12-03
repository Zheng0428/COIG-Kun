# Kun: Answer Polishment Saves Your Time for Using Intruction Backtranslation on Self-Alignment

## Introduction

Inspired by Meta's mid-year paper "Self-Alignment with Instruction Backtranslation," our project embarks on an innovative approach to enhance language model training using a novel data augmentation paradigm. This method, rooted in the principles of self-alignment, involves a meticulous process of selecting, refining, and employing high-quality instructional data to fine-tune language models, significantly improving their performance on benchmarks like the Alpaca leaderboard.

## Methodology

Our approach closely follows the self-alignment method described by Meta, with adaptations to optimize the process:

1. **Seed Data Selection and Model Training**: Initially, appropriate seed data are selected and inverted to train a label model (Myx) on a base model. Concurrently, using the same seed data, a preliminary chat model (M0) is trained following the Supervised Fine-Tuning (SFT) method typical of chat models.

2. **Labeling Unlabeled Data**: The label model Myx is then used to annotate preliminarily cleansed unlabeled data. Cleansing involves filtering based on perplexity (ppl) and length, discarding data exceeding 512 tokens.

3. **Instruction Data Generation**: Post-annotation, we obtain our first version of instruction data. Unlike the original project where both instruction and output data pairs are fed into M0 for scoring, our replication revealed limitations in M0's ability to discern high-quality instructions. We innovated by scoring only the instruction component, effectively filtering out noise and selecting high-quality instructions.

4. **Output Data Refinement**: Upon manual inspection, we identified a mismatch between the unlabeled data (used as output) and the standard requirements for output in instruction data. To address this, we introduced an additional step: refining the output data. Using M0's capabilities, the output (originally unlabeled data) is adjusted according to the instructions, making it more suitable as output for the instruction data.

5. **Framework Completion**: Our methodology concludes with the acquisition of a substantial volume of instructional data, achieved with minimal resource expenditure.


![Project Framework](Kun_white.Jpeg)

## Results

Our refined approach yielded a model that demonstrates superior performance on the Alpaca leaderboard, outperforming non-distilled models like LIMA, Claude, and Guanaco. This was achieved through two iterations of dataset refinement and fine-tuning on the LLaMa model, resulting in our enhanced model, Humpback.

## Contributions

This project contributes to the field of language model training by:

- Proposing a novel approach to generate and refine instructional data.
- Demonstrating the effectiveness of selective scoring on instruction data for quality enhancement.
- Offering a resource-efficient methodology for data augmentation.
- Provision of a directly usable Chinese instruction generation model.

## Future Work

We plan to explore further refinements in data selection and scoring methods, aiming to enhance the quality of the generated instructional data even more. Additionally, adapting this methodology to other language models and contexts remains an area of interest.

## Citations

To cite our project in your work, please use the following BibTeX entry:

```bibtex
@misc{COIG-Kun,
  title={Kun: Answer Polishment Saves Your Time for Using Intruction Backtranslation on Self-Alignment},
  author={Tianyu, Zheng* and Shuyue, Guo* and Xingwei, Qu and Xinrun, Du and Wenhu, Chen and Jie, Fu and Wenhao, Huang and Ge, Zhang},
  year={2023},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={https://github.com/Zheng0428/COIG-Kun}
}
```

For referencing the Humpback model as described in "Kun", use:

```bibtex
@article{li2023self,
  title={Self-alignment with instruction backtranslation},
  author={Li, Xian and Yu, Ping and Zhou, Chunting and Schick, Timo and Zettlemoyer, Luke and Levy, Omer and Weston, Jason and Lewis, Mike},
  journal={arXiv preprint arXiv:2308.06259},
  year={2023}
}
```

For referencing the datasets as described in "Kun", use:

```bibtex
@misc{he2023wanjuan,
      title={WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models}, 
      author={Conghui He and Zhenjiang Jin and Chao Xu and Jiantao Qiu and Bin Wang and Wei Li and Hang Yan and Jiaqi Wang and Dahua Lin},
      year={2023},
      eprint={2308.10755},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model}, 
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@article{YUAN202165,
title = {WuDaoCorpora: A super large-scale Chinese corpora for pre-training language models},
journal = {AI Open},
volume = {2},
pages = {65-68},
year = {2021},
issn = {2666-6510},
doi = {https://doi.org/10.1016/j.aiopen.2021.06.001},
url = {https://www.sciencedirect.com/science/article/pii/S2666651021000152},
author = {Sha Yuan and Hanyu Zhao and Zhengxiao Du and Ming Ding and Xiao Liu and Yukuo Cen and Xu Zou and Zhilin Yang and Jie Tang},
keywords = {Pre-trained language models, Chinese corpus, Transformer-XL},
abstract = {Using large-scale training data to build a pre-trained language model (PLM) with a larger volume of parameters can significantly improve downstream tasks. For example, OpenAI trained the GPT3 model with 175 billion parameters on 570 GB English training data, enabling downstream applications building with only a small number of samples. However, there is a lack of Chinese corpus to support large-scale PLMs. This paper introduces a super large-scale Chinese corpora WuDaoCorpora, containing about 3 TB training data and 1.08 trillion Chinese characters. We also release the base version of WuDaoCorpora, containing about 200 GB training data and 72 billion Chinese characters. As a baseline, we train a model transformer-XL with 3 billion parameters on the base version to test the corpora's effect. The results show that the models trained on this corpora can achieve excellent performance in Chinese. The data and model are available at https://data.wudaoai.cn and https://github.com/THUDM/Chinese-Transformer-XL, respectively.}
}
```



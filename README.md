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
  author={Tianyu, Zheng* and Shuyue, Guo* and Xingwei, Qu and Wenhu, Chen and Jie, Fu and Wenhao, Huang and Ge, Zhang},
  year={2023},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={https://github.com/Zheng0428/COIG-Kun}
}
```

For referencing the Humpback model as described in "Self-Alignment with Instruction Backtranslation," use:

```bibtex
@article{li2023self,
  title={Self-alignment with instruction backtranslation},
  author={Li, Xian and Yu, Ping and Zhou, Chunting and Schick, Timo and Zettlemoyer, Luke and Levy, Omer and Weston, Jason and Lewis, Mike},
  journal={arXiv preprint arXiv:2308.06259},
  year={2023}
}
```



<h1 align="center"><span style="font-weight:normal">The Strawberry Problem 🍓 <br> Emergence of Character-level Understanding in Tokenized Language Models</h1>

<div align="center">

[Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en), [Stefan Ruseti](https://scholar.google.com/citations?user=aEyJTykAAAAJ&hl=en), [Emilian Radoi](https://scholar.google.com/citations?user=yjtWIf8AAAAJ&hl=en), [Mihai Dascalu](https://scholar.google.ro/citations?user=3L9yY8UAAAAJ&hl=en)
</div>

<div align="center">
  
[📜 Paper PDF](https://arxiv.org/abs/2505.14172)|
[📘 Abstract](#intro)|
[⚒️ Usage](#usage)|
[📖 Citation](#citation)|
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Abstract
Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge slowly, suddenly, and only late in training. We further show that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available.

## <a name="usage"></a> ⚒️ Usage

Go to `cd experiments/` and run: 

### Step 1 - Generate vocabularies

```
    bash generate_datasets.sh
```

### Step 2 - Train the models

```
    bash train.sh
    bash wiki_train.sh
```

### Step 3 - Perform ablation studies 

```
    bash ablation.sh
```

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

```
@misc{cosma2025strawberry,
      title={The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models}, 
      author={Adrian Cosma and Stefan Ruseti and Emilian Radoi and Mihai Dascalu},
      year={2025},
      eprint={2505.14172},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14172}, 
}
```

## <a name="license"></a> 📝 License

This work is protected by [Attribution-NonCommercial 4.0 International](LICENSE)

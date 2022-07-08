<div id="top"></div>

# SPOBET
Pose-based word-level sign language recognition with BERT-style transformer

<!-- ABOUT THE PROJECT -->
## About The Project

  This repository implements, using Keras, a pose-based, word-level sign language recognition with BERT-style transformer.

- Model is trained on WLASL 2D pose data, on the ASL100 split. https://github.com/dxli94/WLASL
- Model is built with Keras layers; highly transferable and configurable 
- Comparable* accuracy levels achieved on the ASL100 split as compared to other pose-based word-level sign language recognition models

Further details on the implementation and results discussion can be found in https://medium.com/@kennethong.ai/spobet-d9d952836c48

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Using the Repo -->

## Using the Repo

### Getting Started


<br/>

### Training

<br/>

### Evaluation

<br/>

### Inferencing

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Trained Weights -->

## Trained Weights

|                 | Top 1 | Top 5 | Top 10 |
|-----------------|-------|-------|--------|
| [SPOBET (ASL100)](https://drive.google.com/file/d/18X35zpWx7rTnWz2m1EqAc2SpwIlaEKgV/view?usp=sharing) |63.95% | 87.98%| 91.86% |

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

The code is published under the Apache License 2.0.

The accompanying data of the WLASL dataset used for training and experiments are, however, allow only non-commercial usage. This, therefore, extends the terms of non-commmerical usage to the uploaded model weights and its derivatives.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- References -->
## References

- [WLASL: A large-scale dataset for Word-Level American Sign Language (WACV 20' Best Paper Honourable Mention)](https://dxli94.github.io/WLASL/)
- [SPOTER: Sign Pose-based Transformer](https://github.com/matyasbohacek/spoter)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)
- [Explained: Attention Visualization with Attention Rollout](https://storrs.io/attention-rollout)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

<p align="right">(<a href="#top">back to top</a>)</p>

<div id="top"></div>

# SPOBET
Pose-based word-level sign language recognition with BERT-style transformer in Keras

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

1. Clone this repository 
2. Install the required packages using the requirements.txt. 
3. Download the dataset from the [WLASL website](https://github.com/dxli94/WLASL). We just need the keypoints files and the split files.
4. Place the keypoint folders in dataset/annotations and the split files in dataset.
5. The model, dataset and training parameters are controlled by the config files found in the config folder

<br/>

### Training

In the root folder, run

```
python main.py --run train
```
- Tensorboard logs will be saved in the logs directory. 
- Masked encoder weights will be saved as weights/pretrain. 
- Model weights will be saved as weights/spobet

<br/>

### Evaluation

In the root folder, run

```
python main.py --run evaluation
```

- The accuracy scores for the Top 1, Top 5 and Top 10 will be printed at the end.

<br/>

### Inferencing

This repo does not include the implementation of OpenPose to retrieve the keypoints needed for inferencing. To do inferencing, you will need to:

1. Retrieve keypoints usng OpenPose
2. Format the results similar to those in WLASL
3. Create a "split" file with the neccesary information. I.e. video_id (annotation folder must be of the same name), start_frame and end_frame (each annotation file is named according to the frame number), and the train/test split be equals to "test"
4. In dataconfig.cfg, set SHOW_RES = 1. This will print out the inference results at the end of evaluation. The res is shown as a list of predicted labels, from the lowest probability to the highest. I.e the label with the highest probability is res[-1].
4. Run evaluation as per normal 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Trained Weights -->

## Trained Weights

|                 | Top 1 | Top 5 | Top 10 |
|-----------------|-------|-------|--------|
| [SPOBET (ASL100)](https://drive.google.com/file/d/18X35zpWx7rTnWz2m1EqAc2SpwIlaEKgV/view?usp=sharing), [BERT encoder](https://drive.google.com/file/d/1ygqc3yVcLtS5d8_NFiHKhL-HuFplJjkM/view?usp=sharing) |63.95% | 87.98%| 91.86% |

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

The code is published under the Apache License 2.0.

The accompanying data of the WLASL dataset used for training and experiments, however, allow only non-commercial usage. This, therefore, extends the terms of non-commmerical usage to the uploaded model weights and its derivatives.

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

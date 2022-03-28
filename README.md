## Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation
This is a PyTorch implementation of [Future Data Helps Training: Modelling Future Contexts for Session-based Recommendation](https://arxiv.org/pdf/1906.04473.pdf).

Session-based recommender systems have attracted much attention recently. To capture the sequential dependencies, existing methods resort either to data augmentation techniques or left-to-right style autoregressive training. Since these methods are aimed to model the sequential nature of user behaviors, they ignore the future data of a target interaction when constructing the prediction model for it. However, we argue that the future interactions after a target interaction, which are also available during training, provide valuable signal on user preference and can be used to enhance the
recommendation quality.

Properly integrating future data into model training, however, is non-trivial to achieve, since it disobeys machine learning principles and can easily cause data leakage. To this end, we propose a new encoder-decoder framework named Gap-filling based Recommender (GRec), which trains the encoder and decoder by a gap-filling mechanism. Specifically, the encoder takes a partially-complete session sequence (where some items are masked by purpose) as input, and the decoder predicts these masked items conditioned on the encoded representation. We instantiate the general GRec framework using convolutional neural network with sparse kernels, giving consideration to both accuracy and efficiency. We conduct experiments on two real-world datasets covering short-, medium-, and longrange user sessions, showing that GRec significantly outperforms the state-of-the-art sequential recommendation methods. More empirical studies verify the high utility of modeling future contexts
under our GRec framework.


## Implementation
NextItNet pytorch version: https://github.com/syiswell/NextItNet-Pytorch, of which architecture is useful to this work.

### Requirements
- Python 3.9, CPU or NVIDIA GPU
- PyTorch 1.11.0


## Run GRec
The hyperparameters used to train the GRec model are set default in the `argparse.ArgumentParser` and `model_para` in the `GRecTorch.py` file, you can change them if needed. Then simply run `python GRecTorch.py`.


## Reference
    @inproceedings{yuan2020future,
      title={Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation},
      author={Yuan, Fajie and He, Xiangnan and Jiang, Haochuan and Guo, Guibing and Xiong, Jian and Xu, Zhezhao and Xiong,   Yilin},
      booktitle={Proceedings of The Web Conference 2020},
      pages={303--313},
      year={2020}
    }




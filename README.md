# Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning
This repository is the official pytorch implementation of [Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning](https://openreview.net/pdf?id=a4DbIONcpb).

## Experiments
### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed.

To install requirements, please run:
```
pip install -r requirements.txt
```

### Pretraining
We provide pretrained dynamics and diffusion models in `Dynamics_models/` and `Diffusion_models/`.

You can also pretrain the models from scratch.  
For example, to pretrain the dynamics model on `halfcheetah-medium-v2` dataset, run:
```bash
python pretrain_dynamics.py --env_name halfcheetah-medium-v2
```

To pretrain the diffusion model, run:
```bash
python pretrain_diffusion.py --env_name halfcheetah-medium-v2
```

### Training
Below we use `halfcheetah-medium-v2` dataset as an example:
```bash
python main.py --env_name halfcheetah-medium-v2 --device 0 --seed 0
```

To ensure reproduction of the reported results in our paper, the optimal hyperparameters for DOSER are pre-configured in `main.py`. Users are welcome to fine-tune these settings or implement their own configurations to explore beyond the current results.

## Citation
If you find this code useful, please cite our paper:
```
@inproceedings{wangbeyond,
  title={Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning},
  author={Wang, Qingjun and Zhou, Hongtu and Yu, Hang and Zhao, Junqiao and Zhao, Yanping and Ye, Chen and Wang, Ziqiao and Chen, Guang},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
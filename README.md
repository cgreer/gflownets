# gflownets
A Basic GFlowNet Setup

![Training Dash](./img/training_dash.png)

## Overview

`gflownets` implements a GFlowNet using the trajectory balance loss on the [Smiley Environment](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR).

Includes some basic goodies that were recommended at the [GFlowNet Workshop](#resources):
- Off-policy training via dithering (tempering + eps-greedy)
- Gradient clipping
- Monitoring
  - Pf/Pb Entropy
  - ||Gradient||
  - Rewards (avg/max)

See [torchgfn](#resources) and [gflownet](#resources) for mature libraries.

<br></br>
## Install

### Clone Repo

    git clone git@github.com:cgreer/gflownets.git

### Create Virtual Environment

    python3 -m venv gfn

### Install Packages

Activate virtual environment:

    source gfn/bin/activate

Install requirements:

    pip install -r requirements.txt

<br></br>
## Train on Smiley Environment

    python train_smiley.py

After training completes it will run the evaluation analysis and show the training dashboard:

![Training Dash](./img/training_dash.png)
![Smiley Eval](./img/smiley_eval.png)

If training ran correctly, then smiley faces should be sampled proportional to their reward (~66% smiley) and the estimate for Z should be ~12.


<a name="resources" />

<br></br>
## Resources
- [GFlowNet Smiley Tutorial (Amazing Resource!)](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR)
- [Mila GFlowNet Workshop: Day 1](https://youtu.be/HHwhQx7W8jg?t=2776)
- [Mila GFlowNet Workshop: Day 2](https://youtu.be/wYrZrPsm2NM?t=1510)
- [Mila GFlowNet Workshop: Day 3](https://youtu.be/tMVJnzFqa6w?t=1177)
- [torchgfn](https://github.com/GFNOrg/torchgfn)
- [gflownet](https://github.com/alexhernandezgarcia/gflownet)
- [Curated List of GFlowNet Resources](https://github.com/zdhNarsil/Awesome-GFlowNets)

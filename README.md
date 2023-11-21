# gflownets
A basic GFlowNet Setup

Implements a GFlowNet using the trajectory balance loss on the [Smiley Environment](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR) + additional invalid action masking.

Includes some basic goodies that were recommended at the [GFlowNet Workshop](#resources):
- Off-policy training via dithering (tempering + eps-greedy)
- Gradient clipping
- Monitoring
  - Pf/Pb Entropy
  - ||Gradient||
  - Rewards (avg/max)

See [torchgfn](#resources) and [gflownet](#resources) for more mature libraries.

## Install

### Clone Repo

    git clone git@github.com:cgreer/gflownets.git

### Create Virtual Environment

    python3 -m venv gfn

### Install Packages

    source gfn/bin/activate
    pip install -r requirements.txt

## Train on Smiley Environment

    python train_smiley.py

After training completes it will run the evaluation analysis and show the training dashboard. If all goes well, then smiley faces should be samples proportional to their reward (~66% smiley) and the estimate for Z should be ~12.

EVAL_IMAGE (todo)
DASHBOARD_IMAGE (todo)

<a name="resources" />

## Resources
- [GFlowNet Smiley Tutorial](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR)
- [Mila GFlowNet Workshop: Day 1](https://youtu.be/HHwhQx7W8jg?t=2776)
- [Mila GFlowNet Workshop: Day 2](https://youtu.be/wYrZrPsm2NM?t=1510)
- [Mila GFlowNet Workshop: Day 3](https://youtu.be/tMVJnzFqa6w?t=1177)
- [torchgfn](https://github.com/GFNOrg/torchgfnTorchGFN)
- [gflownet](https://github.com/alexhernandezgarcia/gflownet)
- [Curated List of GFlowNet Resources](https://github.com/zdhNarsil/Awesome-GFlowNets)

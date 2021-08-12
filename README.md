# DiffAqua: A Differentiable Computational Design Pipeline for Soft Underwater Swimmers with Shape Interpolation

![teaser](assets/teaser.png)

[Pingchuan Ma](https://pingchuan.ma/),
[Tao Du](https://people.csail.mit.edu/taodu/),
[John Z. Zhang](http://bnc.mit.edu/),
[Kui Wu](https://people.csail.mit.edu/kuiwu/),
[Andrew Spielberg](http://www.andrewspielberg.com/),
[Robert K. Katzschmann](https://srl.ethz.ch/the-group/prof-robert-katzschmann.html),
and
[Wojciech Matusik](http://people.csail.mit.edu/wojciech/)

**SIGGRAPH 2021**
[[Project Page]](http://diffaqua.csail.mit.edu/)
[[Paper]](https://arxiv.org/abs/2104.00837)
[[Video]](https://youtu.be/fVUfXIJEQDg)

```text
@article{ma2021diffaqua,
  title={DiffAqua: A Differentiable Computational Design Pipeline for Soft Underwater Swimmers with Shape Interpolation},
  author={Ma, Pingchuan and Du, Tao and Zhang, John Z and Wu, Kui and Spielberg, Andrew and Katzschmann, Robert K and Matusik, Wojciech},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={4},
  pages={132},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```

## Get Started

### Prerequistes

- Ubuntu 18.04 LTS
- CUDA 10.2
- Anaconda 2020.11

    ```sh
    wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    sh /tmp/Anaconda3-2020.11-Linux-x86_64.sh
    ```

- Suitesparse

    ```sh
    # probably need `sudo`
    apt-get install libsuitesparse-dev
    ```

### Installation

- Restore Anaconda environment

    ```sh
    conda env create -f environment.yml
    conda activate diffaqua
    ```

- Install `diffpd`

    ```sh
    cd /path/to/root/external/diffpd/
    pip install -e . -v
    ```

- Install `diffaqua`

    ```sh
    cd /path/to/root/
    pip install -e . -v
    ```

### Run Experiment

- Open-loop co-optimization

    ```sh
    cd /path/to/root/example/
    python openloop.py
    ```

    More to come!

- Visualize the results using `tensorboard`

    ```sh
    cd /path/to/root/example/experiments/
    tensorboard --logdir . --port 8888
    ```

## Contact

If you have any questions about the paper or the codebase, please feel free to contact [pcma@csail.mit.edu](mailto:pcma@csail.mit.edu).

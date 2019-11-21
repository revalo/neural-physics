# Playground

Attempt to be sample efficient in learning intuitive physics by treating the environment
as a playground.

## Vanilla NPE

[Pre-Trained Model Zoo](https://www.dropbox.com/sh/bfxlaqa9uz88mg1/AADh_zUYml83lTLD3y0R4xnha?dl=0)

This is a recreation of Chang et. al.'s neural physics engine. The main driver file is
`npe_main.py`.

Generate and train as,

```
python npe_main.py --gen_data --dataset PATH_TO_DATASET
python npe_main.py --train --dataset PATH_TO_DATASET --model PATH_TO_MODEL
python npe_main.py --model_simulation --model PATH_TO_MODEL
```

Of course, one can also just show the actual chipmunk simulation,

```
python npe_main.py --show_world
```

# Video Maker - tool for creating video frames, written in Python.

### Requirements

Install environment with [Anaconda](https://www.continuum.io/downloads):

```sh
conda env create -f environment.yml
```

The environment should be listed via `conda info --envs`:

```sh
# conda environments:
#
gtcdsh       /usr/local/anaconda3/envs/gtcdemo
root          *  /usr/local/anaconda3
```

Further documentation on [working with Anaconda environments](https://conda.io/docs/using/envs.html#managing-environments). 

Particularly useful sections:

https://conda.io/docs/using/envs.html#change-environments-activate-deactivate
https://conda.io/docs/using/envs.html#remove-an-environment

### Video Maker

Generates frames, frame-by-frame, as a means of video generation. Generated frames can be saved to a folder,
or used in a pipeline for direct video generation.

to_frames and from_frames - are two utilities that allow for splitting/creating videos into/from frames.

See VideoMakerRunner.ipynb for tests/examples how to use

Effects include: text generation, splitting screens, cuts-in, zooms-in and out, drawing, copying from folders and images, and some more.

You can have scenarios, sub-scenarios, or use python generators (the latter allows fluent effect application).

Still a lot of work to do - if I ever have time.


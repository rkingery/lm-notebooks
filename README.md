# Language Modeling from Scratch: Notebook Edition

This repository contains my conversion of the course assignments from the [Stanford CS336](https://stanford-cs336.github.io/spring2025/) *Language Models from Scratch* course into an open Jupyter Notebook format. If you wish to follow along with the original course lectures, those are on YouTube [here](https://youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&si=2W7K7Lgpg9Xr4s7M). Currently the 2025 edition of this course is supported, though the 2024 and any later 2026+ editions are unlikely to differ significantly, and can be easily adapted as desired.

## Rationale

In the original CS336 course, each assignment was its own self-contained GitHub repository containing any instructions, source code, data, and tests needed for *Stanford students* to complete that particular assignment. What I've attempted to do here is take all of those original assignments across all repositories and consolidated them into this single repository, and to remove or modify anything specific to Stanford students so that anyone with access to a GPU can do these assignments with minimal friction while following along with the main course lectures.

Most importantly, all of the original assignment PDFs were converted (mostly by hand) into a markdown format, which I then inserted into Jupyter notebooks broken up by topic. This new structure reflects the fact that, in my opinion, notebooks are much more natural for coding assignments like this than sprawled out projects full of scripts and documentation. With notebooks we can seemlessly integrate the assignment instructions together with the assignment code itself so the learner can do as much in one place as possible.

Since some of the original assignments are much longer than others, when converting each assignment into notebook format I made the decision to split each assignment topic into its own separate notebook where possible. This hopefully makes getting through each assignment a bit more bearable.

The original assignment repositories also contained test suites to test students' code. I wanted to integrate these tests into the notebooks as well so learners could test their code immediately after implementing it to get feedback. To achieve that, I've removed all PyTest specific functionality to make the tests easily notebook executable.

Finally, the original assignment repositories contained a bunch of course-specific setup configurations. I've largely ditched these in favor of simple pip install commands embedded directly inside each notebook. This way, any learner going through a specific notebook assignment can setup everything needed without having to leave the notebook. Of course, it is recommended to create a virtual environment for each notebook (e.g. using uv or conda), but I leave that decision up to the user.

## Structure

Currently, the relationship between the original assignments and the notebooks in this repository are:
- [Assignment 1](https://github.com/stanford-cs336/assignment1-basics/tree/main): Building a Transformer LM
    - `1a-tokenization.ipynb`: The tokenizer implementation part of this assignment.
    - `1b-transformer.ipynb`: The transformer architecture implementation part of this assignment.
    - `1c-training.ipynb`: The optimizer and training loop implementation part of this assignment.
    - `1d-experiments.ipynb`: The training experiments part of this assignment.
- [Assignment 2](https://github.com/stanford-cs336/assignment2-systems/tree/main): Systems and Parallelism
    - `2a-single-gpu-optimization.ipynb`: The single GPU optimization part of this assignment.
    - `2b-distributed-training.ipynb`: The distributed training part of this assignment.
- [Assignment 3](https://github.com/stanford-cs336/assignment3-scaling/tree/main): Scaling Laws
    - `3a-scaling-laws.ipynb`: The entire assignment on scaling law experiments.
- [Assignment 4](https://github.com/stanford-cs336/assignment4-data/tree/main): Filtering Language Modeling Data
    - `4a-data-filtering.ipynb`: The entire assignment on data filtering.
- [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment): Alignment and Reasoning RL, Instruction Tuning and RLHF
    - `5a-alignment.ipynb`: The SFT and reasoning component of this assignment.
    - `5b-instruction-tuning.ipynb`: The instruction tuning and RLHF component of this assignment.

*Note:* All notebooks mentioned above are in the `notebooks/` directory.

## Requirements

*Hardware:* Most of these notebooks will require access to an Nvidia GPU, preferably one or more H100 GPUs. If you don't have access to a GPU you'll find your ability to complete most of these notebooks difficult.

*Data:* Some notebooks will require access to certain open text datasets, in particular the TinyStories and OpenWebText datasets. Any datasets we use will either be provided in the `data/` directory, or we'll provide instructions on how to download them from the Web as needed.

*Packages:* We'll make extensive use of PyTorch throughout this course. The Assignment 2 notebooks will also make extensive use of Triton for GPU optimizations. The Assignment 5 notebooks will also make use the Huggingface Transformers library. Other than these, the packages you'll need are pretty standard for any machine learning project: Numpy, Scipy, Matplotlib, Pandas, Regex, Scikit Learn, etc.
## Install
Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## Dataset and Pretrain Model
This project runs with a specific dataset and a series of pretrained models to function properly. Please download the dataset and models from the following link and save it to a local folder:

[Download Dataset and pretrained model](https://drive.google.com/drive/folders/111cncohSHP6_y6Gucg7Prpxfpr4U8DvU?usp=sharing)

Ensure that you have downloaded the entire dataset and models and place the data files in the correct folder as specified in the project instructions.

## Quick Start

Follow these steps to quickly start using the pre-trained `BatteryGPT` model to predict battery status with the provided Python script:

1. Ensure that you have downloaded and properly set up the dataset as described in the [Dataset section](#dataset).

2. Clone the repository to your local machine:

3. Navigate to the repository directory:

4. Install any necessary dependencies:

5. Run the script to load the pre-trained `BatteryGPT` model and get battery status predictions:
```
$ python sample.py
```

## Acknowledgements

This project makes use of code from the [nanoGPT](https://github.com/karpathy/nanoGPT) by [karpathy]. Specifically, we adapted the implementation of GPT model to enhance our application. We appreciate the efforts of the original authors and recommend checking out their work for its robust features and excellent documentation.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file included with this repository.
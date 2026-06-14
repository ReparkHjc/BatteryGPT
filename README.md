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

TODO List

We are actively working on the next updates of this project. The following items will be released in future versions:

* Release the corrected and updated dataset and code to avoid potential misunderstandings caused by earlier synchronized versions.
* Update the pretrained models with newly organized data and improved training settings.
* Introduce BatteryGPT v2, which incorporates additional external feature embedding schemes beyond the original V, I, and T signal representations.
* Improve the embedding strategy for voltage, current, temperature, and other battery-related signals to better capture multi-dimensional battery degradation patterns.
* Integrate BatteryGPT with BatteryLife for a more complete battery modeling and prediction workflow.
* Incorporate more battery datasets to support broader cross-dataset evaluation and improve generalization across different battery chemistries, protocols, and operating conditions.
* Provide clearer documentation, examples, and scripts for data preprocessing, model training, inference, and evaluation.

## Acknowledgements

This project makes use of code from the [nanoGPT](https://github.com/karpathy/nanoGPT) by [karpathy]. Specifically, we adapted the implementation of GPT model to enhance our application. We appreciate the efforts of the original authors and recommend checking out their work for its robust features and excellent documentation.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file included with this repository.

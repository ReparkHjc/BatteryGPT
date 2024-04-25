## install
Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## Dataset
This project requires a specific dataset to function properly. Please download the dataset from the following link and save it to a local folder:

[Download Dataset](https://drive.google.com/drive/folders/111cncohSHP6_y6Gucg7Prpxfpr4U8DvU?usp=sharing)

Ensure that you have downloaded the entire dataset and place the data files in the correct folder as specified in the project instructions.

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

This project makes use of code from the [nanoGPT](https://github.com/karpathy/nanoGPT) by [karpathy]. Specifically, we adapted the implementation of [specific functionality] to enhance our application. We appreciate the efforts of the original authors and recommend checking out their work for its robust features and excellent documentation.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file included with this repository.

### MIT License

The MIT License is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don’t hold you liable.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

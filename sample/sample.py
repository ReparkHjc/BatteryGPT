"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = 'out-picture-char'  # ignored if init_from is not 'resume'
# out_dir = 'out-battery'  # ignored if init_from is not 'resume'
# out_dir = 'out-battery-v1'  # ignored if init_from is not 'resume'
out_dir = 'out-battery-46'  # ignored if init_from is not 'resume'
# start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = [892, 112, 117, 126, 680, 790, 591, 12, 219, 437, 872, 66, 956, 868, 129, 217, 578, 671, 759, 887, 591, 783,
#          109, 974, 499, 61, 553, 473, 614, 453, 125, 672, 823, 582, 914, 618, 790, 1008, 778, 253, 12, 256, 977, 297,
#          759, 192, 899, 913, 790, 714, 256, 823, 661, 591, 869, 546, 471, 483, 163, 287, 343, 92, 941, 148, 609, 31,
#          783, 546, 857, 600, 432, 977, 914, 256, 669, 726, 503, 868, 176, 749, 414, 925, 972, 389, 402, 656, 551, 666,
#          681, 586, 6, 125, 914, 564, 407, 406, 812, 40, 894,
#          719]  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = [362, 526, 208, 830, 893, 586, 473, 780, 546, 389, 373, 598, 591, 789, 193, 580, 586, 772, 642, 243, 811, 276,
#          642, 627, 389, 706, 193, 789, 35, 119, 1023, 583, 636, 339, 623, 66, 771, 464, 627, 720, 737, 459, 35, 672,
#          505, 965, 660, 1012, 666, 313, 719, 771, 662, 627, 790, 712, 122, 790, 583, 163, 263, 297, 580, 433, 610, 1002,
#          892, 334, 253, 479, 61, 591, 112, 617, 112, 737, 245, 580, 642, 924, 301, 599, 192, 903, 389, 297, 193, 591,
#          175, 680, 899, 94, 680, 680, 800, 789, 780, 600, 539, 27, 706, 219, 1023, 656, 225, 857, 333, 707, 260, 840,
#          758, 644, 477, 653, 166, 269, 574, 869, 402, 112, 849, 897, 587, 772, 870, 996, 473, 985, 539, 345, 611, 805,
#          611, 726, 66, 129, 119, 406, 653, 787, 899, 783, 256, 593, 518, 339, 339, 518, 339, 72, 168, 313, 965, 910,
#          894, 805, 845, 36, 982, 918, 526, 202, 526, 918, 638, 803, 666, 325, 828, 66, 455, 416, 946, 469, 148, 202,
#          148, 202, 148, 868, 737, 459, 237, 129, 325, 600, 595, 154, 593, 988, 688, 49, 49, 988, 250, 517, 143, 354,
#          669, 617, 514, 915, 382, 433, 749, 250, 250, 250, 250, 325, 698, 941, 551, 607, 363, 595, 961, 518, 433, 556,
#          556, 556, 688, 556, 325]
# start = [813, 402, 746, 404, 599, 870, 404, 979, 761, 404, 884, 884, 404, 404, 884, 254, 92, 870, 979, 608, 745, 887,
#          797, 608, 587, 329, 329, 329, 574, 276, 761, 913, 772, 810, 168, 185, 35, 628, 759, 389, 546, 514, 1007, 136,
#          590, 572, 202, 970, 597, 657, 185, 1017, 316, 306, 528, 528, 107, 915, 610, 164, 748, 684, 671, 258, 250, 572,
#          913, 564, 110, 122, 185, 915, 772, 759, 498, 597, 1021, 473, 515, 520, 772, 988, 444, 63, 663, 632, 872, 587,
#          479, 731, 473, 96, 593, 873, 189, 626, 437, 985, 541, 143, 821, 63, 585, 598, 520, 520, 746, 7, 718, 913, 772,
#          725, 870, 661, 821, 761, 329, 966, 255, 597, 550, 333, 712, 936, 726, 306, 462, 707, 914, 897, 12, 400, 84,
#          656, 435, 82, 932, 432, 187, 244, 966, 469, 730, 979, 587, 373, 417, 365, 462, 414, 988, 172, 49, 402, 80, 787,
#          974, 936, 418, 166, 493, 821, 425, 662, 769, 426, 660, 44, 589, 794, 982, 749, 44, 250, 72, 247, 6, 343, 523,
#          172, 330, 192, 375, 4, 49, 551, 188, 617, 539, 92, 714, 719, 577, 6, 813, 913, 147, 1022, 473, 202, 845, 471,
#          638, 316, 147, 147, 541, 414, 258, 313, 315, 731, 1008, 545, 1008, 545, 558, 132, 132, 132, 132, 452, 132, 774,
#          774]
# start = [362, 526, 208, 830, 893, 586, 473, 780, 546, 389, 373, 598, 591, 789, 193, 580, 586, 772, 642, 243, 811, 276,
#          642, 627, 389, 706, 193, 789, 35, 119, 1023, 583, 636, 339, 623, 66, 771, 464, 627, 720, 737, 459, 35, 672,
#          505, 965, 660, 1012, 666, 313, 719, 771, 662, 627, 790, 712, 122, 790, 583, 163, 263, 297, 580, 433, 610, 1002,
#          892, 334, 253, 479, 61, 591, 112, 617, 112, 737, 245, 580, 642, 924, 301, 599, 192, 903, 389, 297, 193, 591,
#          175, 680, 899, 94, 680, 680, 800, 789, 780, 600, 539, 27, 706, 219, 1023, 656, 225, 857, 333, 707, 260, 840,
#          758, 644, 477, 653, 166, 269, 574, 869, 402, 112, 849, 897, 587, 772, 870, 996, 473, 985, 539, 345, 611, 805,
#          611, 726, 66, 129, 119, 406, 653, 787, 899, 783, 256, 593, 518, 339, 339, 518, 339, 72, 168, 313, 965, 910,
#          894, 805, 845, 36, 982, 918, 526, 202, 526, 918, 638, 803, 666, 325, 828, 66, 455, 416, 946, 469, 148, 202,
#          148, 202, 148, 868, 737, 459, 237, 129, 325, 600, 595, 154, 593, 988, 688, 49, 49, 988, 250, 517, 143, 354,
#          669, 617, 514, 915, 382, 433, 749, 250, 250, 250, 250, 325, 698, 941, 551, 607, 363, 595, 961, 518, 433, 556,
#          556, 556, 688, 556, 325]
# start = [1, 16, 354, 1, 22, 354, 2, 29, 354, 2, 35, 354, 3, 45, 354, 4, 51, 354, 5, 58, 354, 6, 64]
start = [1, 16, 354, 1, 22, 354, 2, 29, 354, 2, 35, 354, 3, 45, 354, 4, 51, 354, 5, 58, 354, 6, 64, 354, 7, 74, 354, 8, 80,
    354, 9, 86, 354, 10, 93, 354, 11, 99, 354, 13, 109, 354, 14, 115, 354, 15, 122, 354, 16, 128, 354, 17, 138, 354, 18,
    144, 354, 19, 150, 354, 20, 157, 354, 21, 163, 354, 22, 173, 354, 23, 179, 354, 24, 186, 354, 25, 192, 354, 25, 198,
    354, 26, 208, 354, 27, 214, 354, 27, 221, 354, 27, 227, 354, 28, 234, 354, 29, 243, 354, 29, 253, 354, 30, 266, 354,
    30, 278, 354, 31, 288, 354, 31, 301, 354, 32, 314, 354, 32, 317, 354, 33, 317, 352, 33, 320, 352, 34, 320, 352, 34,
    320, 352, 35, 320, 352, 35, 320, 352, 36, 320, 352, 36, 320, 352, 36, 320, 352, 37, 320, 352, 37, 320, 352, 38, 320,
    352, 38, 320, 352, 38, 320, 352, 39, 320, 352, 39, 320, 352, 40, 320, 352, 40, 320, 352, 40, 320, 352, 41, 320, 352,]
# start = [873, 208, 96, 701, 96, 96, 96, 96, 96, 96, 96, 208, 208, 96, 96, 996, 295, 913, 656, 82, 849, 552, 92, 92, 748,
#          49, 746, 432, 362, 471, 969, 705, 671, 828, 990, 856, 325, 250, 172, 172, 343, 974, 83, 830, 680, 257, 922, 4,
#          255, 985, 438, 922, 897, 498, 500, 586, 534, 83, 503, 31, 576, 459, 597, 533, 29, 343, 312, 96, 893, 974, 432,
#          1002, 996, 1002, 339, 515, 761, 1002, 55, 964, 362, 417, 635, 806, 671, 393, 749, 269, 517, 583, 922, 517, 671,
#          27, 657, 376, 830, 61, 120, 675, 148, 112, 599, 220, 406, 1021, 780, 503, 503, 925, 964, 638, 483, 55, 334,
#          745, 894, 202, 1002, 851, 435, 626, 414, 276, 255, 618, 685, 780, 417, 58, 783, 899, 611, 202, 315, 764, 498,
#          208, 894, 58, 402, 92, 323, 208, 515, 803, 558, 588, 220, 1022, 884, 595, 881, 663, 813, 894, 884, 635, 881,
#          821, 610, 551, 599, 129, 553, 862, 518, 450, 749, 96, 639, 523, 965, 868, 548, 868, 517, 813, 172, 892, 107,
#          188, 168, 623, 577, 812, 769, 330, 518, 63, 751, 503, 662, 47, 808, 922, 518, 36, 925, 937, 925, 966, 925, 966,
#          244, 749, 638, 894, 816, 681, 299, 299, 96, 873, 96, 96, 96, 96, 96, 96, 96, 596, 299, 299, 299]
# start = [892, 112, 117, 126, 680]  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 360000  # number of tokens generated in each sample
# max_new_tokens = 2250 * 10  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16'  # 'float32' or 'bfloat16' or 'float16'
dtype = 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    # ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    ckpt_path = os.path.join(out_dir, 'current_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint[
    'config']:  # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ', '.join([str(itos[i]) for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()

start_ids = encode(start)
# start_ids = [89, 51]
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            decodeMessage = decode(y[0].tolist())
            import pickle
            file = open(r"samples\list_{}.bin".format(k), "wb")
            pickle.dump(decodeMessage, file)  # 保存list到文件
            file.close()
            print(decodeMessage)
            print('---------------')

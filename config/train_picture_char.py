# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-picture-char'
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'picture-char'
wandb_run_name = 'mini-gpt'

dataset = 'Picture'
batch_size = 4
# block_size = 512  # context of up to 256 previous characters
# block_size = 256  # context of up to 256 previous characters
block_size = 1024  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 5
n_head = 128
n_embd = 256
# n_embd = 1024
dropout = 0.25

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 60000
lr_decay_iters = 60000  # make equal to max_iters usually
min_lr = 1e-7  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model



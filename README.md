# HyperHuman
an attempt to implement [HyperHuman](https://arxiv.org/abs/2310.08579) \
this repository is based on [diffusers 0.21.4](https://github.com/huggingface/diffusers/tree/v0.21.4)
## installation
```
$ git clone https://github.com/MasahideOkada/HyperHuman
$ cd HyperHuman
$ pip install -r requirements.txt
```
I recommend to use venv. 

## train Latent Structural Diffusion: first stage of HyperHuman
download stable diffusion 2.0 base
```
$ source ./download-sd2-base.sh
```
and train a LSD model using the pretrained stable diffusion
```
$ accelerate launch --mixed_precision="fp16" train_lsdm.py \
--pretrained_model_name_or_path="checkpoints/stable-diffusion-2-base" \
--from_sd \
--train_data_dir="data" \
--target_dirs "rgb" "depth" "normal" \
--caption_dir="rgb" \
--condition_dir="pose" \
--rgb_index=0 \
--output_dir="hyper-human-lsd" \
--resolution=512 \
--num_train_epochs=100 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--learning_rate=1e-05 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--enable_xformers_memory_efficient_attention
```
In the example above, the training data is supposed to be structured like
```
data/
├── rgb/
│   ├── 0001.png
│   ├── 0001.txt
│   ├── 0002.png
│   ├── 0002.txt
│   └── ...
├── depth/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── normal/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── pose/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

to continue training the model from a checkpoint,
```
$ accelerate launch --mixed_precision="fp16" train_lsdm.py \
--pretrained_model_name_or_path="hyper-human-model" \
--train_data_dir="data" \
--target_dirs "rgb" "depth" "normal" \
--caption_dir="rgb" \
--condition_dir="pose" \
--output_dir="hyper-human-lsd" \
--resolution=512 \
--num_train_epochs=100 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--learning_rate=1e-05 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--enable_xformers_memory_efficient_attention
```

## train Structure-Guided Refiner: second stage of HyperHuman
download stable diffusion XL base 1.0
```
$ source ./download-sdxl1-base.sh
```
and train a SGR model using the pretrained stable diffusion
```
$ accelerate launch --mixed_precision="fp16" train_sgr.py \
--pretrained_model_name_or_path="checkpoints/stable-diffusion-xl-base-1.0" \
--from_sd \
--train_data_dir="data" \
--output_dir="hyper-human-sgr" \
--condition_dirs "pose" "normal" "depth" \
--caption_dir="rgb" \
--resolution=512 \
--num_train_epochs=100 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--learning_rate=1e-05 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--enable_xformers_memory_efficient_attention
```
to continue training the model from a checkpoint,
```
$ accelerate launch --mixed_precision="fp16" train_sgr.py \
--pretrained_model_name_or_path="hyper-human-sgr" \
--train_data_dir="data" \
--output_dir="hyper-human-sgr" \
--condition_dirs "pose" "normal-lsd" "depth-lsd" \
--caption_dir="rgb" \
--resolution=1024 \
--num_train_epochs=100 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--learning_rate=1e-05 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--enable_xformers_memory_efficient_attention
```

### some idea
can facial landmarks be effective as a condtion or target for face image generation or face super resolution?

#!/bin/sh
apt -y install -qq aria2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/text_encoder/model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-2-base/text_encoder/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-2-base/unet/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-2-base/vae/bin
mv ./checkpoints/stable-diffusion-2-base/text_encoder/bin/681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668 \
./checkpoints/stable-diffusion-2-base/text_encoder/model.safetensors
mv ./checkpoints/stable-diffusion-2-base/unet/bin/f6a94cd99d3860d654a0e817b0a249d1b1048bb8e366e95f84e65c5b2a95fecb \
./checkpoints/stable-diffusion-2-base/unet/diffusion_pytorch_model.safetensors
mv ./checkpoints/stable-diffusion-2-base/vae/bin/3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a \
./checkpoints/stable-diffusion-2-base/vae/diffusion_pytorch_model.safetensors

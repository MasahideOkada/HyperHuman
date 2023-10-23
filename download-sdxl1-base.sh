#!/bin/sh
apt -y install -qq aria2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-xl-base-1.0/text_encoder/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-xl-base-1.0/text_encoder_2/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-xl-base-1.0/unet/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-xl-base-1.0/vae/bin
mv ./checkpoints/stable-diffusion-xl-base-1.0/text_encoder/bin/660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd \
./checkpoints/stable-diffusion-xl-base-1.0/text_encoder/model.safetensors
mv ./checkpoints/stable-diffusion-xl-base-1.0/text_encoder_2/bin/ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4 \
./checkpoints/stable-diffusion-xl-base-1.0/text_encoder_2/model.safetensors
mv ./checkpoints/stable-diffusion-xl-base-1.0/unet/bin/83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139 \
./checkpoints/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors
mv ./checkpoints/stable-diffusion-xl-base-1.0/vae/bin/bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68 \
./checkpoints/stable-diffusion-xl-base-1.0/vae/diffusion_pytorch_model.safetensors

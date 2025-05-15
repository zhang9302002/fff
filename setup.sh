# create env
conda create -n vstream2 python=3.10
conda activate vstream2


# for inference, may need a ti-zi
pip install torch==2.6 torchvision==0.21.0
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install opencv-python accelerate decord Pillow
pip install peft

# pip install -U flash-attn --no-build-isolation
# check your environment, cu12 means cuda12, torch2.6 means pytorch2.6, and cxx11abiTRUE means c++11, abiFLASE is important, cp310 means python3.10
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


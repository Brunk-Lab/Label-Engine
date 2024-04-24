MODEL_CONFIG=./opencount/models/auto/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=2 \
	     --master-port 29504 \
	     ./opencount/train.py ${MODEL_CONFIG} \
	     --batch-size=12 \
	     --gpus=0,3

MODEL_CONFIG=./opencount/models/default/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=4 \
	     --master-port 29504 \
	     ./opencount/train.py ${MODEL_CONFIG} \
	     --batch-size=12 \
	     --gpus=0,1,2,3

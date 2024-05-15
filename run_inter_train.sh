MODEL_CONFIG=./engine/models/interactive/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=4 \
	     --master-port 29504 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=2 \
	     --batch-size=4 \
	     --gpus=0,1,2,3

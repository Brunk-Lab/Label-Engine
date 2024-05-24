MODEL_CONFIG=./engine/models/inter/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=4 \
	     --master-port 29504 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=3 \
	     --batch-size=3 \
	     --gpus=0,1,2,3
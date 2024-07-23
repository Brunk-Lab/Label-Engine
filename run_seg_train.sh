MODEL_CONFIG=./engine/models/inter/unet_2048x2448_roi.py
torchrun --nproc-per-node=4 \
	     --master-port 29506 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=1 \
	     --batch-size=3 \
	     --gpus=0,1,2,3
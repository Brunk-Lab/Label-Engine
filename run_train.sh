MODEL_CONFIG=./opencount/models/auto/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=2 \
	     --master-port 29504 \
	     ./opencount/train.py ${MODEL_CONFIG} \
		 --workers=2 \
	     --batch-size=4 \
	     --gpus=0,1 \
		 --weights /playpen-raid2/qinliu/models/model_0422_2024/coarse/checkpoints/chk_500/params.pth

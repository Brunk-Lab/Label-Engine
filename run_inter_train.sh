MODEL_CONFIG=./engine/models/inter/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=2 \
	     --master-port 29504 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=2 \
	     --batch-size=2 \
	     --gpus=0,3 \
		 --weights /playpen-raid2/qinliu/projects/LabelEngine/saves/model_0424_2024/inter/inter_unet_2048x2448_ecDNA/027/checkpoints/last_checkpoint.pth
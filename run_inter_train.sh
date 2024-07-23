MODEL_CONFIG=./engine/models/inter/unet_2048x2448_ecDNA.py
torchrun --nproc-per-node=4 \
	     --master-port 29506 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=1 \
	     --batch-size=3 \
	     --gpus=0,1,2,3 \
		 --weights /playpen-raid2/qinliu/projects/LabelEngine/saves/model_0424_2024/inter/inter_unet_2048x2448_ecDNA/152/checkpoints/60.pth
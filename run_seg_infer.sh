MODEL_CONFIG=./engine/models/inter/unet_2048x2448_roi.py
torchrun --nproc-per-node=1 \
	     --master-port 29505 \
	     ./engine/train.py ${MODEL_CONFIG} \
		 --workers=1 \
	     --batch-size=1 \
	     --gpus=0 \
         --weights /playpen-raid2/qinliu/projects/LabelEngine/saves/model_0424_2024/inter/inter_unet_2048x2448_roi/006/checkpoints/100.pth
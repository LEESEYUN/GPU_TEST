VRAM=8
NUM_GPUS=4


# This is for tensorflow
echo *********************************************************************
echo START TENSORFLOW GPU TEST 
python3 GPU_stress_test_in_tensorflow/main.py --vram $VRAM --num_gpus $NUM_GPUS

#This is for pytorch
echo *********************************************************************
echo *********************************************************************
echo START PYTORCH GPU TEST
python3 GPU_stress_test_in_pytorch/main.py --vram $VRAM --num_gpus $NUM_GPUS
echo *********************************************************************


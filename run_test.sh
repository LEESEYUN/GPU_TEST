VRAM=8


# This is for tensorflow
echo *********************************************************************
echo START TENSORFLOW GPU TEST 
python3 GPU_stress_test_in_tensorflow/main.py --vram $VRAM

#This is for pytorch
echo *********************************************************************
echo *********************************************************************
echo START PYTORCH GPU TEST
python3 GPU_stress_test_in_pytorch/main.py --vram $VRAM
echo *********************************************************************

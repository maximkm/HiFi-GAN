#download HiFi-GAN checkpoint
gdown "19YcqbeoFjys2qiFSJJm3bl08j2YfL6Ir&confirm=t"
gdown "1fx5upTPjfJmpVFptD_DYTCfbpF-W3o3v&confirm=t"

mkdir checkpoints -p
mv base_model.pth checkpoints/
mv finetune_model.pth checkpoints/

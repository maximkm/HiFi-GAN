#download HiFi-GAN checkpoint
gdown "19YcqbeoFjys2qiFSJJm3bl08j2YfL6Ir&confirm=t"

mkdir checkpoints -p
mv base_model.pth checkpoints/

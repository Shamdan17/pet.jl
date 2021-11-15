# ALBERT XXLARGEV2 config
MODEL=https://huggingface.co/albert-xxlarge-v2/resolve/main/pytorch_model.bin
MODELCONFIG=https://huggingface.co/albert-xxlarge-v2/resolve/main/config.json
# Name used in training scripts
MODELNAME=xxlargev2
cd src
mkdir $MODELNAME
cd $MODELNAME
echo "Downloading pretrained model"
wget $MODEL
echo "MD5 hash of checkpoint used in replication: 835f23e9f367c1f7483056745d34710d"
echo "Checking md5 hash of downloaded checkpoint"
md5sum pytorch_model.bin
echo "Downloading pretrained model config"
wget $MODELCONFIG
echo "MD5 hash of config used in replication: 662316a238a3aea57e91d8cb15bcfee9"
echo "Checking md5 hash of downloaded config"
md5sum config.json

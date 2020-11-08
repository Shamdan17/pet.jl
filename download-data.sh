cd data
echo "Downloading SuperGLUE data"
mkdir SuperGLUE
cd SuperGLUE
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip
unzip combined.zip
rm combined.zip
cd ..
echo "Downloading FewGLUE data"
wget https://github.com/Shamdan17/fewglue/releases/download/v1/FewGLUE.zip
unzip FewGLUE.zip
rm FewGLUE.zip
echo "Done."

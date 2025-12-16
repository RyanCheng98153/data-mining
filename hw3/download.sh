#!/bin/bash

# Create dataset directory if it doesn't exist
mkdir -p ./dataset/

curl -L -o ./glove6b100dtxt.zip\
  https://www.kaggle.com/api/v1/datasets/download/danielwillgeorge/glove6b100dtxt

unzip ./glove6b100dtxt.zip -d ./dataset/
rm ./glove6b100dtxt.zip

# rename the extracted file
mv ./dataset/glove.6B.100d.txt ./dataset/glove_6B_100d.txt

# Print completion message
echo "Download Finished, and is saved to ./dataset/glove_6B_100d.txt"
echo "Remember to put competition data in ./dataset/train/ and ./dataset/test/" 
gdown 1vRYAie0JcNStcSwagmCq55eirGyMYGm5
gdown 1FB24BGVrPOeUpmYbKZJYL5ermqUvBo_6
wget http://vis-www.cs.umass.edu/motionSegmentation/data/CamouflagedAnimalDataset.zip

unzip MoCA-Mask.zip
rm MoCA-Mask.zip
unzip COD10K-v3.zip
rm COD10K-v3.zip
unzip unzip CamouflagedAnimalDataset.zip
rm CamouflagedAnimalDataset.zip
rm -rf CamouflagedAnimalDataset/.DS_Store

mkdir raw
mv COD10K-v3 raw
mv MoCA_Video raw
mv CamouflagedAnimalDataset raw
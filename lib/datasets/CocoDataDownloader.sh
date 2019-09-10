# get COCO dataset
mkdir coco/
cd coco/

mkdir images
mkdir annotations

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip -d annotations/

unzip train2017.zip -d images
unzip val2017.zip -d images
unzip test2017.zip -d images

rm -f annotations_trainval2017.zip
rm -f train2017.zip
rm -f val2017.zip
rm -f test2017.zip

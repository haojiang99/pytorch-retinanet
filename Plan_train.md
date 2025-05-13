* Prepare training data by looking at folder /Users/haojiang/Documents/DDSM
* In this folder, use mass_case_description_test_set.csv file to train the retinanet model of Res50Net based on mass test dataset
* The data is in dicom format with info of 'image file path' column in the csv file
* Image needs to be converted into jpg files
* The bounding box info can be generated from dicom file with info at 'ROI Mask file path'
* The bounding box will be x1,y1,x2,y2 format
* Generate a training data txt file with format as (example like: /data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,)

and for out data, have boudning box, either benign or malignant as calcification


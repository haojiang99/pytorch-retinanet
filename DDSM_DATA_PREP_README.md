# DDSM Data Preparation for RetinaNet

This document explains how to prepare DDSM (Digital Database for Screening Mammography) data for training a RetinaNet object detection model.

## Prerequisites

- Python 3.6+
- Required Python packages:
  ```
  pip install pandas numpy pydicom opencv-python tqdm
  ```

## Dataset Structure

The script is designed to work with the CBIS-DDSM dataset, which has the following structure:
- CSV file with annotations: `mass_case_description_test_set.csv`
- DICOM images located in the CBIS-DDSM directory

## Running the Script

```bash
python prepare_ddsm_data.py --csv_file /path/to/mass_case_description_test_set.csv --ddsm_dir /path/to/CBIS-DDSM --output_dir ddsm_retinanet_data
```

### Command-line Arguments

- `--csv_file`: Path to the mass_case_description_test_set.csv file
- `--ddsm_dir`: Path to the CBIS-DDSM directory
- `--output_dir`: Output directory for prepared data (default: 'ddsm_retinanet_data')
- `--limit`: Limit processing to N samples (for testing)
- `--debug`: Enable debug outputs

## Output

The script produces:
1. A directory of JPEG images extracted from DICOM files
2. annotations.csv file for RetinaNet in the format: `path,x1,y1,x2,y2,class`
3. class_map.csv file with class mappings for RetinaNet

## Training RetinaNet

Once the data is prepared, you can train the RetinaNet model using:

```bash
python train.py --dataset csv --csv_train ddsm_retinanet_data/annotations.csv --csv_classes ddsm_retinanet_data/class_map.csv --depth 50
```

## Classes

The script converts the pathology information to the following classes:
- BENIGN → 'benign'
- MALIGNANT → 'malignant'

## Troubleshooting

If the script cannot find DICOM files:
1. Check that the paths to the CSV file and CBIS-DDSM directory are correct
2. Ensure the CSV file has the expected format
3. Verify that the DICOM files exist in the expected locations
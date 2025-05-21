import argparse
import os
import csv
import glob

def fix_paths(input_csv, output_csv, images_dir):
    """Fix image paths in the annotations.csv file"""
    
    # Get a list of all image files in the images directory
    image_files = []
    if os.path.exists(images_dir):
        image_files = [os.path.basename(f) for f in glob.glob(os.path.join(images_dir, "*.jpg"))]
    
    # Read the input CSV
    fixed_rows = []
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # path,x1,y1,x2,y2,class_name
                img_path = row[0]
                img_basename = os.path.basename(img_path)
                
                # Check if the image file exists in the images directory
                if img_basename in image_files:
                    # Update the path to point to the correct location
                    row[0] = os.path.join("images", img_basename).replace('\\', '/')
                    fixed_rows.append(row)
                else:
                    print(f"Warning: Image file not found: {img_basename}")
            else:
                print(f"Warning: Invalid row: {row}")
    
    # Write the output CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    
    print(f"Fixed {len(fixed_rows)} rows in {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Fix image paths in annotations.csv')
    parser.add_argument('--input', default='ddsm_train3/annotations.csv', help='Input annotations.csv file')
    parser.add_argument('--output', default='ddsm_train3/annotations_fixed.csv', help='Output annotations.csv file')
    parser.add_argument('--images_dir', default='ddsm_train3/images', help='Images directory')
    
    args = parser.parse_args()
    
    # Fix paths
    fix_paths(args.input, args.output, args.images_dir)
    
    print("Done!")

if __name__ == '__main__':
    main()

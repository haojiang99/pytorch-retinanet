import os
import pandas as pd
import shutil

def clean_up_dataset(base_dir):
    """
    Clean up dataset by removing entries with bounding box coordinates (columns 2 and 3) equal to 0,
    and remove the corresponding images.
    
    Args:
        base_dir: Base directory containing annotations.csv and images folder
    """
    # Define paths
    annotations_path = os.path.join(base_dir, 'annotations.csv')
    images_dir = os.path.join(base_dir, 'images')
    
    # Check if paths exist
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    # Load annotations
    try:
        df = pd.read_csv(annotations_path)
        print(f"Loaded {len(df)} entries from annotations.csv")
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return
    
    # Create backup of annotations file
    backup_path = annotations_path + '.backup'
    shutil.copy2(annotations_path, backup_path)
    print(f"Created backup of annotations file at {backup_path}")
    
    # Identify rows with bounding box first two values as 0
    # Note: Assuming columns 2 and 3 are the bounding box x and y coordinates
    # Adjust the column names/indices if they're different in your dataset
    
    # Get column names from csv (assuming columns 1 is image name, 2-5 are box coordinates)
    columns = df.columns.tolist()
    
    # If the dataset has named columns, find the bbox coordinate columns
    if len(columns) >= 5:
        # Assuming the format is something like filename, x, y, width, height
        image_col = columns[0]
        x_col = columns[1]
        y_col = columns[2]
        
        # Find rows where both x and y are 0
        invalid_rows = df[(df[x_col] == 0) & (df[y_col] == 0)]
        
        print(f"Found {len(invalid_rows)} entries with bounding box coordinates (0,0)")
        
        # Get list of images to remove
        images_to_remove = invalid_rows[image_col].tolist()
        
        # Remove rows from dataframe
        df_cleaned = df[(df[x_col] != 0) | (df[y_col] != 0)]
        
        # Save cleaned annotations
        df_cleaned.to_csv(annotations_path, index=False)
        print(f"Saved cleaned annotations with {len(df_cleaned)} entries")
        
        # Remove corresponding images
        removed_count = 0
        not_found_count = 0
        
        for img_name in images_to_remove:
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing image {img_name}: {e}")
            else:
                not_found_count += 1
                print(f"Warning: Image {img_name} not found")
        
        print(f"Removed {removed_count} images")
        if not_found_count > 0:
            print(f"Warning: {not_found_count} images were not found")
        
        print("Cleanup completed successfully!")
    else:
        print("Error: Could not determine column structure in annotations.csv")

if __name__ == "__main__":
    # Path to the ddsm_train3 folder
    base_dir = os.path.join('c:', os.sep, 'Users', 'hao_j', 'Documents', 'GitHub', 
                           'pytorch-retinanet', 'ddsm_train3')
    
    # Run the cleanup
    clean_up_dataset(base_dir)
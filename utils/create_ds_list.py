import os
from tqdm import tqdm 

def write_directory_info(path, output_file):
    with open(output_file, 'w') as file:
        unique_id = 0
        num_subs = 6968123 # ds size
        for root, dirs, files in tqdm(os.walk(path), total=num_subs, desc="Processing"):
            if root != path:  # Skip the root directory itself
                for f in files: 
                    file.write(f"{root}/{f} {unique_id}\n")
                unique_id += 1

#  -------------------------------------------------------------------------------

def write_directory_info_opt(path, output_file):
    with open(output_file, 'w') as file:
        unique_id = 0
        num_subs = 6968123 # ds size
        
        for entry in tqdm(os.scandir(path), total=num_subs, desc="Processing"):
            if entry.is_dir():
                files = os.listdir(entry)
                for f in files:
                    f_path = os.path.join(entry.path, f)
                    file.write(f"{f_path} {unique_id}\n")
                unique_id += 1
            
# Example usage:
input_path = "/home/..."
output_file = "/home/..."

write_directory_info_opt(input_path, output_file)
print(f"Directory information written to {output_file}")

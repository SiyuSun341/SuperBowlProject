import os


directory = r'S:.\superbowl_ads'


print("Files in the directory:")
for filename in os.listdir(directory):
    print(filename)

print(f"\nTotal number of files: {len(os.listdir(directory))}")
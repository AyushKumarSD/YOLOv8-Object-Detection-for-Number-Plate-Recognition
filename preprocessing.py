from sklearn.model_selection import train_test_split
import cv2
import os
import yaml


#################################################################
### Split the Data
#################################################################

root_dir = "datasets/car-number-plate/"
valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

def file_paths(root, valid_formats):
    "get the full path to each image/label in the dataset"
    file_paths = []

    # loop over the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # loop over the filenames in the current directory
        for filename in filenames:
            # extract the file extension from the filename
            extension = os.path.splitext(filename)[1].lower()

            # if the filename has a valid extension we build the full 
            # path to the file and append it to our list
            if extension in valid_formats:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

    return file_paths


image_paths = file_paths(root_dir + "images", valid_formats[:3])
label_paths = file_paths(root_dir + "labels", valid_formats[-1])

# split the data into training, validation and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)


def write_to_file(images_path, labels_path, X):

    # Create the directories if they don't exist
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # loop over the image paths
    for img_path in X:
        # Get the image name and extension
        img_name = img_path.split("/")[-1].split(".")[0]
        img_ext = img_path.split("/")[-1].split(".")[-1]
        # read the image
        image = cv2.imread(img_path)
        # save the image to the images directory
        cv2.imwrite(f"{images_path}/{img_name}.{img_ext}", image)

        # open the label file and write its contents to the new label file
        f = open(f"{labels_path}/{img_name}.txt", "w")
        label_file = open(f"{root_dir}/labels/{img_name}.txt", "r")
        f.write(label_file.read())
        f.close()
        label_file.close()

write_to_file("datasets/images/train", "datasets/labels/train", X_train)
write_to_file("datasets/images/valid", "datasets/labels/valid", X_val)
write_to_file("datasets/images/test", "datasets/labels/test", X_test)


#################################################################
### Create a YAML file
#################################################################

# Create a dictionary with the paths to the train, valid, and test sets
data = {
    "path": "../datasets", # dataset root dir (you can also use the full path to the `datasets` folder)
    "train": "images/train", # train images (relative to 'path')
    "val": "images/valid", # val images (relative to 'path')
    "test": "images/test", # test images (optional)

    # Classes
    "names":["number plate"]
}

# write the dictionary to a YAML file
with open("number-plate.yaml", "w") as f:
    yaml.dump(data, f)

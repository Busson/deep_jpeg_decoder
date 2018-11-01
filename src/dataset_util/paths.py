import os

def load_all_images_in_dir(dirName):
    listOfFiles = []
    valid_ext = ["bmp", "jpg", "png"]
    for (dirpath, dirnames, filenames) in os.walk(dirName):
            for file in filenames:
                if "_flip" in file:
                    continue
                if file.split(".")[-1] in valid_ext:
                    listOfFiles += [os.path.join(dirpath, file)]

    return listOfFiles



def split_by_category(fileList, categoryList, path_str_index):
    DIC = {}
    for category in categoryList:
        DIC[category] = []

    for file in fileList:
        file_cat = file.split("/")[path_str_index]
        if file_cat in DIC:
            DIC[file_cat].append(file)

    return DIC

    

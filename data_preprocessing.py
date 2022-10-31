import os








def filter_by_num_images(data_dir : str, min_examples: int) -> 'list[str]':
    '''
    Used to filter dataset so that only classes with at least min_examples
    datapoints are used.
    takes as input the path to the root data directory and returns
    a list of all the subfolders that have at least min_examples images 
    in them.
    Expects a dataset with the following structure:
    Root_Dir
        -- Class 1
            --ex 1
            --ex 2
            .
            .
            .
            --ex n_1
        -- Class 2
            --ex 1
            .
            .
            .
            --ex n_2
        .
        .
        .
        --- Class k
            -- ex 1
            .
            .
            .
            --ex n_k
    '''
    data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
    all_class_directories = next(os.walk(data_dir))[1]
    suitable_directories = []
    for dir in all_class_directories:
        num_files = len(next(os.walk(data_dir + dir))[2])
        if num_files >= min_examples:
            suitable_directories.append([dir, num_files])
    return suitable_directories







def main():
    test = filter_by_num_images("./Data/", 100)
    for dir in test:
        print(dir)
    # os.mkdir("./dummy_testing_directory")
    



if __name__ == "__main__":
    main()
import h5py

class File_Database:
    def __init__(self, output_path: str = ".", buffSize: int = 1000, dimension: tuple or list = [])->None:
        """
        output_path [str]: output path of the cached dataset,
        buffSize [int]: size of temporary storage,
        dimension [tuple | list]: dimension of cached datasets,
        """
        
        # creates the database file
        self.db = h5py.File(name=output_path, mode="w")
        
        # create "features" and "labels" datasets
        self.data = self.db.create_dataset(name="features", shape=dimension, dtype="float")
        self.label = self.db.create_dataset(name="labels", shape=(dimension[0],), dtype="int")
        
        # create buffer
        self.buffer = {"data": [], "label": []}
        # buffer size
        self.buffSize = buffSize
        # keep track of data moved the storage or database
        self.index = 0
        
    def add(self, data, labels)-> None:
        """
        data [array]: new features to add to the database
        labels [array]: new labels to add to the database
        """
        self.buffer["data"].extend(data)
        self.buffer["label"].extend(labels)
        
        # move the new features and labels directly to the
        # database if the buffer size is reach or exceeded
        if len(self.buffer["label"]) >= self.buffSize:
            self.flush()
            
    def flush(self):
        # new index
        new_index = self.index + len(self.buffer["label"])
        
        # move buffer content to database
        self.data[self.index : new_index] = self.buffer["data"]
        self.label[self.index : new_index] = self.buffer["label"]
        
        # update index
        self.index = new_index
        # clean buffer
        self.buffer = {"data": [], "label": []}
    
    def store_class_labels(self, class_labels : list or tuple) -> None:
        """ stores the string format of self.labels """
        # create dtype "str"
        custom_type = h5py.special_dtype(vlen=str)
        
        # create dataset in the database for the string labels
        label_store = self.db.create_dataset(name="class_labels", shape=(len(class_labels),), dtype=custom_type)
        label_store[:] = class_labels
        
    def close(self)-> None:
        """ closes the database """
        # move any entries left in the buffer
        # to the database
        if len(self.buffer["data"]) > 0:
            self.flush()
        
        # close the database
        self.db.close()
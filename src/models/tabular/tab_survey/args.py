class TabSurveyArgs:

    def __init__(self,
                 objective,

                 use_gpu,
                 gpu_ids,
                 data_parallel,

                 scale,
                 target_encode,
                 one_hot_encode,

                 batch_size,
                 val_batch_size,
                 early_stopping_rounds,
                 epochs,
                 logging_period,

                 num_features,
                 num_classes,
                 cat_idx,
                 cat_dims,
                 ):
        self.objective = objective

        # GPU parameters
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.data_parallel = data_parallel

        # Preprocessing parameters
        self.scale = scale
        self.target_encode = target_encode
        self.one_hot_encode = one_hot_encode

        # Training parameters
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.epochs = epochs
        self.logging_period = logging_period

        # About the data
        self.num_features = num_features
        self.num_classes = num_classes
        self.cat_idx = cat_idx
        self.cat_dims = cat_dims

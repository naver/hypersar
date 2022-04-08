from optparse import OptionParser

class OptParser(OptionParser):

    def __init__(self):
        OptionParser.__init__(self)

        self.add_option(
            "--dataset", type="str",
            help="name of the dataset to use for the experiment"
        )
        self.add_option(
            "--num_epoch", type="int", default=100,
            help="number of epochs"
        )
        self.add_option(
            "--batch_size", type="int", default=1024,
            help="size of a minibatch"
        )
        self.add_option(
            "--eval_batch_size", type="int", default=128,
            help="size of a minibatch at evaluation time"
        )
        self.add_option(
            "--embed_dim", type="int", default=64,
            help="dimension of the embeddings"
        )
        self.add_option(
            "--lr", type="float", default=0.001,
            help="learning rate of optimizer"
        )
        self.add_option(
            "--edge_dropout", type="float", default=0.0,
            help="probability of dropout for edges in the graph"
        )
        self.add_option(
            "--weight_dropout", type="float", default=0.0,
            help="probability of dropout for weight matrices"
        )
        self.add_option(
            "--weight_decay", type="float", default=0.0,
            help="weight of the L2 regularization"
        )
        self.add_option(
            "--num_neg_sample", type="int", default=1,
            help="number of negative sample items"
        )
        self.add_option(
            "--num_layer", type="int", default=1,
            help="number of layers for deep models"
        )
        self.add_option(
            "--num_keyword", type="int", default=2000,
            help="maximum number of keywords to consider in the queries"
        )
        self.add_option(
            "--seed", type="int", default=2019,
            help="seed for reproducibility"
        )
        self.add_option(
            "--device_embed", type="str", default="cuda",
            help="device on which the embeddings should be stored (cpu or cuda)"
        )
        self.add_option(
            "--device_ops", type="str", default="cuda",
            help="device on which the operations should be performed (cpu or cuda)"
        )
        self.add_option(
            "--cuda", type="int", default="0",
            help="index of the cuda to use (if gpu is used)"
        )
        self.add_option(
            "--model", type="str",
            help="type of model to use (one of MatrixFactorization, LightGCN, FactorizationMachine, DeepFM, JSR, DREM,"
                 " or HyperSaR)"
        )
        self.add_option(
            "--num_workers", type="int", default=4,
            help="number of workers in the data loaders"
        )
        self.add_option(
            "--use_valid", action="store_true",
            help="indicates whether to use the validation set for model selection"
        )
        self.add_option(
            "--load", action="store_true",
            help="indicates whether to load a previously trained model"
        )
        self.add_option(
            "--w2v_dir", type="str",
            default='cc.en.300.vec',
            help="directory from which to load the pre-trained word embeddings for keywords (must be in w2v/)"
        )
        self.add_option(
            "--loss_weight", type="float", default=0.0,
            help="hyperparameter balancing the reconstruction/QL loss wrt the recommendation/CIM loss (for JSR/HyperSaR)"
        )
        self.add_option(
            "--lm_weight", type="float", default=1.0,
            help="hyperparameter balancing the item-specific language model wrt the corpus language model  (for JSR)"
        )
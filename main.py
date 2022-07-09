import tensorflow as tf
import numpy as np
from datetime import datetime
import tensorflow_addons as tfa 
import configparser
import argparse

from lib import bert_encoder, process_data, spobet

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="",
                        help="train, evaluation")

    return parser

class ModelConfig:
    def __init__(self, config) -> None:
        self.MAX_LEN = int(config['model']['MAX_LEN'])
        self.INPUT_DIM = int(config['model']['INPUT_DIM'])
        self.EMBED_DIM = int(config['model']['EMBED_DIM'])
        self.NUM_HEAD = int(config['model']['NUM_HEAD'])  
        self.FF_DIM = int(config['model']['FF_DIM'])  
        self.NUM_LAYERS = int(config['model']['NUM_LAYERS'])
        self.USE_POS_EMBED = int(config['model']['USE_POS_EMBED'])
        self.PRE_TRAINED =  int(config['model']['PRE_TRAINED'])
        self.PRETRAINED_WEIGHTS = config['model']['PRETRAINED_WEIGHTS']  
        self.WEIGHTS = config['model']['WEIGHTS'] 

class TrainConfig:
    def __init__(self, config) -> None:

        self.PRETRAIN_LR = float(config['pretrain']['PRETRAIN_LR'])
        self.PRETRAIN_OPT = config['pretrain']['PRETRAIN_OPT']
        self.PRETRAIN_EPOCH = int(config['pretrain']['PRETRAIN_EPOCH'])
        self.PRETRAIN_SUB_EPOCH = int(config['pretrain']['PRETRAIN_SUB_EPOCH'])
        
        self.TRAIN_LR = float(config['train']['TRAIN_LR'])
        self.TRAIN_WARMUP_RATIO = float(config['train']['TRAIN_WARMUP_RATIO'])
        self.TRAIN_MIN_LR = float(config['train']['TRAIN_MIN_LR'])
        self.TRAIN_EPOCH = int(config['train']['TRAIN_EPOCH'])

        self.SHOW_RES = int(config['evaluation']['SHOW_RES'])

class DataConfig:
    def __init__(self, config) -> None:
        self.SPLIT_JSON = config['wlasl']['SPLIT_JSON']
        self.NAMES_FILE = config['wlasl']['NAMES_FILE']
        self.NUM_GLOSS = int(config['wlasl']['NUM_GLOSS'])

        self.PRETRAIN_BATCH = int(config['pretrain']['PRETRAIN_BATCH'])
        self.PRETRAIN_MASK_RATIO = float(config['pretrain']['PRETRAIN_MASK_RATIO'])

        self.TRAIN_BATCH = int(config['train']['TRAIN_BATCH'])
        self.TRAIN_USE_AUGMENT_ROTATE = int(config['train']['TRAIN_USE_AUGMENT_ROTATE'])
        self.TRAIN_USE_AUGMENT_MASK = int(config['train']['TRAIN_USE_AUGMENT_MASK'])

def main(args):

    modelconfig_path = "configs/modelconfig.cfg"
    modelconfigparser = configparser.ConfigParser()
    modelconfigparser.read(modelconfig_path)
    modelconfig = ModelConfig(modelconfigparser)

    dataconfig_path = "configs/dataconfig.cfg"
    dataconfigparser = configparser.ConfigParser()
    dataconfigparser.read(dataconfig_path)
    dataconfig = DataConfig(dataconfigparser)

    trainconfig_path = "configs/trainconfig.cfg"
    trainconfigparser = configparser.ConfigParser()
    trainconfigparser.read(trainconfig_path)
    trainconfig = TrainConfig(trainconfigparser)

    print("Creating Encoder...")
    mfm_model = bert_encoder.create_masked_frame_bert_model(modelconfig)

    print("Parsing dataset...")
    training_data, training_label, validation_data, validation_label, test_data, test_label = process_data.create_data_and_label(dataconfig)
    training_data_length = len(training_data)

    if args.run == "train":
        print("Running train...")
        if modelconfig.PRE_TRAINED == 1:
            is_pretrained = False
            print("Loading pre-trained encoder...")
            if modelconfig.PRETRAINED_WEIGHTS is not None:
                try:
                    mfm_model.load_weights(modelconfig.PRETRAINED_WEIGHTS).expect_partial()
                    is_pretrained = True
                except Exception as e:
                    is_pretrained = False
                    print("There is an error loading pre-trained weights, pretraining model...")
                    print(str(e))
                    pass
            
            print("Pretraining encoder...")
            if is_pretrained == False:
                if trainconfig.PRETRAIN_OPT == "Adam":
                    pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=trainconfig.TRAIN_LR)
                else:
                    pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=trainconfig.TRAIN_LR) 
                
                mfm_model.compile(optimizer=pretrain_optimizer)

                for i in range(trainconfig.PRETRAIN_EPOCH):
                    masked_train_ds = process_data.mask_ds(training_data, modelconfig, dataconfig)
                    masked_val_ds = process_data.mask_ds(validation_data, modelconfig, dataconfig)
                    mfm_model.fit(masked_train_ds, epochs=trainconfig.PRETRAIN_SUB_EPOCH, validation_data=masked_val_ds)
                
                mfm_model.save_weights("weights/pretrain")

        print("Creating model...")
        spobet_model = spobet.create_model(mfm_model, modelconfig, dataconfig)
        optimizer_classifier = tfa.optimizers.RectifiedAdam(
            learning_rate=trainconfig.TRAIN_LR, 
            total_steps= int(trainconfig.TRAIN_EPOCH * np.ceil(training_data_length / dataconfig.TRAIN_BATCH)), 
            warmup_proportion=trainconfig.TRAIN_WARMUP_RATIO, 
            min_lr=trainconfig.TRAIN_MIN_LR,)
        spobet_model.compile(
            optimizer=optimizer_classifier, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
            )
        
        print("Creating training dataset...")
        train_ds = process_data.create_ds(training_data, training_label, modelconfig, dataconfig, args.run)
        val_ds = process_data.create_ds(validation_data, validation_label, modelconfig, dataconfig, args.run)

        print("Training model...")
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        spobet_model.fit(train_ds,epochs=trainconfig.TRAIN_EPOCH,validation_data=val_ds, callbacks=[tensorboard_callback])
        spobet_model.save_weights("weights/spobet")
        print("Training done!")

    if args.run == "evaluation":

        print("Running evaluation...")
        print("Creating model...")
        spobet_model = spobet.create_model(mfm_model, modelconfig, dataconfig)

        try:
            spobet_model.load_weights(modelconfig.WEIGHTS).expect_partial()
        except Exception as e:
            print("Unable to load model weights!")
            print(str(e))
            raise
        
        print("Creating test dataset...")
        test_ds = process_data.create_ds(test_data, test_label, modelconfig, dataconfig, args.run)

        print("Running inference...")
        results = []
        acc_metric = tf.keras.metrics.CategoricalAccuracy('acc')    
        inference = []

        for k in [1,5,10]:
            for x, y in test_ds:
                num_class =  dataconfig.NUM_GLOSS
                if len(x.shape) == 3:
                    batch = x.shape[0]
                if len(x.shape) == 2:
                    batch = 1
                    x = tf.expand_dims(x, 0)
                    y = tf.expand_dims(y, 0)

                res = spobet_model(x)
                res_arg = np.argsort(res,-1)
                inference.extend(res_arg)
                
                res_one_hot = np.zeros((batch,num_class))
                for i, sorted in enumerate(res_arg):
                    truth = y[i]
                    if truth in sorted[-k:]:
                        res_one_hot[i,truth] = 1
                y_one_hot = tf.one_hot(y, num_class)
                acc_metric(res_one_hot, y_one_hot)

            acc = acc_metric.result()
            results.append(acc)
            acc_metric.reset_state()

        if trainconfig.SHOW_RES == 1:
            print("Inference Results")
            print(inference)
     

        print("Accuracy Top 1: %2f, Accuracy Top 5: %2f, Accuracy Top 10: %2f" % (results[0], results[1], results[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
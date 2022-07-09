import tensorflow as tf
import numpy as np
import glob, os
import json
import math


def create_data_and_label(dataconfig):
    class_names = []

    training_data = []
    validation_data = []
    test_data = []

    training_label = []
    validation_label = []
    test_label = []

    split = dataconfig.SPLIT_JSON
    names_file = dataconfig.NAMES_FILE
    
    with open(split, 'r') as f: 
        split_json = json.load(f)

    with open(names_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_names.append(line)

    body_pose_exclude = {8, 9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    dir = "dataset/annotations"

    for cls_id, gloss in enumerate(split_json):
        
        class_name = gloss['gloss']
        if names_file is not None:
            cls_id = class_names.index(class_name)
        else:
            class_names.append(class_name)

        samples = gloss['instances']
        
        for sample in samples:
            video_id = sample["video_id"]
            start_frame = sample['frame_start']
            end_frame = sample['frame_end']
            train_test = sample['split']
          
            sample_folder = os.path.join(dir, video_id)
            
            instance_data =[]
            frames = glob.glob(os.path.join(sample_folder,"*.json"))
            for frame in frames:
                frame_num = int(os.path.basename(frame).split("_")[1])
                
                if start_frame <= frame_num <= end_frame:                   
                    try:
                        with open(frame, 'r') as f: 
                            j = json.load(f)

                        x = []
                        y = []
                        pose_kp = []
                        pose_kp = j['people'][0]['pose_keypoints_2d']
                        pose_kp.extend(j['people'][0]['hand_right_keypoints_2d'])
                        pose_kp.extend(j['people'][0]['hand_left_keypoints_2d'])

                        for i, kp in enumerate(pose_kp):
                            if i % 3 == 0 and i // 3 not in body_pose_exclude: 
                                x.append(kp)
                            if i % 3 == 1 and i // 3 not in body_pose_exclude: 
                                y.append(kp)
                        
                        neck_x = x[1]
                        neck_y = y[1]

                        rhand_x = x[4]
                        rhand_y = y[4]

                        lhand_x = x[7]
                        lhand_y = y[7]

                        head_abs_size = np.linalg.norm(np.array([x[1],y[1]])-np.array([x[0],y[0]]))

                        for i in range(12):
                            if x[i] != 0:
                                x[i] = (x[i] - neck_x) / (3*head_abs_size) 
                            if y[i] != 0:
                                y[i] = (y[i] - neck_y) / (3*head_abs_size)

                        for i in range(12,33):
                            if x[i] != 0:
                                x[i] = (x[i] - rhand_x) / (head_abs_size)
                            if y[i] != 0:
                                y[i] = (y[i] - rhand_y) / (head_abs_size)

                        for i in range(33,54):
                            if x[i] != 0:
                                x[i] = (x[i] - lhand_x) / (head_abs_size)
                            if y[i] != 0:
                                y[i] = (y[i] - lhand_y) / (head_abs_size)

                        instance_data.append(x+y)
                    except Exception as e:
                        pass

            if train_test == 'train':
                training_data.append(np.array(instance_data))
                training_label.append(cls_id)
            if train_test == 'val':
                validation_data.append(np.array(instance_data))
                validation_label.append(cls_id)
            if train_test == 'test':
                test_data.append(np.array(instance_data))
                test_label.append(cls_id)
    
    if names_file is None:
        with open('names.txt', 'w') as f:
            for line in class_names:
                f.write(line)
                f.write('\n')
    return training_data, training_label, validation_data, validation_label, test_data, test_label

def mask_ds(data, modelconfig, dataconfig):
    training = []
    labels = []
    weights = []

    training_data_temp = data.copy()
    training_label_temp = data.copy()
    for sample, label in zip(training_data_temp, training_label_temp):
        num_frames = sample.shape[0]
        mask_ids = []    

        weight = np.zeros(modelconfig.MAX_LEN)
        sample_ph = np.zeros((modelconfig.MAX_LEN,modelconfig.INPUT_DIM))
        label_ph = np.zeros((modelconfig.MAX_LEN,modelconfig.INPUT_DIM))

        sample_ph[0] = np.ones(modelconfig.INPUT_DIM) 
        label_ph[0] = np.ones(modelconfig.INPUT_DIM) 

        for i in range(num_frames):
            sample_ph[i+1] = sample[i]
            label_ph[i+1] = label[i]
        
        for i in range(int(num_frames*dataconfig.PRETRAIN_MASK_RATIO)):
            mask_id = np.random.randint(1,num_frames+1)
            if np.random.rand() < 0.9:
                mask_embeddings = np.zeros_like(modelconfig.INPUT_DIM)
                sample_ph[mask_id] = mask_embeddings
            else:
                embeddings = np.random.rand(modelconfig.INPUT_DIM)
                sample_ph[mask_id] = embeddings
            mask_ids.append(mask_id)

        for id in mask_ids:
            weight[id] = 1
            
        training.append(sample_ph)
        labels.append(label_ph)
        weights.append(weight)

    mfm_ds = tf.data.Dataset.from_tensor_slices( (training,labels,weights) )
    mfm_ds = mfm_ds.shuffle(1000).batch(dataconfig.PRETRAIN_BATCH)

    return mfm_ds

def create_ds(data, label, modelconfig, dataconfig, ds_type):
    x = []
    data_copy = data.copy()
 
    for sample in data_copy:
        num_frames = sample.shape[0]
        sample_ph = np.zeros((modelconfig.MAX_LEN,modelconfig.INPUT_DIM))
        sample_ph[0] = np.ones(modelconfig.INPUT_DIM) 

        for i in range(num_frames):
            sample_ph[i+1] = sample[i]
        
        x.append(sample_ph)
    
    x = np.stack(x,0)       
    class_ds = tf.data.Dataset.from_tensor_slices( (x,label) )

    if ds_type == "train":
        if dataconfig.TRAIN_USE_AUGMENT_ROTATE:
            class_ds = class_ds.map(augment_rotate)
        
        if dataconfig.TRAIN_USE_AUGMENT_MASK:
            class_ds = class_ds.map(augment_mask)

    class_ds = class_ds.shuffle(1000).batch(dataconfig.TRAIN_BATCH)

    return class_ds

def augment_mask(data, label):
    ph = np.ones((250,108))
    for i in range(250 // 3):
        mask_id = np.random.randint(1,250)
        mask_embeddings = np.zeros(108)
        ph[mask_id] = mask_embeddings
    
    ph = tf.convert_to_tensor(ph)
    
    output = tf.math.multiply(ph, data)

    return output, label

def augment_rotate(data, label):
    rand = np.random.random()
    
    if rand < 0.3:
        angle_deg = np.random.random() * 13
        angle_rad = math.radians(angle_deg)
        sin = tf.constant(math.sin(angle_rad), shape=(54,), dtype=tf.float64)
        cos = tf.constant(math.cos(angle_rad), shape=(54,), dtype=tf.float64)
        
        cls = tf.expand_dims(data[0,:], 0)
        x = data[1:,0:54]
        y = data[1:,54:108]

        neg = tf.constant(-1, shape=(54,), dtype=tf.float64)

        x1 = tf.math.multiply(cos, x)
        x2 = tf.math.multiply(sin, y)
        x2 = tf.math.multiply(x2, neg)
        x_out = tf.math.add(x1, x2)

        y1 = tf.math.multiply(sin, x)
        y2 = tf.math.multiply(cos, y)
        y_out = tf.math.add(y1, y2)

        outputs = tf.concat([x_out, y_out], -1)
        outputs = tf.concat([cls, outputs], 0)

        return outputs, label

    else:
        return data, label
import argparse
import cv2
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel
from IPython.display import Image
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
checkpoint_dir = "/content/inpainting/model_logs/release_celeba_hq_256_deepfill_v2"

path = "/content/inpainting/data/"
if __name__ == "__main__":
    st_time = time()
    FLAGS = ng.Config('/content/inpainting/inpaint.yml')

    args, unknown = parser.parse_known_args()
    input_image = args.image
    model = InpaintCAModel()
    image = cv2.imread(path + input_image)
    ipimg = image
    input_image = input_image[:input_image.rfind("_")]
    mask = cv2.imread(path + input_image + "_mask.png")
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        prc_img = result[0][:, :, ::-1]
        # combo = np.hstack((ipimg, prc_img))
        cv2.imwrite("/content/result/result.jpg", prc_img)
        cv2.imwrite("/content/result/input.jpg", ipimg)
        # cv2.imwrite("/content/result/combo.jpg", combo)
        # print("Images Saved")
        # Image("/content/result/combo.jpg")
        # print("Processing Time: ", time()-st_time)

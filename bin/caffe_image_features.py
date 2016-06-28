import argparse
import sys
import os
os.environ['GLOG_minloglevel'] = '4'
sys.path.append("caffe/python")
import caffe
import numpy as np
import skimage

def crop_image(x, target_height=227, target_width=227):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = skimage.transform.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = skimage.transform.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = skimage.transform.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return skimage.transform.resize(resized_image, (target_height, target_width))

class CNN(object):

    def __init__(self, deploy, model, mean, batch_size=10, width=227, height=227):

        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        #caffe.set_mode_cpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        return net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes, dtype=np.float32)

        for start, end in zip(list(range(0, iter_until, self.batch_size)), \
                              list(range(self.batch_size, iter_until, self.batch_size))):

            image_batch_file = image_list[start:end]
            image_batch = np.array([crop_image(x, target_width=self.width, target_height=self.height) for x in image_batch_file])

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]

            all_feats[start:end] = feats

        return all_feats

def shape(string):
    return [int(s) for s in string.split("x")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image feature extraction")
    parser.add_argument("--model-prototxt", type=str, required=True)
    parser.add_argument("--model-parameters", type=str, required=True)
    parser.add_argument("--img-mean", type=str, required=True)
    parser.add_argument("--feature-layer", type=str, required=True)
    parser.add_argument("--image-directory", type=str, required=True)
    parser.add_argument("--image-list", type=argparse.FileType('r'), required=True)
    parser.add_argument("--output-file", type=argparse.FileType('wb'), required=True)
    parser.add_argument("--img-shape", type=shape, required=True)
    parser.add_argument("--output-shape", type=shape, required=True)
    args = parser.parse_args()

    cnn = CNN(deploy=args.model_prototxt, model=args.model_parameters, mean=args.img_mean,
              batch_size=10, width=args.img_shape[0], height=args.img_shape[1])
    path_list = [os.path.join(args.image_directory, f.rstrip()) for f in args.image_list]
    features_shape = [args.output_shape[2]] + args.output_shape[:2]
    features = cnn.get_features(path_list, layers=args.feature_layer, layer_sizes=features_shape)

    np.save(args.output_file, features.transpose((0, 2, 3, 1)))

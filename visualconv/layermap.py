import numpy as np
import cv2
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt


class Layermap:
    
    def __init__(self,model,vis_model,layer_name):
        self.model = model
        self.vis_model = vis_model
        self.layer_name = layer_name


    def get_CAM(self,processed_image, predicted_label):
        """
        This function is used to generate a heatmap for a sample image prediction.

        Args:
            processed_image: any sample image that has been pre-processed using the 
                           `preprocess_input()`method of a keras model
            predicted_label: label that has been predicted by the network for this image

        Returns:
            heatmap: heatmap generated over the last convolution layer output 
        """
        # we want the activations for the predicted label
        predicted_output = self.model.output[:, predicted_label]

        # choose the last conv layer in your model
        last_conv_layer = self.model.get_layer(self.layer_name)

        # get the gradients wrt to the last conv layer
        grads = K.gradients(predicted_output, last_conv_layer.output)[0]

        # take mean gradient per feature map
        grads = K.mean(grads, axis=(0,1,2))

        # Define a function that generates the values for the output and gradients
        evaluation_function = K.function([self.model.input], [grads, last_conv_layer.output[0]])

        # get the values
        grads_values, conv_ouput_values = evaluation_function([processed_image])

        # iterate over each feature map in yout conv output and multiply
        # the gradient values with the conv output values. This gives an 
        # indication of "how important a feature is"
        for i in range(8): # we have 512 features in our last conv layer
            conv_ouput_values[:,:,i] *= grads_values[i]

        # create a heatmap
        heatmap = np.mean(conv_ouput_values, axis=-1)

        # remove negative values
        heatmap = np.maximum(heatmap, 0)

        # normalize
        heatmap /= heatmap.max()

        return heatmap

    def output(self,image_path):

        #The input image is received and preprocessing is done.
        #This function normalizes the input images to the range[1,255].
        try:
            sample_image = cv2.imread(image_path)
        except:
            print("File not found. Please place the image file in the cwd.")
        
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        res = cv2.resize(sample_image, dsize=(150,150))
        sample_image = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sample_image_processed = np.expand_dims(sample_image, axis=0)


        # generate activation maps from the intermediate layers using the visualization model
        activations = self.vis_model.predict(sample_image_processed)

        # get the label predicted by our original model
        pred_label = np.argmax(self.model.predict(sample_image_processed), axis=-1)[0]

        # choose any random activation map from the activation maps 
        sample_activation = activations[0][0,:,:,8]

        # normalize the sample activation map
        sample_activation-=sample_activation.mean()
        sample_activation/=sample_activation.std()

        # convert pixel values between 0-255
        sample_activation *=255
        sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

        # get the heatmap for class activation map(CAM)
        heatmap = self.get_CAM(sample_image_processed, pred_label)
        heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
        print((sample_image.shape[0], sample_image.shape[1]))
        heatmap = heatmap *255
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #Generating the plots
        f,ax = plt.subplots(2,2, figsize=(15,8))
        ax[1,0].imshow(heatmap)
        ax[1,0].set_title("Class Activation Map")
        ax[1,0].axis('off')
        super_imposed_image = heatmap * 0.5 + sample_image
        super_imposed_image = np.clip(super_imposed_image, 0,255).astype(np.uint8)
        ax[0,0].imshow(sample_image)
        ax[0,0].set_title(f" Predicted label: {pred_label}")
        ax[0,0].axis('off')

        ax[0,1].imshow(sample_activation)
        ax[0,1].set_title("Random feature map")
        ax[0,1].axis('off')


        ax[1,1].imshow(super_imposed_image)
        ax[1,1].set_title("Activation map superimposed")
        ax[1,1].axis('off')
        plt.show()
        print("The probability of other classes:")
        print(self.model.predict(sample_image_processed))
                # print(model.predict(sample_image_processed))
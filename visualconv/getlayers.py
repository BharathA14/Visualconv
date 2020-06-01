from tensorflow.keras import Model
from .layermap import Layermap

class Layers(Layermap):
    
    def __init__(self,model):
      
        self.model  = model
        self.vis_model = None
        self.layers_names = []

    def getLayerNames(self):

        try:
            outputs = [layer.output for layer in self.model.layers[1:18]]
            # Define a new model that generates the above output
            vis_model = Model(self.model.input, outputs)
            self.vis_model = vis_model
            # check if we have all the layers we require for visualization 
            for layer in outputs:
                self.layers_names.append(layer.name.split("/")[0])
            if(len(self.layers_names)==0):
                print('No visualization layers available.')
            else:
                print('Available Layers:')
                print ('\n'.join(self.layers_names))

        except:
            print("Please check the model. Only Keras model supported.")
   
    def initializeLayer(self,layername):
        #Check if the passed input is present in the model layers
        if(layername in self.layers_names):
            super(Layers, self).__init__(self.model,self.vis_model,layername)
        else:
            print("Please enter a valid Layer name or initialze\
                   getLayerNames() if not initialied .")

    def imagepath(self,filename):
        ''' 
            This function takes the file name as input and returns the plot.
            Refer the base class for other functions.
        '''
        self.output(filename)
        
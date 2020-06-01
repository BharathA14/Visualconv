# Visualconv
Created a package to visualize different layers of a keras model for a given image. The package is also hosted as pypi. You can use the package using pip install visualconv

# Steps to use the Visualconv package
1. Initialsise the Layers from visualconv with the model.
> visual = visualconv.Layers(model)
2. Get the layers that can be visualized in the model.
> visual.getLayerNames()
3. Select the layer which you want to visualize.
>visual.initializeLayer('layer_name')
4. Select the image to be visualzied.
>visual.output('imagename.format')
 
 Example
 
 [github-small](https://github.com/BharathA14/Visualconv/edit/master/example.png)
 
It is recommended to have the image file in the current working directory.

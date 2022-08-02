from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf #---> "2.6.0" 
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template  ## templ base.html ,index.html
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)   ## __name__ -- > magic func

# Model saved with Keras model.save()
MODEL_PATH ='model_inception (2).h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255) # [0,1]
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x) ### probabilities class--> op probists
    preds=np.argmax(preds, axis=1)      # [0.9,02,0.3,-0.4,0.8,-0.5,0.1]
    if preds==0:
        preds='''This is the disease causeed : ' Bacterial_spot ' ; Symptoms: Leaf spots are small and translucent in the beginning. Later, they enlarge to circular, brown to black greasy spots surrounded with a yellowish halo. Stem lesions are black and canker-like and the fruit lesions are corky.
Management:
Crop rotation with cruciferous vegetables, field bean, maize or soybean
Seedling root dip in asafoetida solution (@ 10g/litre of water)'''
    elif preds==1:
        preds='''This is the disease causeed : ; ' Early_blight ' ; 
        1.Use Proper Plant Care Recommendations Using proper care and gardening techniques is the best way to prevent late blight. Rotate crops, clear away debris and weeds, and space tomato plants appropriately for air circulation. Always water tomato plants at the base, not on the stem.
        2.An Organic Fungicide Look for a brand of fungicides that is compliant with organic gardening. One option is called Safer; look at your local garden center for other choices. Garden fungicides can treat plants infected with early blight.
        3.Use Liquid Copper Fungicide Another option is to spray the plant with liquid copper fungicide concentrate. It’s an organic treatment that needs to be done on a dry day. The next time, remove the lower branches and repeat the treatment in one to two weeks.'''
    elif preds==2:
        preds='''This is the disease causeed : ' Late_blight ' ; 
        Getting rid of late blight is tricky, so the first steps are to prevent the disease. Purchase certified disease-free plants and seeds and keep your garden beds cleared out to remove debris that might harbor disease. 
        1.Water Properly
        It’s wise to water adequately. Keep the foliage dry and water the base of the plant only. Overhead watering spreads diseases even more. When planting, provide extra room between the tomato plants; air circulation reduces diseases. 
        2.Pull Out Plants
        Plants infected with late blight often needed to be pulled out of the garden and destroyed. Don’t compost them; they should be trashed or burned. 3.Use Copper Fungicide
In some cases, a copper fungicide can be applied and used to treat late blight effectively.'''
    elif preds==3:
        preds='''This is the disease causeed : ' Leaf_Mold ' ; This is are the Pesticides : Vinegar – Similar to mouthwash, the acetic acid of vinegar can control powdery mildew. A mixture of 2-3 tablespoons of common apple cider vinegar, containing 5% acetic acid mixed with a gallon of water does job.
Standard fungicides are an effective way to manage powdery mildew disease on plants'''
    elif preds==4:
        preds='''This is the disease causeed : ' Septoria_leaf_spot ' ; Septoria leaf spot, also known as septoria blight is a common disease of the tomato plant, which also affects other members of the plant family Solanaceae, namely potatoes and eggplant. The disease is caused by the fungus Septoria lycopersici, and is known to affect crops in different regions all around the world.
how to manage :
Remove diseased leaves, improve air circulation around plants, use drip irrigation instead of overhead watering or water soil directly, taking care to keep leaves as dry as possible. Pull up any weeds you see and mulch around the base of the plant. Practice crop rotation, alternating where you plant tomatoes, never planting tomatoes in a location where tomatoes, potatoes, or eggplant was grown in the previous year. Spray with fungicidal sprays that contain copper or potassium bicarbonate. Use chemical controls, like Fungonil, if the infestation is severe. '''
    elif preds==5:
        preds='''This is the disease causeed : ' Spider_mites ' ; Two-spotted_spider_mite This is are the Pesticides : products containing natural plant extracts, insecticidal soap, and powerful miticides based on chemicals providing a knockdown effect. Keep in mind that spider mites cannot be killed with regular insecticides, therefore, you should opt only for products labeled as miticides
1. Garden Safe Brand Insecticidal Soap    
2. BioAdvanced 701290B Insecticide Miticide
3. Safer Brand Insect Killing Soap'''
    elif preds==6:
        preds='''This is the disease causeed : 'Target_Spot' ;  Characteristic symptoms of Target Spot include brown lesions, sometimes approaching 2 cm (~1 inch) in diameter, exhib-iting a series of concentric rings. Unlike Stemphylium and Alternaria Leaf Spot, the spots are typically not bordered by a dark band. Leaf spots and premature defoliation are generally confined to the interior canopy (unlike that found in Stemphylium and Alternaria diseases.'''
    elif preds==7:
        preds='''This is the disease causeed : ' Tomato_Yellow_Leaf_Curl_Virus ' ; symptomatic plants should be carefully covered by a clear or black plastic bag and tied at the stem at soil line. Cut off the plant below the bag and allow bag with plant and whiteflies to desiccate to death on the soil surface for 1-2 days prior to placing the plant in the trash. Do not cut the plant off or pull it out of the garden and toss it on the compost! The goal is to remove the plant reservoir of virus from the garden and to trap the existing virus-bearing whiteflies so they do not disperse onto other tomatoes'''
    elif preds==8:
        preds='''This is the disease causeed :'Tomato_mosaic_virus' A homemade solution is prepared for these purposes: 100 g of micronutrient fertilizers are added per liter of milk whey. The prepared liquid carefully processed leaves on tomatoes (it is necessary to go through each leaf) and stems.For the best treatment, use a conventional spray gun.
General disease prevention will protect adult bushes until harvest. Planted seedlings must be vaccinated against the virus (vaccination is carried out), 20% hydrochloric acid is used to disinfect seeds, seedlings and stems of bushes.
'''
    else:
        preds=" ' Healthy HURRY! ' It is a good Leaf \U0001f60D"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True,port=9002)
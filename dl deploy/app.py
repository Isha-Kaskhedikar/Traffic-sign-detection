from flask import *
# from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
# from PIL import Image
from keras.preprocessing import image
import os
# import matplotlib.pyplot as plt

app = Flask(__name__)

classes = { 0:'No passing',
            1:'No passing veh over 3.5 tons',
            2:'Right-of-way at intersection',
            3:'Priority road',
            4:'Yield',
            5:'Stop',
            6:'Vehicle > 3.5 tons prohibited',
            7:'No entry',
            8:'General caution',
            9:'Dangerous curve left',
            10:'Dangerous curve right',
            11:'Bumpy road',
            12:'Slippery road',
            13:'Road narrows on the right',
            14:'Road work',
            15:'Traffic signals',
            16:'Pedestrians',
            17:'End speed + passing limits',
            18:'Go straight or left',
            19:'End of no passing',
            20:'End no passing vehicle > 3.5 tons' }

# Classes of trafic signs
# classes = { 0:'Speed limit (20km/h)',
            # 1:'Speed limit (30km/h)',
            # 2:'Speed limit (50km/h)',
            # 3:'Speed limit (60km/h)',
            # 4:'Speed limit (70km/h)',
            # 5:'Speed limit (80km/h)',
            # 6:'End of speed limit (80km/h)',
            # 7:'Speed limit (100km/h)',
            # 8:'Speed limit (120km/h)',
            # 9:'No passing',
            # 10:'No passing veh over 3.5 tons',
            # 11:'Right-of-way at intersection',
            # 12:'Priority road',
            # 13:'Yield',
            # 14:'Stop',
            # 15:'No vehicles',
            # 16:'Vehicle > 3.5 tons prohibited',
            # 17:'No entry',
            # 18:'General caution',
            # 19:'Dangerous curve left',
            # 20:'Dangerous curve right',
            # 21:'Double curve',
            # 22:'Bumpy road',
            # 23:'Slippery road',
            # 24:'Road narrows on the right',
            # 25:'Road work',
            # 26:'Traffic signals',
            # 27:'Pedestrians',
            # 28:'Children crossing',
            # 29:'Bicycles crossing',
            # 30:'Beware of ice/snow',
            # 31:'Wild animals crossing',
            # 32:'End speed + passing limits',
            # 33:'Turn right ahead',
            # 34:'Turn left ahead',
            # 35:'Ahead only',
            # 36:'Go straight or right',
            # 37:'Go straight or left',
            # 38:'Keep right',
            # 39:'Keep left',
            # 40:'Roundabout mandatory',
            # 41:'End of no passing',
            # 42:'End no passing vehicle > 3.5 tons' }

model = load_model('C:/Users/91700/Downloads/signal_model2.h5',custom_objects=None, compile=True)

def model_predict(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    # preds = model.predict(x)
    # preds=np.argmax(preds, axis=1)
    # if preds==0:
    #     preds="The leaf is diseased cotton leaf"
    # elif preds==1:
    #     preds="The leaf is diseased cotton plant"
    # elif preds==2:
    #     preds="The leaf is fresh cotton leaf"
    # else:
    #     preds="The leaf is fresh cotton plant"

        # val=model.predict(file_path)
    images=np.vstack([x])
    val=model.predict(images)
    value=val[0]
    if 1 not in val[0]:
        string="No data present"
    else:
        string="image is of :   ", classes[np.where(value==1)[0][0]]
  
    # result = "Predicted Traffic🚦Sign is: " +classes[preds]
    return string


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

    # for i in os.listdir( r'C:/Users/91700/dl deploy/uploads'):
    #     img=image.load_img(r'C:/Users/91700/dl deploy/uploads', target_size=(224,224))
    #     plt.imshow(img)
    #     plt.show()   
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        
        # X=image.img_to_array(img)
        # X= np.expand_dims(X,axis=0)
        # images=np.vstack([X])
        # val=model.predict(file_path)
        # value=val[0]
        # if 1 not in val[0]:
        #     print("No data present")
        # else:
        #     print("image is of :   ", classes[np.where(value==1)[0][0]])

        # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        # # Make prediction
        preds = model_predict(r'C:/Users/91700/dl deploy/uploads/00005_00019.ppm.jpg')
        # result=preds
        # string="image is of :   ", classes[np.where(value==1)[0][0]]
    return None



# if __name__ == '__main__':
#     app.run(port=5001,debug=True)


# def image_processing(img):
#     model = load_model('C:/Users/91700/Downloads/signal_model2.h5',custom_objects=None, compile=True)
#     data=[]
#     image = Image.open(img)
#     image = image.resize((30,30))
#     data.append(np.array(image))
#     X_test=np.array(data)
#     Y_pred = model.predict_classes(X_test)
#     return Y_pred

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']
#         file_path = secure_filename(f.filename)
#         f.save(file_path)
#         # Make prediction
#         result = image_processing(file_path)
#         s = [str(i) for i in result]
#         a = int("".join(s))
#         result = "Predicted Traffic🚦Sign is: " +classes[a]
#         os.remove(file_path)
#         return result
#     return None

if __name__ == '__main__':
    app.run(debug=True)
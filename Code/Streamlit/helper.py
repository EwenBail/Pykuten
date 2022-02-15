import os
import cv2
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

from tensorflow.python.keras.backend import ones
backup_img='Data_sources/Images/test_image.png'
dict_img_classes={'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26}

def get_class_num_from_img_idx(int_index):
    for str_key, int_val in dict_img_classes.items():
        if int_index == int_val:
            real_class = str_key
    return int(real_class)

def get_img_path(img_id,product_id,train=True):
    #fonction pour accéder facilement aux répertoire image
    """
    Function to get img_path from img_id and product_id
    """
    source_path='Data_sources/images/'
    subfold='image_train/'
    if not train: subfold='image_test/'        
    ext='.jpg'
    img_code='image_{0}_product_{1}'.format(img_id,product_id)
    return source_path+subfold+img_code+ext

# IMAGE MODEL LOADER
def get_img_model_name(IMAGE_ALGO='ResNet50',BATCH_SIZE=64, LEARNING_RATE=0.01):
  MODEL_BASE = 'Modeles/' + str(BATCH_SIZE) + '_' + str(LEARNING_RATE) + '_'
  MODEL_JSON = MODEL_BASE + 'classifierTranferLearning'+ IMAGE_ALGO + '.json'
  MODEL_H5 = MODEL_BASE + 'classifierTranferLearning'+ IMAGE_ALGO + '.h5'
  return MODEL_JSON, MODEL_H5

def load_img_model(base, LR):
    json, h5=get_img_model_name(IMAGE_ALGO=base,BATCH_SIZE=64, LEARNING_RATE=LR)
    # print('___________JSON___________')
    # print(json)
    # print('___________h5___________')
    # print(h5)
    image_model_=None
    if os.path.exists(h5) & os.path.exists(json):
        with open(json, 'r') as fx:
            model_json_string = fx.read()

        image_model_ = model_from_json(model_json_string)
        image_model_.load_weights(h5)
    return os.path.exists(h5) & os.path.exists(json), image_model_

#TEXT MODEL LOADER
def get_txt_model_name(IMAGE_ALGO='GRU',BATCH_SIZE=64, LEARNING_RATE=0.01):
  MODEL_BASE = 'Modeles/' + str(BATCH_SIZE) + '_' + str(LEARNING_RATE) + '_'
  if IMAGE_ALGO=='GRU':
      IMAGE_ALGO=''
  MODEL_H5 = MODEL_BASE + 'textClassifierRNN'+ IMAGE_ALGO + 'Bow.h5'
  return MODEL_H5

def load_txt_model(base, LR):
    h5=get_txt_model_name(IMAGE_ALGO=base, LEARNING_RATE=LR)
    # print('___________h5___________')
    # print(h5)
    text_model_=None
    if os.path.exists(h5):
        text_model_ = keras.models.load_model(h5)
    return os.path.exists(h5), text_model_

def format_html_markdown(txt,txt_color):    
    return '<p style="font-family:sans-serif; color:'+ txt_color +'; font-size: 25px;">'+ txt +'</p>' 

#Preprocessing

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    stop_words = stopwords.words(['french', 'english'])
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]

    return ' '.join(mots).strip()

def load_data_test():
    xx_df = pd.read_csv("Data_sources/data_test_csv.csv")
    xx_df['label']=xx_df['label'].astype('string')
    xx_df['text_cleaned'] = xx_df['text'].apply(lambda x: preprocess_sentence(x))    
    return xx_df.iloc[:, 1:]


def get_single_line_df(df, index):
    return df.loc[[index]]

def redirect_img_path(img_path):
    """
    Fonction qui permet de rediriger les chemins des fichiers images pour ne pas avoir à déplacer une quantité monstrueuse de données
    """
    
    # true_folder='../Projet Rakuten/Data_sources'
    # img_path=img_path.replace('./DATA', true_folder)
    # print(img_path)
    if not os.path.exists(img_path):
        img_path=backup_img
    return img_path

def get_img_prediction_list_prob(single_line_df,model):
    # Configure ImageDataGenerator
    img_gen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    data_flow_test_one = img_gen_test.flow_from_dataframe(
        dataframe=single_line_df,
        target_size=(256, 256),
        shuffle=False,
        x_col='path',  # column containing path to image
        y_col='label',  # column containing label
        class_mode='sparse',  # classes are not one-hot encoded
        batch_size=64,
        verbose=0
    )
   
    y_img_list_prob = model.predict(
        x=data_flow_test_one,
        batch_size=None,  # specified by generator
        steps=None,  # specified by generator
        verbose=0
    )
    coorrected_y_img_list_prob=correct_img_list_prob(y_img_list_prob)

    return coorrected_y_img_list_prob

def get_img_prediction(single_line_df,model):

    y_img_list_prob=get_img_prediction_list_prob(single_line_df,model)
    # print('Img_LIST')
    # print(y_img_list_prob)
    return get_class_prct(y_img_list_prob)

def get_txt_prediction_list_prob(single_line_df, txt_model,tokenizer):
    # Nettoyage avant calcul
    sequences_one = tokenizer.texts_to_sequences(single_line_df['text_cleaned'])
    sequences_one = tf.keras.preprocessing.sequence.pad_sequences(sequences_one, padding='post')

    y_txt_list_prob = txt_model.predict(sequences_one)

    return y_txt_list_prob

def predict_text(single_line_df, txt_model,tokenizer):
    y_txt_list_prob= get_txt_prediction_list_prob(single_line_df, txt_model,tokenizer)
    return get_class_prct(y_txt_list_prob)

def get_class_prct(prob_list):
    pred_class = np.argmax(prob_list, axis=1)[0]
    prct=prob_list[0,pred_class]
    return pred_class, prct

def get_full_predict(single_line_df,img_model, txt_model,tokenizer):
    #predict_img
    y_pred_img_list_prob=get_img_prediction_list_prob(single_line_df,img_model)
    # print("______________IMG____________") 
    # print(y_pred_img_list_prob)
    #predict_txt
    y_pred_text_list_prob=get_txt_prediction_list_prob(single_line_df, txt_model,tokenizer)
    # print("______________TXT____________")     
    # print(y_pred_text_list_prob)
    y_pred_mean=(y_pred_text_list_prob+y_pred_img_list_prob)/2
    # print("______________Mean____________")
    # print(y_pred_mean)
    return get_class_prct(y_pred_mean)

def correct_img_list_prob(prob_mat):
    # assignation par valeur pour pouvoir modifier la matrice plus tard
    tmp_matrix=np.zeros(prob_mat.shape)
    for index in range(prob_mat.shape[1]):
        true_ind=get_class_num_from_img_idx(index)
        tmp_matrix[0][true_ind]=prob_mat[0][index]
    return tmp_matrix

def load_product_catalog():
    return pd.read_csv('Data_sources/classes produits.csv', index_col=0)

def load_y_train():
    return pd.read_csv('Data_sources/Y_train.csv', index_col=0)

def load_X_train():
    df=pd.read_csv('Data_sources/X_train_csv.csv', index_col=0)
    df = df.iloc[:, 1:]
    df['label'] = df['label'].astype('string')
    df['text'] = df['text'].astype('string')
    df['text'] = df['text'].apply(
        lambda x: x if isinstance(x, str) == True else "")
    return df

def get_conf_mat_path(model_img, img_LR, model_txt, txt_LR):
    path='Conf_mat/64_'+ img_LR + '_mc'+ model_img + '_RNN'+ model_txt + '_' + txt_LR + '.png'
    if not os.path.exists(path):
        path=backup_img
    return path



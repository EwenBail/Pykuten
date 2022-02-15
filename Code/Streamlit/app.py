from numpy import nan
import streamlit as st
from streamlit.state.session_state import SessionState
from helper import *
import tensorflow as tf

def are_model_loaded():
    return st.session_state.img_model_loaded & st.session_state.txt_model_loaded

def main():
    """Streamlit presentation: Projet Pykuten"""
    
    # Create sidebar and set title
    st.title("Projet Pykuten")
    st.write("""
    ## Presentation d'un projet de classification multimodal
    """
    )
    st.sidebar.text("Projet présenté par:\n\
    Sana Lamiri\n\
    Ewen Bail\n\
    Olivier Douangvichith")
    
    st.sidebar.text("Supervision:\n\
    Gaspard")

    st.sidebar.text("Datascientest:\n\
    Bootcamp Data Scientist\n\
    Octobre 2021")    
    st.sidebar.text("Le 07/01/2022")


    ### Initialise session states:
    if 'data_test' not in st.session_state:
        #Step 0: Charger data_test
        st.session_state.data_test=load_data_test()
        st.session_state.product_catalog=load_product_catalog()
        st.session_state.y_train = load_y_train()
        st.session_state.x_train=load_X_train()     
        # Définition du tokenizer
        st.session_state.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        # Mettre à jour le dictionnaire du tokenizer
        st.session_state.tokenizer.fit_on_texts(st.session_state.x_train.text)

    
    dict_code_to_id = {}
    dict_id_to_code = {}
    list_tags = list(st.session_state.y_train['prdtypecode'].unique())

    for i, tag in enumerate(list_tags):
        dict_code_to_id[tag] = i
        dict_id_to_code[i] = tag
        # print('________________Index___________________________')
        # print(str_index)
        # print('________________Real_class___________________________')
        # print(real_class) 


    def get_index_from_libelle(libelle):
        # print('________________libelle___________________________')
        # print(libelle)
        # print('________________PRDTYPECODE___________________________')
        prdtypecode=st.session_state.product_catalog[st.session_state.product_catalog['prdtypecategory']==libelle].index[0]
        # print(prdtypecode)
        # print('________________Index___________________________')
        index=dict_code_to_id[prdtypecode]
        # print(index)
        return(index)        

    def get_libelle(real_class):       
        # print('________________Real_class___________________________')
        # print(real_class)
        prdtypecode=dict_id_to_code[int(real_class)]
        # print('________________PRDTYPECODE___________________________')
        # print(prdtypecode)
        # print('________________Libele___________________________')
        libelle=st.session_state.product_catalog.loc[prdtypecode]['prdtypecategory']
        # print(libelle)

        return(libelle)

    if 'img_model_loaded' not in st.session_state:
        st.session_state.img_model_loaded=False

    if 'txt_model_loaded' not in st.session_state:
        st.session_state.txt_model_loaded=False

    if 'article_loaded' not in st.session_state:
        st.session_state.article_loaded=False

    
    img_col, txt_col=st.columns([2,1])

    ###### Chargement des modeles:

    ### Modele image
    img_col.header("Image")
    img_param=img_col.container()
    img_param.subheader("Paramètres")
    img_model_name= img_param.radio("Select img base model", ("VGG16","ResNet50","MobileNet_V2"))
    img_model_LR= img_param.radio("Select img model Learning rate", ("0.01","0.001"))
    load_img_model_btn=img_param.button('Load Model', key='img_model_btn')
    
    if load_img_model_btn:
        model_found, st.session_state.img_model=load_img_model(img_model_name,img_model_LR)

        if model_found:
            st.session_state.img_model_loaded=True
            st.session_state.img_model_name=img_model_name
            st.session_state.img_model_LR=img_model_LR
            img_param.markdown(format_html_markdown('Model chargé','green'), unsafe_allow_html=True)
        else:
            st.session_state.img_model_loaded=False
            img_param.write(format_html_markdown('Model non existant','red'), unsafe_allow_html=True)  

    if st.session_state.img_model_loaded:
        img_param.write('Base : **{0}**  \nLR    : **{1}**'.format(st.session_state.img_model_name,st.session_state.img_model_LR))
    else:
        img_param.write("Aucun model chargé")

    ### Modele Text
    txt_col.header("Texte")
    txt_param=txt_col.container()

    txt_param.subheader("Paramètres")
    txt_model_name= txt_param.radio("Select txt base model", ("GRU","LSTM"))
    txt_model_LR= txt_param.radio("Select txt model Learning rate", ("0.01","0.001"))
    load_txt_model_btn=txt_param.button('Load Model',key='txt_model_btn')

    if load_txt_model_btn:
        model_found, st.session_state.txt_model=load_txt_model(txt_model_name,txt_model_LR)
        if model_found:
            st.session_state.txt_model_loaded=True
            st.session_state.txt_model_name=txt_model_name
            st.session_state.txt_model_LR=txt_model_LR
            txt_param.markdown(format_html_markdown('Model chargé','green'), unsafe_allow_html=True)
        else:
            st.session_state.img_model_loaded=False
            txt_param.write(format_html_markdown('Model non existant','red'), unsafe_allow_html=True)  
    
    if st.session_state.txt_model_loaded:
        txt_param.write('Base : **{0}**  \nLR    : **{1}**'.format(st.session_state.txt_model_name,st.session_state.txt_model_LR))
    
    if are_model_loaded():
        mat_conf_path=get_conf_mat_path(st.session_state.img_model_name, st.session_state.img_model_LR, st.session_state.txt_model_name, st.session_state.txt_model_LR)
        mat_conf=cv2.imread(mat_conf_path)
        with st.expander("Matrice de confusion"):
            st.write("IMG: {0} - LR: {1}  \nTXT: {2} - LR: {3}".format(st.session_state.img_model_name, st.session_state.img_model_LR, st.session_state.txt_model_name, st.session_state.txt_model_LR))
            st.image(mat_conf,channels ='BGR')

    ###### Chargement de l'article
    article=st.container()
    article.write("""# ARTICLE""")
    chkbox, sel_box = article.columns(2)
    force_category_chkbox=chkbox.checkbox('Forcer la categorie')
    forced_idx=None
    if force_category_chkbox:
        forced_libelle=sel_box.selectbox('Selection de la categorie',st.session_state.product_catalog)
        forced_idx=get_index_from_libelle(forced_libelle)


    load_article_btn=article.button('Charger Article', key='Load_Article_btn')
    article_img, article_txt = article.columns(2)

    #### PREDICTION

    predict=st.container()
    predict.write("""# PREDICTION""")
    predict_img, predict_txt, predict_final = predict.columns(3)

   

    if load_article_btn:

        #Step 1: Getting random index
        if (forced_idx is None):
            index = np.random.choice(st.session_state.data_test.index)
        else:
            subset=st.session_state.data_test[st.session_state.data_test['label']==str(forced_idx)]
            index = np.random.choice(subset.index)
            # print('INDEX__FORCé')
            # print(forced_idx)
            # print('libelle associé')
            # print(get_libelle(forced_idx))

        #Permet de forcer manuellement l'index 
        
        # index=1988
        print("________________INDEX DE L'ARTICLE SELECTIONNE____________________")
        print(index)
        single_line_df=get_single_line_df(st.session_state.data_test,index)

        txt=single_line_df['text'].iloc[0]
        img_path=single_line_df['path'].iloc[0]

        img_path=redirect_img_path(img_path)
        img=cv2.imread(img_path)

        article_img.image(img,channels ='BGR')
        article_txt.write(txt)


        st.session_state.article_loaded=True

        if are_model_loaded():

            img_pred,img_score=get_img_prediction(single_line_df,st.session_state.img_model)
            # img_pred_real_class=int(get_class_num_from_img_idx(img_pred))
            img_pred_lib=get_libelle(img_pred)

            txt_pred,txt_score=predict_text(single_line_df, st.session_state.txt_model,st.session_state.tokenizer)
            txt_pred_lib=get_libelle(txt_pred)

            final_pred,final_score=get_full_predict(single_line_df,st.session_state.img_model, st.session_state.txt_model,st.session_state.tokenizer)
            final_pred_lib=get_libelle(final_pred)

            true_class=int(single_line_df['label'])           
            true_class_lib=get_libelle(true_class)

            article.write("### Classe Réelle")
            article.markdown(format_html_markdown(true_class_lib,'blue'), unsafe_allow_html=True)

            predict_img.write("### Image")
            
            if img_pred==true_class:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_img.markdown(format_html_markdown(img_pred_lib,tmp_color), unsafe_allow_html=True)
            predict_img.markdown(format_html_markdown('{:.1f}%'.format(img_score*100),tmp_color), unsafe_allow_html=True)

            predict_txt.write("### Text")


            if txt_pred==true_class:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_txt.markdown(format_html_markdown(txt_pred_lib,tmp_color), unsafe_allow_html=True)
            predict_txt.markdown(format_html_markdown('{:.1f}%'.format(txt_score*100),tmp_color), unsafe_allow_html=True)

            predict_final.write("### Full Model")

            if final_pred==true_class:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_final.markdown(format_html_markdown(final_pred_lib,tmp_color), unsafe_allow_html=True)
            predict_final.markdown(format_html_markdown('{:.1f}%'.format(final_score*100),tmp_color), unsafe_allow_html=True)

        else:
            predict.write("## Model non chargé")


if __name__ == "__main__":
    main()
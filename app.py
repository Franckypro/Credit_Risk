import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import base64
#import streamlit_authenticator as stauth
#from authenticator import sign_up, fetch_users
from pycaret.classification import load_model, predict_model
import joblib
 



st.set_page_config(page_title='Streamlit', page_icon='🐍', initial_sidebar_state='collapsed')


#try:
#    users = fetch_users()
#    emails = [] 
#    usernames = []
#    passwords = []
#    
#    for user in users:
#        emails.append(user['key'])
#        usernames.append(user['username'])
#        passwords.append(user['password'])
#    
#    credentials = {'usernames': {}} 
#    for index in range(len(emails)):
#        credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}
#    
#    Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)
#    
#    email, authentication_status, username = Authenticator.login(':green[Login]', 'main')
#    
#   info, info1 = st.columns(2)
#   
#    if not authentication_status:
#        sign_up()
    
#    if username:
#        if username in usernames:
#            if authentication_status:
                # let User see app
                
    
    # Contenu de la page
            
                    
#code de la page d'accuille et les menus 
st.sidebar.image("CFC.jpg",width=220)
st.sidebar.subheader(f'Welcome :orange[User]')
#Authenticator.logout('Log Out', 'sidebar')

st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

#image de fond 
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("imagebare2.jpg")




menu=["ACCUEIL","ANALYSE DES DONNÉES","VISUALISATION","CLASSIFICATION","PRÉDICTION"]  
choice = st.sidebar.selectbox("selectionne un Menu",menu)

#les differents menus et les fonctionalités

#menu HOME

if choice == "ACCUEIL":

    #Image de fond du menu
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://www.creditfoncier.cm/images/revslider/uploads/slides-1bis_1.jpg");
    background-size: 800%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: top lef; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    left,middle,right = st.columns((2,3,2))
    with middle:
        st.image("logo_CFC.png",width=300)
    st.markdown("<h1 style='text-align:center;color: orange;'>CREDIT FONCIER DU CAMEROUN </h1>",unsafe_allow_html=True)
    st.write("<h2 style='text-align:center;color: black;'>Vous loger,notre seul souci.</h2>",unsafe_allow_html=True)
        

    
    #bande deroulente pour la decription du CFC
    expander = st.expander("DESCRIPTION DE L'APPLICATION")
    expander.write("""
    L'application ' ' est une application conçue pour résoudre les problèmes de prise de décision automatisée 
    pour l'octroi de prêt classique ordinaire au sein du CFC. Grâce à des techniques avancées de machine Learning,
    l'application analyse les profits des demandeurs de prêt en se basant sur des differentes paramètres pertinents. 
    Elle génère ensuite des évaluations de risque précises et rapides, permettant au CFC de prendre des décisions
    éclairées et efficentes en matière d'octroi de prêts. L'application ' ' vise à améliorer la précision des décisions
    de crédit tout en optimisant le processus global d'octroi de prêts, ce qui contribue à renforcer la qualité des services
    financiers du CFC.
        
    """)# mettre une description ici
    expander.image("https://www.creditfoncier.cm/images/revslider/uploads/slides-1bis_1.jpg")#changer cette image

         

    #bande deroulante pour les services qu'offre le cfc
    expander = st.expander("SERVICES QU'OFFRE LE NOM DE L'APPLICATION")
    expander.write("""
        -I- la classification des individus à risque ou pas pour le remboursement d'un credit reçu. 
    """)# mettre une description ici
    expander.image("https://media.istockphoto.com/id/1347375207/fr/photo/mains-tenant-le-visage-triste-cach%C3%A9-derri%C3%A8re-un-visage-heureux-bipolaire-et-d%C3%A9pression-sant%C3%A9.webp?b=1&s=612x612&w=0&k=20&c=Cv6RY2Io-HTM8sN-5I587k4BruOAadtkotU8R3r83cM=")
    expander.write("""



    """)
    expander.write("""
        -I- La proposides des individus qui ont  reçus un prêt et ont rembousés et la proposides des individus qui ont reçus un prêt et n'ont pas rembousés  
    """)
    expander.image("cercle de risque.png")
    expander = st.expander("GUIDE")
    expander.write("""
        
    """)# mettre une description ici
    expander.write("Quelle est la mission du Crédit Foncier du Camemour")
    expander.caption(" Le CFC a pour mission de faciliter la réalisation des projets immobiliers initiés par tout camerounais, en leur octroyant des crédits à des taux d’intérêts relativement bas et remboursables sur de longues durées")


        
    #page

#menu DATA ANALYSIS

if choice == "ANALYSE DES DONNÉES":
    #image de font 

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://cdn.pixabay.com/photo/2017/02/26/12/38/sunset-2100140_640.jpg");
    background-size: 700%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: top lef; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;color: red;'>ANALYSE DE DONNÉES </h1>",unsafe_allow_html=True)
    st.write ("<br><br>",unsafe_allow_html=True)
    tb, tb1, tb2 = st.tabs([":clipboard: Dataset",":bar_chart: summary",":chart_with_upwards_trend: correlation"])
    data = load_data("credit_risk_dataset.csv")

    with tb: #display les 5 premières lignes de notre dataset 
        st.subheader("Credit risk Dataset")
        st.write(data.head(10))
    with tb1:

            #AFFICHER LE SUMMARY
        st.subheader("TABLEAU DE SUMMARY ")
        st.write(data.describe().head(8))

        #AFFICHER LA MATRIXE DE CORELATION
    with tb2:
        st.subheader("MATRIXE DE CORRÉLATION")
        fig = plt.figure(figsize=(15,15))
        st.write(sns.heatmap(data.corr(), annot=True))
        st.pyplot(fig)
    
        # afficher la figure 

        #fig , ax = plt.histplot()
        #ax.bar(data["loan_amnt"], data["person_age"])
        #plt.xlabel("person_age")
        #plt.ylabel("loan_amnt")
        #plt.title('Presentation.......')
        #afficher le graphique

#menu DATA VISUALISATION

if choice == "VISUALISATION":

    #image de font 

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://cdn.pixabay.com/photo/2017/10/05/10/59/background-2819026_640.jpg");
    background-size: 700%;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: top lef; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;color: red;'>VISUALISATION GRAPHIQUE </h1>",unsafe_allow_html=True)
    data = load_data("credit_risk_dataset.csv")
    st.write(data.head(5))

    #representation du Countplot
    if st.checkbox("Countplot"):
        fig = plt.figure(figsize = (13,13))
        sns.countplot(x="person_age",data=data)
        st.pyplot(fig)
    
    #representation du scatterplot
    if st.checkbox(" REPRESENTATION "):
        fig = plt.figure(figsize = (8,8))
        data = load_data('credit_risk_dataset_prop.csv')
        sns.scatterplot(x="person_age",y="loan_amnt",data=data,hue="loan_status")
        st.pyplot(fig)


    #representation de l'importance des colonnes 
    if st.checkbox("Importance des colonnes"):
        #visualiser l'importance des colonnes
        model=joblib.load('credit_risk_prop.pkl')
        feature_importances = model.feature_importances_
        #recupération des noms de colonnes
        column_names = data.columns
        #creation d'un dictionnaire pour assoicier chaque colonne à son importance
        feature_importances_dict = dict(zip(column_names, feature_importances))
        #tri des colonnes par importance décroissante
        sorted_features = sorted(feature_importances_dict.items(),key = lambda x:[1], reverse=True)

        #extration des noms de colonnes trié et de leurs importances
        sorted_columns, sorted_importances = zip(*sorted_features)
        fig, ax = plt.subplots (figsize=(10,6))
        ax.bar(sorted_columns, sorted_importances)
        ax.set_xticklabels(sorted_columns, rotation = 90)
        ax.set_xlabel('colonnes')
        ax.set_ylabel('Importance')
        ax.set_title ('Importance des colonnes')
        #affichage de la figure 
        st.pyplot(fig)

#menu Prediction Score_risK         
if choice == "CLASSIFICATION":
    st.markdown("<h1 style='text-align:center;color: red;'>CLASSIFICATION DES INDIVIDUS </h1>",unsafe_allow_html=True)
    
    #inporter un data set pour faire une classification
    
    uploaded_file = st.sidebar.file_uploader("Importer un le DATASET", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        from pycaret.classification import *
        
        # Charger les données
        #data = pd.read_csv('credit_risk_predict.csv')

        # Initialiser l'environnement de classification
        clf = setup(data=data, target='loan_status', session_id=123)

        # Comparer les modèles
        compare_models()

        # Créer le modèle
        model = create_model('lightgbm')

        # Afficher les performances du modèle
        plot_model(model, plot='auc')

        # Prédire les valeurs sur de nouvelles données
        predictions = predict_model(model)
        
        # Afficher les prédictions
        predictions.prediction_label.replace(0, "NO Credit_Risk", inplace = True)
        predictions.prediction_label.replace(1, "Credit a riskque",inplace = True)
        st.write(predictions)

        #pour télécharger le dataset deja classé
        def filedownload(data):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="credit_risk_dataset_1.csv">Download CSV File</a>'
            return href
        button = st.button("Download (telecharger)")
        if button:
            st.markdown(filedownload(predictions), unsafe_allow_html=True)

    
    

#menu Prédiction
if choice == "PRÉDICTION":

    #image de font 

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: 250%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: top left; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


    tab1, tab2 = st.tabs([":clipboard: Data",":bar_chart: Visualisation", ":angry: :smile: Prediction"])
    uploaded_file = st.sidebar.file_uploader("Importer un le DATASET", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        with tab1:
            st.subheader("Loaded dataset")
            st.write(df)

        with tab2:
            st.subheader("REPRESENTATION")

        #with tab3:
        #    data =load_model("credit_risk_prop")
        #    prediction = data.predict(df)
        #    st.subheader('Prediction')
        #    pp = pd.DataFrame(prediction,columns=["Prediction"])
        #    ndf = pd.concat([df,pp],axis=1)
        #    ndf.Prediction.replace(0, "NO Credit_Risk", inplace = True)
        #    ndf.Prediction.replace(1, "Credit_Risk", inplace = True)
        #    st.write(ndf)
        #    def filedownload(df):
        #        csv = df.to_csv(index=False)
        #        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        #        href = f'<a href="data:file/csv;base64,{b64}" download="credit_risk_dataset_prop.csv">Download CSV File</a>'
        #        return href
        #    button = st.button("Download (telecharger)")
        #    if button:
        #        st.markdown(filedownload(ndf), unsafe_allow_html=True)

    
#            elif not authentication_status:
#                with info:
#                    st.error('Incorrect Password or username')
#            else:
#                with info:
#                    st.warning('Please feed in your credentials')
#        else:
#            with info:
#                st.warning('Username does not exist, Please Sign up')


#except:
#    st.success('Refresh Page')
# Import libraries
from nltk.corpus import stopwords
import pickle
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

# Open cleaned dataframe
with open ('cleaned_data','rb') as file:
	movie_df = pickle.load(file)

#################### Named Entities ####################

# Named Entity Declaration
entities = {' new york ':' new_york ',
            ' los angeles ':' los_angeles ',
            ' van helsing ':' van_helsing ',
            ' high school ':' high_school ',
            ' united state ':' united_states ',
            ' united states ':' united_states ',
            ' hong kong ':' hong_kong ',
            ' kingdom ':' king ',
            ' world war ':' world_war ',
            ' world_war ii ': ' world_war ',
            ' gun shot ':' gun_shot ',
            ' performance ':' perform ',
            ' commit suicide ':' commit_suicide ',
            ' central park ':' central_park ',
            ' police officer ':' police_officer ',
            ' steal money ':' steal_money ',
            ' college student ':' college_student ',
            ' set free ':' set_free ',
            ' haunt house ':' haunted_house ',
            ' marry ':' marriage ',
            ' investigate ':' investigation ',
            ' develops ':' develop ',
            ' teacher ':' teach ',
            ' form story ':' form_story ',
            ' dr ':' doctor ',
            ' best friend ':' best_friend ',
            ' childhood friend ':' childhood_friend ',
            ' close friend ':' close_friend ',
            ' car accident ':' car_accident ',
            ' commits suicide ':' suicide ',
            ' commit suicide ':' suicide ',
            ' happily ':' happy ',
            ' small town ':' small_town ',
            ' writer ':' write ',
            ' writes ':' write ',
            ' heart attack ':' heart_attack ',
            ' die ':' death ',
            ' dead ':'death ',
            ' small town ':' small_town ',
            ' player ':' play ',
            ' night club ':' night_club ',
            ' singer ':' sing ',
            ' police station ':' police_station ',
            ' destroyed ':' destroy ',
            ' competition ':' compete ',
            ' cross country ':' cross_country ',
            ' marries ':' marriage ',
            ' air force ':' air_force ',
            ' married ':' marriage ',
            ' newly wed ':' newly_wed ',
            ' romantically ':' romantic ',
            ' seek revenge ':' seek_revenge ',
            ' reading ':' read ',
            ' sings ':' sing '
           }

# Named Entity function
def named_entities(text):
    '''
    Replaces all named entities
    before vectorization.
    '''
    for k, v in entities.items():
        text = text.replace(k, v)
    return text

# Named Entity Application
movie_df["Plot"] = movie_df["Plot"].apply(lambda x: named_entities(x))

#################### Stopwords ####################

# Add general English stopwords without apostrophes
more_stopwords = []

for word in list(stopwords.words('english')):
    more_stopwords.append(word.replace('\'',''))

# Join's the stop words above to the standard English list
stop_words = text.ENGLISH_STOP_WORDS.union(more_stopwords)

# Misc Category
other = ['rama','later','night','away','manner','door',
         'left','new','away','way','process','purpose','sens',
         'el','pas','section','good','multiple','attractive',
         'favorite','calcutta','interested','repeatedly','thing',
         'von','time','leaf','dinner','babu','big','inside',
         'outside','window','rao','day','hand','hard','end',
         'yearold','face','second','unable','reason','happens'
         ,'meantime','problem','life','true','past','care','sight'
         ,'eventually','year','ago','long','old','lose','present',
         'great','need','age','soon','head','happy','honest','head',
         'arm','role','department','result','room','wall','sudden',
         'suddenly','house','hall','different','elder','beautiful',
         'young','handsome','real','actually','truth','really','mistake',
         'set','large','despite','final','trip','store','east','park',
         'small','social','bad','couple','home','mate','exact','london',
         'india','paris','case','fall']

# Verbs
verb = ['come','leave','stay','say','tell','make','help','meet',
        'know','like','asks','use','want','follow','stake','kill',
        'pull','try','visit','return','let','stop','start','ask',
        'miss','lot','talk','reveals','run','begin','explains',
        'decides','change','open','run','walk','attempt','plan',
        'refuse','complete','decision','inform','pick','confuse',
        'attach','parking','approach','dislike','raise','lift',
        'increase','choose','dy','rest','look','rid','look',
        'realizes','spend','arrives','fail','turn','hold',
        'confronts','turn','realize','chase','knock','grab',
        'cause','throw','agrees','include','cause','manages',
        'arrive','happen','decide','reach','ride','fall','appear',
        'wake','watch','eat','cut','lock','attack','watch','hears',
        'wish','revolves','sends','play','sent','feel','think','focus',
        'described','save','share','attend','board','cross','accompany',
        'grow','save','lead','played','join','involve','involves',
        'receives','love']

# People
people = ['man','woman','girl','boy','sir','madam','professor',
         'guy','doc','boss','mr','person','lady','men']
         
# Names
name = ['michael','peter','sam','john','jane','max','tim',
        'curtis','jimmy','charlie','elizabeth','mike','paul',
        'nick','jimmy','eddie','tony','henry','paul','joes',
        'joe','emily','lily','amy','edward','frank','johnny',
        'helen','ben','diane','frank','johnny','martin','george',
        'anne','lucy','linda','leo','carl','alice','bobby',
        'martha','tom','jerry','rachel','ross','jenny','ann',
        'jennifer','lloyd','raj','walter','james','mary','steve',
        'billy','norman','ann','ray','jonathan','arthur','nikki',
        'frederick','jason','jessica','david','mia','katherine',
        'judy','steven','julie','susan','cynthia','shane','allan',
        'alex','sally','kim','lou','victor','ash','harris','wendy',
        'adam','grace','jim','glen','terry','al','margaret','carrie',
        'danny','alan','robert','christine','jack','thomas','ralph',
        'charlotte','nancy','simon','jake','pete','joseph','jacob',
        'hank','kelly','anna','stephen','dan','sean','larry','sarah',
        'karl','jackie','carter','scott','pete','harry','kate','eve',
        'phil','dean','cole','graham','jordan','phyllis','bob','sue',
        'rita','michelle','diana','mark','daniel','matt','lisa','duke',
        'morgan','marie','raymond','karen','maria','todd','janet','fred',
        'richard','annie','drake','julia','francis','charles','stewart',
        'richards','olivia','lawrence','lee','jeff','ellen','andy','andrew',
        'ruth','ed','miller','jones','taylor','kumar','shankar','ajay',
        'signh','prakash','prasad','joan','rahul','li','chris','singh',
        'khan','mohan','krishna','ravi','rajah','anand','vijay','kapoor',
        'raja','radha','lakshmi']

# Family
family = ['family','son','brother','sister','child','wife','daughter',
          'mother','husband','father','parent','uncle','cousin','grandfather',
          'aunt']

add_stop_words = other + verb + people + name + family

# Join's the stop words above to the standard English list
stop_words = stop_words.union(add_stop_words)

#################### Vectorization ####################

# Create the vectorizer object
vectorizer = TfidfVectorizer(ngram_range = (1,3), stop_words = stop_words, min_df = .01, binary = False)

# Create the doc_word sparse matrix
doc_word = vectorizer.fit_transform(movie_df["Plot"])

# Create a dataframe for easy labeleled viewing
doc_word_df = pd.DataFrame(doc_word.toarray(), columns = vectorizer.get_feature_names())

#################### NMF Modeling ####################

# Create and NMF object with 35 topics
nmf = NMF(n_components = 35)

# Fit the doc_word sparse matrix
doc_topic = nmf.fit_transform(doc_word)

#################### Final Movie Identification ####################

# Create a list of movie ids and movie titles
movie_ids = movie_df["Title"].index.tolist()
movie_titles = movie_df["Title"].tolist()

# Create dictionarys to access them both ways
movie_to_id = {}
id_to_movie = {}

# Populate movie to id
for idx in range(len(movie_titles)):
    movie_to_id[movie_titles[idx]] = movie_ids[idx]

# Populate id to movie
for idx in range(len(movie_titles)):
    id_to_movie[movie_ids[idx]] = movie_titles[idx]

#################### Pickling ####################

# Export data for recommending
with open('modeled_data','wb') as file:
	pickle.dump(movie_to_id, file)
	pickle.dump(id_to_movie, file)
	pickle.dump(movie_titles, file)
	pickle.dump(doc_topic, file)
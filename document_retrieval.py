import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from math import *
import os


path=os.getcwd()
docs=os.listdir(path)

keywords=[]
doc_keywords={}

porter=PorterStemmer()

for doc in docs:
        f=open(path+"\\"+doc,"r")
        tokens=nltk.word_tokenize(f.read())
        raw=[w.lower() for w in tokens if not w.lower() in stopwords.words('english')]
        doc_keywords[doc]=[porter.stem(t) for t in raw]
        keywords=keywords+doc_keywords[doc]

keywords=sorted(set(keywords))
     
tfreq={}

for doc in docs:
        freq={}
        for word in keywords:
                count=0
                for w in doc_keywords[doc]:
                       if w==word:
                        count+=1
                if count==0:
                        freq[word]=count
                else:
                        freq[word]=1+log(count)
        tfreq[doc]=freq
idfreq={}

for word in keywords:
        count=0
        for doc in docs:
                if word in doc_keywords[doc]:
                        count+=1
        idfreq[word]=log(len(docs)/count)

tfidf={}

for doc in docs:
        freq={}
        for word in keywords:
               freq[word]=tfreq[doc][word]*idfreq[word]
        tfidf[doc]=freq       

    
def pearson_correlation(x, y):
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)

    subtracted_mean_x = [i - mean_x for i in x]
    subtracted_mean_y = [i - mean_y for i in y]

    x_times_y = [a * b for a, b in list(zip(subtracted_mean_x, subtracted_mean_y))]

    x_squared = [i * i for i in x]
    y_squared = [i * i for i in y]

    return sum(x_times_y) / sqrt(sum(x_squared) * sum(y_squared))

def square_rooted(x):
 
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 
   numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)
   return round(numerator/float(denominator),3)    

def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality) 


search_sentence=input("Enter text to Search ")

tokens=nltk.word_tokenize(search_sentence)
raw=[w.lower() for w in tokens if not w.lower() in stopwords.words('english')]
search_sentence=[porter.stem(t) for t in raw]

tf_sent={}
for word in keywords:
        count=0
        for w in search_sentence:
                if w==word:
                        count+=1
        if count==0:
                tf_sent[word]=count
        else:
                tf_sent[word]=1+log(count)
        
tfidf_sent={}

for word in keywords:
        tfidf_sent[word]=tf_sent[word]*idfreq[word]

cosine_sim={}   
pearson_corr={}
jaccard_sim = {}

try:     
        for doc in docs:
                cosine_sim[doc]=cosine_similarity(tfidf[doc].values(),tfidf_sent.values())
                jaccard_sim[doc]=jaccard_similarity(doc_keywords[doc],search_sentence)
                pearson_corr[doc]=pearson_correlation(tfidf[doc].values(),tfidf_sent.values())
                
        points={}        
        cosine_sim_points={}
        jaccard_sim_points={}
        pearson_corr_points={}       

        count=1
        for key, value in sorted(cosine_sim.items(), key=lambda item: item[1]):
                cosine_sim_points[key]=count
                count+=1
       
        print("\nCosine Similarity Score:\n")
        print(cosine_sim_points)

        count=1       
        for key, value in sorted(pearson_corr.items(), key=lambda item: item[1]):
                pearson_corr_points[key]=count
                count+=1

        print("\nPearson Correlation Score\n")       
        print(pearson_corr_points)

        count=1
        for key, value in sorted(jaccard_sim.items(), key=lambda item: item[1]):
                jaccard_sim_points[key]=count
                count+=1
       
        print("\nJaccard Similarity Score\n")       
        print(jaccard_sim_points)

        for doc in docs:
                points[doc]=0.4*cosine_sim_points[doc]+0.3*pearson_corr_points[doc]+0.3*jaccard_sim_points[doc]
      
        print("\nFinal Rankings of Documents\n")
        rank=1        
        for key, value in sorted(points.items(), key=lambda item: item[1],reverse=True):
                print("%s: %s" % (key, rank))
                rank+=1
except:
        print("\nNo document related to search query found\n")
#By Mahboobeh "Mabi" Harandi
#At iSchool, Syracuse University
#Spring 2018


# -*- coding: utf8 -*-

import nltk,csv,numpy
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import itertools
import datetime
import pandas as pd

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

stop_words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
porter = nltk.PorterStemmer()


def prepration(file_name):
    with open(file_name, encoding="ISO-8859-1") as ss:
        csvreader = csv.reader(ss)
        with open("%s_prep2.csv" %file_name, "w") as output:
            theoutput = csv.writer(output)
            for rows in csvreader:
                if file_name == "snapshot.csv":
                    # lowercase
                    comments = rows[2].lower()
                else:
                    comments = rows[8].lower()
                # remove any patterns of http
                filtered_words = re.sub(r"(http\S+|\(\s*http(\s+\S+)*\)|http(\s+\S+)*)", '', comments, flags=re.MULTILINE)

                # replace non alphnumeric characters with a whitespace
                filtered_words = re.sub(r"[^a-zA-Z0-9]", ' ', filtered_words,
                                        flags=re.MULTILINE)
                # replace the varation of gazal bustard monkey if the terms gazal bustard monkey are not in the comment
                m1 = re.search("(thomson’s|thomsons|thomson)", filtered_words)
                m2 = re.search("(grant’s|grants|grant)", filtered_words)
                m3 = re.search("(gazelle|gazelles)", filtered_words)
                if m1 and not m3:
                    filtered_words = re.sub(r"(thomson’s|thomsons|thomson)", 'gazelle', filtered_words, flags=re.MULTILINE)
                if m2 and not m3:
                    filtered_words = re.sub(r"(grant’s|grants|grant)", 'gazelle', filtered_words, flags=re.MULTILINE)

                m4 = re.search("(kori)", filtered_words)
                m5 = re.search("(bustard|bustards)", filtered_words)
                if m4 and not m5:
                    filtered_words = re.sub(r"(kori)", 'bustard', filtered_words, flags=re.MULTILINE)

                m6 = re.search("(vervet)", filtered_words)
                m7 = re.search("(samango)", filtered_words)
                m8 = re.search("(monkey)", filtered_words)
                if m6 and not m8:
                    filtered_words = re.sub(r"(vervet|vervets)", 'monkey', filtered_words, flags=re.MULTILINE)
                if m7 and not m8:
                    filtered_words = re.sub(r"(samango|samngos)", 'monkey', filtered_words, flags=re.MULTILINE)



                # get lemma
                tokens = nltk.word_tokenize(filtered_words)
                pos = nltk.pos_tag(tokens)
                token_list = [wordnet_lemmatizer.lemmatize(element[0], get_wordnet_pos(element[1])) for element in pos]
                newtokenlist = ' '.join(token_list)
                filtered_words = " ".join(filter(lambda word: word not in stop_words, newtokenlist.split()))
                if file_name == "snapshot.csv":
                    theoutput.writerow((filtered_words, rows[3], rows[4], rows[6]))
                else:
                    theoutput.writerow((rows[3], filtered_words, rows[11], rows[12]))






def count_word_classes(rows, file_name):
    science_words = 0
    if file_name == "snapshot_prep2.csv":
        with open("ss-classes.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[0])
    else:
        with open("gs-classes.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[1])

    for tokens in tokens_list:

        newtokens = tokens.split()
        if newtokens in csvreader:
            science_words += 1
    return (science_words)



def count_word_non_classes(rows, file_name):
    science_words = 0
    if file_name == "snapshot_prep2.csv":
        with open("ss-non-class.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[0])
    else:
        with open("gs-non-class.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[1])

    for tokens in tokens_list:

        newtokens = tokens.split()
        if newtokens in csvreader:
            science_words += 1
    return (science_words)





def classes(file_name):
    with open(file_name, encoding="ISO-8859-1") as input:
        csvreader = csv.reader(input)
        with open("%s_class_count2.csv" %file_name, "w") as output:
            theoutput = csv.writer(output)
            for rows in csvreader:
                science_count = count_word_classes(rows, file_name)
                if file_name == "snapshot_prep2.csv":
                    comments = rows[0]
                    len_comment = len(re.findall(r'\w+', comments))
                else:
                    comments = rows[1]
                    len_comment = len(re.findall(r'\w+', comments))
                theoutput.writerow((rows[0], rows[1], rows[2], rows[3], len_comment, science_count))



def non_classes(file_name):
    with open(file_name, encoding="ISO-8859-1") as input:
        csvreader = csv.reader(input)
        with open("%s_non_class_count2.csv" %file_name, "w") as output:
            theoutput = csv.writer(output)
            for rows in csvreader:

                science_count = count_word_non_classes(rows, file_name)
                if file_name == "snapshot_prep2.csv":
                    comments = rows[0]
                    len_comment = len(re.findall(r'\w+', comments))
                else:
                    comments = rows[1]
                    len_comment = len(re.findall(r'\w+', comments))
                theoutput.writerow((rows[0], rows[1], rows[2], rows[3], len_comment, science_count))




def min__max_date():

    input_file = pd.read_csv("gravityspy_non_class_order2.csv" , encoding="ISO-8859-1")
    mindate = input_file.groupby(['user_login'])['Date1'].min()
    maxdate = input_file.groupby(['user_login'])['Date1'].max()

    with open("gravityspy_non_class_order_min2.csv", 'w') as output:
        mindate.to_csv(output, sep=',')
    with open("gravityspy_non_class_order_max2.csv", 'w') as output:
        maxdate.to_csv(output, sep=',')




def create_weeknum():

    input_file = csv.reader(open("gravityspy_non_class_minmax2.csv"))
    with open("gravityspy_non_class_weeknum2.csv", "w") as output:
        theoutput = csv.writer(output)
        for a in input_file:
            start = datetime.datetime.strptime(a[2], "%Y-%m-%d")
            end = datetime.datetime.strptime(a[4], "%Y-%m-%d")
            for n in range(int((end - start).days)):
                if start + datetime.timedelta(n * 7) <= end:
                    startweek = start + datetime.timedelta(n * 7)
                    endweek = startweek + datetime.timedelta(6)
                    theoutput.writerow((a[1], startweek, endweek, n))


def proj_individual_week():
    with open('gravityspy_non_class_order2.csv', encoding="ISO-8859-1") as csvfile1:
        with open("gravityspy_non_class_weeknum_ready2.csv", encoding="ISO-8859-1") as csvfile2:
            with open("gravityspy_non_class_week.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                rows1 = [row for row in reader1]
                rows2 = [row for row in reader2]
                for rowa in rows1:
                    for rowb in rows2:
                        if rowa[3] == rowb[1]:
                            if rowa[7] >= rowb[5] and rowa[7] <= rowb[6]:
                                theoutput.writerow(
                                    ("gravityspy", "nonclass", rowa[3], rowa[5], rowa[6], rowa[7], rowb[4]))





#file_name= "snapshot.csv"
#file_name ="gravityspy.csv"
#prepration(file_name)
#file_name="snapshot_prep2.csv"
#file_name = "gravityspy_prep2.csv"
#classes(file_name)
#non_classes(file_name)
#convert created_at to a year-month-day format and order it before using the next function
#min__max_date()
#create_weeknum()
#proj_individual_week()
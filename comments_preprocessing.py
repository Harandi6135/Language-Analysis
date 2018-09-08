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
        with open("ss_non_classes2.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[0])
    else:
        with open("gs_non_classes2.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[1])

    for tokens in tokens_list:
        newtokens = tokens.split()
        if newtokens in csvreader:
            science_words += 1
    return (science_words)



def count_word_others(rows, file_name):
    science_words = 0
    if file_name == "snapshot_prep2.csv":
        with open("others_ss_no_users.csv", encoding="ISO-8859-1") as input:
            csvreader = list(csv.reader(input))
            tokens_list = nltk.word_tokenize(rows[0])
    else:
        with open("others_gs_no_users.csv", encoding="ISO-8859-1") as input:
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


def others(file_name):
    with open(file_name, encoding="ISO-8859-1") as input:
        csvreader = csv.reader(input)
        with open("%s_others_count2.csv" %file_name, "w") as output:
            theoutput = csv.writer(output)
            for rows in csvreader:

                science_count = count_word_others(rows, file_name)
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

    input_file = pd.read_csv("snapshot_others_order2.csv" , encoding="ISO-8859-1")
    mindate = input_file.groupby(['user_login'])['Date1'].min()
    maxdate = input_file.groupby(['user_login'])['Date1'].max()

    with open("snapshot_others_order_min2.csv", 'w') as output:
        mindate.to_csv(output, sep=',')
    with open("gsnapshot_others_order_max2.csv", 'w') as output:
        maxdate.to_csv(output, sep=',')




def create_weeknum():

    input_file = csv.reader(open("snapshot_class_minmax2.csv"))
    with open("snapshot_class_weeknum2.csv", "w") as output:
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
    with open('snapshot_class_order2.csv', encoding="ISO-8859-1") as csvfile1:
        with open("snapshot_class_weeknum_ready2.csv", encoding="ISO-8859-1") as csvfile2:
            with open("snapshot_class_week.csv", "w") as output:
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
                                    ("snapshot", "class", rowa[3], rowa[5], rowa[6], rowa[7], rowb[4]))


##This is the start point


#we add the beginning and the end of the first month for each user based on their first date of commenting
def first_month_comment():
    with open('gravityspy_others_order_min2.csv', encoding="ISO-8859-1") as csvfile1:
        with open("GS-Month.csv", encoding="ISO-8859-1") as csvfile2:
            with open("gs_others_month.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                rows1 = [row for row in reader1]
                rows2 = [row for row in reader2]
                for rowa in rows1:
                    for rowb in rows2:
                        csv1_date = datetime.datetime.strptime(rowa[1], '%Y-%m-%d')
                        csv2_date = datetime.datetime.strptime(rowb[1], '%Y-%m-%d')
                        if csv1_date >= csv2_date and csv1_date <=  csv2_date + datetime.timedelta(30):
                            theoutput.writerow(
                                    ("gravityspy", "others", rowa[0], rowa[1], csv2_date, csv2_date + datetime.timedelta(30), rowb[0]))


def proj_month():
    with open('SS-Month.csv', encoding="ISO-8859-1") as csvfile1:
        with open("SS-Month-out.csv", "w") as output:
            theoutput = csv.writer(output)
            reader1 = csv.reader(csvfile1)
            rows1 = [row for row in reader1]
            for rowa in rows1:
                csv1_date = datetime.datetime.strptime(rowa[1], '%Y-%m-%d')
                theoutput.writerow(
                    ( rowa[0], rowa[1], csv1_date + datetime.timedelta(30)))



# def st_week_comment_data():
#     with open('gs_ss_terms_weeks.csv', encoding="ISO-8859-1") as csvfile1:
#         with open("gs_ss_terms_months.csv", encoding="ISO-8859-1") as csvfile2:
#             with open("gs_ss_terms_month_data.csv", "w") as output:
#                 theoutput = csv.writer(output)
#                 reader1 = csv.reader(csvfile1)
#                 reader2 = csv.reader(csvfile2)
#                 rows1 = [row for row in reader1]
#                 rows2 = [row for row in reader2]
#                 for rowa in rows1:
#                     for rowb in rows2:
#                         csv1_date = datetime.datetime.strptime(rowa[5], '%Y-%m-%d')
#                         csv2_date = datetime.datetime.strptime(rowb[3], '%Y-%m-%d')
#                         if rowa[0] == rowb[0] and rowa[1] == rowb[1] and rowa[2] == rowb[2]:
#                             if csv1_date >= csv2_date and csv1_date <= csv2_date + datetime.timedelta(30):
#                                 theoutput.writerow(
#                                     (rowa[0], rowa[1], rowa[2], rowa[3], rowa[4], rowa[5], rowb[4]))



def data_before_individual_week():
    with open("gravityspy_others_order2.csv", encoding="ISO-8859-1") as csvfile1:
        with open("data_count6.csv", "w") as output:
            theoutput = csv.writer(output)
            reader1 = csv.reader(csvfile1)
            rows1 = [row for row in reader1]
            for rowa in rows1:
                theoutput.writerow(
                    ("gravityspy", "others", rowa[3], rowa[5], rowa[6], rowa[7]))






def other_terms():
    temp1 = []
    temp2 = []
    with open('ss_non_classes_all2.csv', encoding="ISO-8859-1") as csvfile1:
        with open("ss-class-nonclass.csv", encoding="ISO-8859-1") as csvfile2:
            with open("others_ss.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                for row in reader1:
                    temp1.append(row[0])
                for row2 in reader2:
                    temp2.append(row2[0])
                output = [a for a in temp1 + temp2 if (a not in temp2)]
                for each in output:
                    theoutput.writerow([each])



def remove_usernames():
    temp1 = []
    temp2 = []
    with open('others_ss.csv', encoding="ISO-8859-1") as csvfile1:
        with open("ss_comment_userlogin_unique.csv", encoding="ISO-8859-1") as csvfile2:
            with open("others_ss_no_users.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                for row in reader1:
                    temp1.append(row[0])
                for row2 in reader2:
                    temp2.append(row2[0].lower())
                output = [a for a in temp1 + temp2 if (a not in temp2)]
                for each in output:
                    theoutput.writerow([each])



# input_file = pd.read_csv("ss_comment_userlogin.csv" , encoding="ISO-8859-1")
# unique = input_file.groupby(['user_login'])['user_login'].size()
# with open("ss_comment_userlogin_unique.csv", 'w') as output:
#     unique.to_csv(output, sep=',')






#file_name= "snapshot.csv"
#file_name ="gravityspy.csv"
#prepration(file_name)
#file_name="snapshot_prep2.csv"
#file_name = "gravityspy_prep2.csv"
#classes(file_name)
#non_classes(file_name)
#others(file_name)
#convert created_at to a year-month-day format and order it before using the next function
#min__max_date()
#merge min and max file into one before using the next funtion
#create_weeknum()
#convert the date to the year-month-day before using the newxt function
#proj_individual_week()



#first_month_comment()
#st_week_comment_data()
#proj_month()
#data_before_individual_week()

#other_terms()

#remove_usernames()

def year_no_pub():

    input_file = pd.read_csv("3terms_first_month.csv" , encoding="ISO-8859-1")
    count = input_file.groupby(['user', 'Project'])['Date1'].nunique()


    with open("users_comment_num_day.csv", 'w') as output:
        count.to_csv(output, sep=',')


#year_no_pub()

def diff():
    temp1 = []
    temp2 = []
    with open('out_G.csv', encoding="ISO-8859-1") as csvfile1:
        with open("gs_non_classes_all2 copy.csv", encoding="ISO-8859-1") as csvfile2:
            with open("remaindersG.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                for row in reader1:
                    temp1.append(row[1])
                for row2 in reader2:
                    temp2.append(row2[0])
                output = [a for a in temp1 + temp2 if (a not in temp2)]
                for each in output:
                    theoutput.writerow([each])



#diff()

def max_activity_First_month():

    input_file = pd.read_csv("df_active_users.csv" , encoding="ISO-8859-1")
    count = input_file.groupby(['user', 'Project', 'term'])['Date1'].max()


    with open("df_active_users_lastdate_firstmonth.csv", 'w') as output:
        count.to_csv(output, sep=',')


#max_activity_First_month()

def After_first_month_comment():
    with open('experienced_users.csv', encoding="ISO-8859-1") as csvfile1:
        with open("SS-Month.csv", encoding="ISO-8859-1") as csvfile2:
            with open("experienced_users_Month_S.csv", "w") as output:
                theoutput = csv.writer(output)
                reader1 = csv.reader(csvfile1)
                reader2 = csv.reader(csvfile2)
                rows1 = [row for row in reader1]
                rows2 = [row for row in reader2]

                for rowa in rows1:
                    for rowb in rows2:
                        if rowa[1] == "snapshot":
                            csv1_date = datetime.datetime.strptime(rowa[6], '%Y-%m-%d')
                            csv2_date = datetime.datetime.strptime(rowb[1], '%Y-%m-%d')
                            if csv1_date >= csv2_date and csv1_date <= csv2_date + datetime.timedelta(30):
                                theoutput.writerow(
                                    ( rowa[1], rowa[2], rowa[3], rowa[4], rowa[5], rowa[6], rowb[0], csv2_date,
                                     csv2_date + datetime.timedelta(30)))

After_first_month_comment()
#,project,term,user_login,total,count,Date1
#“mon-num”,”st-date”
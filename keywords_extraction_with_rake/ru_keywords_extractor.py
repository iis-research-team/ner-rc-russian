from multi_rake import Rake
import glob
import io
import os
from openpyxl import load_workbook
from pymystem3 import Mystem
import json

def remove_verbs(text_ru):
    m = Mystem()
    full_info = m.analyze(text_ru)
    result_text_ru = ""
    for element in full_info:
        check = 1
        if element.get('analysis') is not None:
            if len(element['analysis']) > 0:
                if element['analysis'][0]['gr'][0] == 'V':
                    check = 0
    
        if check == 1:
            result_text_ru += element['text']
    return result_text_ru

raw_files = glob.glob("texts/*.txt")
xlsx_files_path = "ans_xlsxs/"
stopwords_file = "stopwords.txt"

stopwords_list = {}
with io.open(stopwords_file, encoding='utf-8') as file:
    stopwords_list = set(line.strip() for line in file.readlines())

# this is try to use all text as text for stopwords (result is bad)
#text_for_stopwords = ""
#for path in raw_files:
#    with io.open(path, encoding='utf-8') as file:
#        for line in file:
#            text_for_stopwords = text_for_stopwords + line-

total_tp = 0
total_all = 0
total_sz = 0
for path in raw_files:
    text_ru = ''
    with io.open(path, encoding='utf-8') as file:
        for line in file:
            text_ru = text_ru + line
    rake = Rake(
        min_chars=3,
        max_words=5,
        min_freq=1,
        language_code = 'ru'
        , stopwords=stopwords_list
    )

    # apply Mystem to current text to delete form it all verbs form
    #text_ru = remove_verbs(text_ru)
    keywords = rake.apply(text_ru)
    #keywords = rake.apply(text_ru, text_for_stopwords=text_for_stopwords) # doesnt work better

    # TODO how to choose the last good candidate
    last_good = 0
    for el in keywords:
        if el[1] >= 2:
            last_good += 1

    with io.open('results/' + path.split('\\')[1], mode='w', encoding='utf-8') as outfile:
        outfile.write('\n'.join('%s %s' % x for x in keywords[:last_good]))
    
    filename = os.path.basename(path).split('.')[0]
    wb = load_workbook(xlsx_files_path + filename + '.xlsx')
    sheet = wb.active
    ind = 2 # 1-based and 1st is "token"
    xlsx_word_list = []
    none_count = 0
    while (none_count != 2): # first is delimiter between title and annotation, second is after last element
        word = sheet['B' + str(ind)].value
        if word == None:
            none_count = none_count + 1
            xlsx_word_list.append("")
        else:
            xlsx_word_list.append(word)
        ind = ind + 1

    ind = 0
    TP = 0
    ALL = 0
    while (ind < len(xlsx_word_list)):
        term = sheet['E' + str(ind + 2)].value
        if term == "B-TERM":
            ALL = ALL + 1
            for keyphrase in keywords[:last_good]:
                keyphrase_words = keyphrase[0].split()
                match = 0
                if keyphrase_words[0].lower() == xlsx_word_list[ind].lower():
                    kw_ind = 1
                    next_ind = ind + kw_ind
                    all_match = 1
                    while next_ind < len(xlsx_word_list):
                        next_term = sheet['E' + str(next_ind + 2)].value
                        if next_term == "I-TERM":
                            if (kw_ind >= len(keyphrase_words) or (kw_ind < len(keyphrase_words) and not keyphrase_words[kw_ind].lower() == xlsx_word_list[next_ind].lower())):
                                all_match = 0
                                break
                        else:
                            break
                        kw_ind = kw_ind + 1
                        next_ind = ind + kw_ind
                    
                    if all_match == 1 and kw_ind == len(keyphrase_words):
                        TP = TP + 1
                        ind = next_ind
                        match = 1
                if match == 1:
                    break
            ind = ind + 1
        else: # can be only O
            ind = ind + 1
    cur_precision = TP / ALL
    cur_recall = TP / last_good
    print(filename)
    print("precision: " + str(cur_precision))
    print("recall: " + str(cur_recall))

    total_tp += TP
    total_all += ALL
    total_sz += last_good

print("RESULTS:")
print("precision: " + str(total_tp / total_all))
print("recall: " + str(total_tp / total_sz))

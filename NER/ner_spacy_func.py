# This script contains the functions that, given an input sentence, return the list of named entities contained in that sentence
# using SpaCy english and multilingual models for NER


# For general use:

def ner_spacy_english_func(testsentence):
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(testsentence)
    doc_ents_list = list(doc.ents)
    for i in range(len(doc_ents_list)):
        doc_ents_list[i] = str(doc_ents_list[i])
    return doc_ents_list


# For testing accuracy: (only load the model once, before calling the function)

def ner_spacy_english_func_test(testsentence,nlp):
    doc = nlp(testsentence)
    doc_ents_list = list(doc.ents)
    for i in range(len(doc_ents_list)):
        doc_ents_list[i] = str(doc_ents_list[i])
    return doc_ents_list


# For general use:

def ner_spacy_multilingual_func(testsentence):
    import spacy
    
    nlp = spacy.load('xx_ent_wiki_sm')
    doc = nlp(testsentence)
    doc_ents_list = list(doc.ents)
    for i in range(len(doc_ents_list)):
        doc_ents_list[i] = str(doc_ents_list[i])
    return doc_ents_list


# For testing accuracy: (only load the model once, before calling the function)

def ner_spacy_multilingual_func_test(testsentence, nlp):
    doc = nlp(testsentence)
    doc_ents_list = list(doc.ents)
    for i in range(len(doc_ents_list)):
        doc_ents_list[i] = str(doc_ents_list[i])
    return doc_ents_list
        
# Function for general use:

def ner_masakhaner_func(testsentence):
    # This function takes as input a string sentence and return a list with all the named entities in the order 
    # as they appear in the sentence as identified by Masakhaner Afro-XLMR NER model

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    import string

    tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
    model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer) 

    ner_results = nlp(testsentence)


    ne_per_sentence = list()
    elem_indx = -1

    for item in ner_results:
        ne = item['word'][1:].translate(str.maketrans('','',string.punctuation))
        if item['entity'][0] == 'B':
            ne_per_sentence.append(ne)
            elem_indx += 1
        else:
            ne_per_sentence[elem_indx] = ne_per_sentence[elem_indx]+' '+ne

    return ne_per_sentence



# For testing accuracy: (only load the model and the tokenizer once, before calling the function)

def ner_masakhaner_func_test(testsentence, nlp):
    # This function takes as input a string sentence and return a list with all the named entities in the order 
    # as they appear in the sentence as identified by Masakhaner Afro-XLMR NER model
    import string
    
    ner_results = nlp(testsentence)


    ne_per_sentence = list()
    elem_indx = -1

    for item in ner_results:
        ne = item['word'][1:].translate(str.maketrans('','',string.punctuation))
        if item['entity'][0] == 'B'or (len(ne_per_sentence)==0 and item['entity'][0] == 'I'):
            ne_per_sentence.append(ne)
            elem_indx += 1
        elif item['entity'][0] == 'I':
            ne_per_sentence[-1] = ne_per_sentence[-1]+' '+ne

    return ne_per_sentence
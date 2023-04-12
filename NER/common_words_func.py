# Given two input sentences, this function outputs a list of the common words; accounts for compound words.

def common_words_func_test(sentence1,sentence2):
    import string
    import re
    
    common_words_set = set(sentence1.translate(str.maketrans('','',string.punctuation)).split()).intersection(set(sentence2.translate(str.maketrans('','',string.punctuation)).split()))
    
    ne_prev = 0  # Check if the previous word was a named entity
    ne_index = -1  # Index of named entities in the final named entities list
    ne_list = []  # Final list of named entities as common words between the two sentences
    
    for word in sentence1.translate(str.maketrans('','',string.punctuation)).split():
        if word in common_words_set:
            if ne_prev==1:
                ne_list[ne_index] = ne_list[ne_index]+" "+word
            else:
                ne_list.append(word)
                ne_index += 1
            ne_prev = 1
        else:
            ne_prev = 0

    ne_list.sort(key=len, reverse=True)
    sentence2_copy=sentence2

    for word in ne_list:
        if word in sentence2_copy:
            sentence2_copy=re.sub(word,'',sentence2_copy)
        else:
            ne_list.remove(word)

    return ne_list



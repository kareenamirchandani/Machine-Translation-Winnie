import json

language_numbers = {"English": 0, "Luganda": 1, "Runyankole": 2, "Ateso": 3, "Lugbara": 4, "Acholi": 5}
new_name = 'sunbirdData_new.txt'

with open('sunbirdData.txt', 'r') as f_in, open(new_name, 'w') as f_out:
    for line in f_in:
        f_out.write(line[1:-2])
        f_out.write('\n')

with open('sunbirdData_new.txt', 'r') as file:
    filedata = file.read()

print(language_numbers["English"])

# Some of the sentences have commas in them which complicates things
# Maybe you can check if the comma follows an end quote before splitting
# Also need to get rid of annoying backslash character

filedata = filedata.replace('",', '"#')
filedata = filedata.replace("\\", "")

filedata = filedata.replace('"English":', '"language": 0, "sentence": ')
filedata = filedata.replace('"Runyankole":', '"language": 2, "sentence": ')
filedata = filedata.replace('"Luganda":', '"language": 1, "sentence": ')
filedata = filedata.replace('"Lugbara":', '"language": 4, "sentence": ')
filedata = filedata.replace('"Ateso":', '"language": 3, "sentence": ')
filedata = filedata.replace('"Acholi":', '"language": 5, "sentence": ')

filedata = filedata.split('#')

with open('sunbird_split.txt', 'w') as file:
    for line in filedata:
        file.write(line + '\n')

# Just need to figure out how to add '{}' to the start and end of each line. Done

with open('sunbird_split.txt', 'r') as f:
    lines = f.readlines()

lines = ["{%s}\n" % line.rstrip('\n') for line in lines]
# lines = ['{'+line + '}' for line in lines]
with open('sunbird_split.json', 'w') as f:
    f.writelines(lines)

# Took a longgg time to figure out.
# The reason why the json file wasn't loading is because some of the sentences have " in them,
# which was messing up the string

# https://www.geeksforgeeks.org/python-split-dictionary-keys-and-values-into-separate-lists/

with open('sunbirdData_copy.json', 'r') as f:
    data = [json.loads(line) for line in f]

dictionary_list = []
#print(data[0]['English'].split())
for x in range(len(data)):
    ini_dict = data[x]

    keys = []
    values = []

    for i in ini_dict:
        temp_words = ini_dict[i].split()
        #print(temp_words)
        for j in range(len(temp_words)):
            keys.append(i)
        for item in temp_words:
            values.append(item)

    #print(values)

    for x in range(len(keys)):
        # 1. for multilabel
        new_dict = {'English': int(keys[x]=='English'), 'Luganda': int(keys[x]=='Luganda'),
                    'Runyankole': int(keys[x]=='Runyankole'), 'Ateso': int(keys[x]=='Ateso'),
                    'Lugbara': int(keys[x]=='Lugbara'), 'Acholi': int(keys[x]=='Acholi'), 'sentence': values[x]}

        #new_dict = {'language': keys[x], 'sentence': values[x]}
        dictionary_list.append(new_dict)

print(dictionary_list[0:10])

with open('../sunbird_split_word.json', 'w') as fout:
    json.dump(dictionary_list, fout)

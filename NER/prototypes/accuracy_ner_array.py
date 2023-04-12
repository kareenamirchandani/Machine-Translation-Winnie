import pyarrow as pa
from datasets import load_dataset
import numpy as np

def in_memory_arrow_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table

def memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    memory_mapped_stream = pa.memory_map(filename)
    opened_stream = pa.ipc.open_stream(memory_mapped_stream)
    pa_table = opened_stream.read_all()
    return pa_table

#dataset_array=memory_mapped_arrow_table_from_file('dataset.arrow')

dataset_array=load_dataset('Sunbird/salt-dataset')
dataset_array=np.asarray(dataset_array)

import string

ne_set=set()

for i in range (len(dataset_array[0])):
  ne_sent_set = set(dataset_array[0][i].translate(str.maketrans('', '', string.punctuation)).split()).intersection(set(dataset_array[1][i].translate(str.maketrans('', '', string.punctuation)).split()))

  ne=0
  ne_index=-1
  ne_list=[]
  for word in dataset_array[0][i].translate(str.maketrans('', '', string.punctuation)).split():
    if word in ne_sent_set:
      if ne==1:
        ne_list[ne_index]=ne_list[ne_index]+" "+word
      else:
        ne_list.append(word)
        ne_index+=1
      ne=1
    else:
      ne=0
  ne_set=ne_set.union(set(ne_list))


print(ne_set)
# Import necessary libraries
from numpy import savetxt


ne_array=np.asarray(list(ne_set))
print(ne_array)
print(type(ne_array))
print(np.size(ne_array))


# save to txt file
savetxt('ner_accuracy_dataset.txt', ne_array, fmt='%s', encoding='utf-8')


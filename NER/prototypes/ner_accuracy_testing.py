import pyarrow as pa

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

dataset_array=memory_mapped_arrow_table_from_file('mt560_test.arrow')
import numpy as np
dataset_array=np.asarray(dataset_array)

swa_dataset = np.asarray([])
eng_dataset = np.asarray([])
a=0
i=0

while a<25006:
    if dataset_array[2][i] == 'swa':
        swa_dataset = np.append(swa_dataset, dataset_array[0][i])
        eng_dataset = np.append(eng_dataset, dataset_array[1][i])
        a+=1
    i+=1

print(len(swa_dataset))

# save to txt file
from numpy import savetxt
savetxt('mt560_swa_dataset.txt', swa_dataset, fmt='%s', encoding='utf-8')
savetxt('mt560_eng_dataset.txt', eng_dataset, fmt='%s', encoding='utf-8')


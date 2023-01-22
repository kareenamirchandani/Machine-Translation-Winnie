# Convert the sunbird Dataset into a csv file

import csv

new_name = 'sunbirdData_new.txt'

with open('sunbirdData.txt', 'r') as f_in, open(new_name, 'w') as f_out:
    for line in f_in:
        f_out.write(line[1:-2])
        f_out.write('\n')

with open('sunbirdData_new.txt','r') as file:
    filedata = file.read()

filedata = filedata.replace('"English":','')
filedata = filedata.replace('"Runyankole":','')
filedata = filedata.replace('"Luganda":','')
filedata = filedata.replace('"Lugbara":','')
filedata = filedata.replace('"Ateso":','')
filedata = filedata.replace('"Acholi":','')

with open('sunbirdData_new.txt','w') as file:
    file.write(filedata)

with open('sunbirdData_new.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('sunbirdData_new.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('English', 'Luganda', 'Runyankole','Ateso','Lugbara','Acholi'))
        writer.writerows(lines)


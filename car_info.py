import os

file_list=os.listdir(r'./combinations/class')

f = open("car_msrp.txt", "a")
file = open("car_msrp.txt")
for count, item in enumerate(file_list):
    msrp = file_list[count].split('_')
    msrp = msrp[3]
    if(msrp in file.read()):
        pass
    else:
        f.write('"' + msrp + '", ' + "\n")

# lines_seen = set() # holds lines already seen
# outfile = open("car_msrp_2.txt", "w")
# for line in open("car_msrp.txt", "r"):
#     if line not in lines_seen: # not a duplicate
#         outfile.write('"' + line + '", ')
#         lines_seen.add(line)
# outfile.close()

a_file = open("car_msrp.txt", "r")
string_without_line_breaks = ""
for line in a_file:
  stripped_line = line.rstrip()
  string_without_line_breaks += stripped_line
  thirdfile = open("car_msrp_3.txt", "w")
  thirdfile.write(string_without_line_breaks)
a_file.close()
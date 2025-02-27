# --- Opening a File: ---

File_object = open(r"File_Name","Access_Mode")

# Open function to open the file "MyFile1.txt"
# (same directory) in append mode and
file1 = open("MyFile1.txt","a")
# store its reference in the variable file1
# and "MyFile2.txt" in D:\Text in file2
file2 = open(r"D:\Text\MyFile2.txt","w+")

# --- Opening a File: ---

# Opening and Closing a file "MyFile.txt"
# for object name file1.
file1 = open("MyFile.txt","a")
file1.close()

# --- Writing to a file ---

File_object.write(str1)
File_object.writelines(L) for L = [str1, str2, str3]

# --- Reading from a file ---

File_object.read([n])
File_object.readline([n])
File_object.readlines()

# Program to show various ways to read and
# write data in a file.
file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"]
# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes
file1 = open("myfile.txt","r+")
print("Output of Read function is ")
print(file1.read())
print()
# seek(n) takes the file handle to the nth
# byte from the beginning.
file1.seek(0)
print( "Output of Readline function is ")
print(file1.readline())
print()
file1.seek(0)

# To show difference between read and readline
print("Output of Read(9) function is ")
print(file1.read(9))
print()
file1.seek(0)
print("Output of Readline(9) function is ")
print(file1.readline(9))
file1.seek(0)
# readlines function
print("Output of Readlines function is ")
print(file1.readlines())
print()
file1.close()

# --- Appending to a file ---

# Python program to illustrate
# Append vs write mode
file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"]
file1.writelines(L)
file1.close()
# Append-adds at last
file1 = open("myfile.txt","a")#append mode
file1.write("Today \n")
file1.close()
file1 = open("myfile.txt","r")
print("Output of Readlines after appending")
print(file1.readlines())
print()
file1.close()
# Write-Overwrites
file1 = open("myfile.txt","w")#write mode
file1.write("Tomorrow \n")
file1.close()
file1 = open("myfile.txt","r")
print("Output of Readlines after writing")
print(file1.readlines())
print()
file1.close()

# --- Write a CSV File ---

df.to_csv('data.csv')

# --- Read a CSV File ---

df = pd.read_csv('data.csv', index_col=0)
df

# --- Using pandas to Write and Read Excel Files ---

# pip install xlwt openpyxl xlsxwriter xlrd

# --- Write an Excel File ---

df.to_excel('data.xlsx')

# --- Read an Excel File ---

df = pd.read_excel('data.xlsx', index_col=0)
df

# --- Read an Excel File ---

df = pd.DataFrame(data=data).T
df.to_json('data-columns.json')

df.to_json('data-index.json', orient='index')

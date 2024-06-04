"""
open files in a few modes:
r = read
w = write
a = append
r+ = read and writing
"""
employee_file = open("Instructionals/Learn_Python/employees.txt","r")
print(employee_file.readable()) #readable function returns boolean indicating if a file is readable
print(employee_file.read()) #reads entire file
print(employee_file.readline()) #reads an individual line and moves cursor to next line
print(employee_file.readline()) #prints next line
print(employee_file.readlines())#takes each line and puts it in a list 

employee_file.close() #always close files

employee_file = open("Instructionals/Learn_Python/employees.txt","a")
employee_file.write("\nToby - Human Resources")
employee_file.close
"""
employee_file = open("Instructionals/Learn_Python/employees.txt","w") #writing will overwrite this text file
employee_file.write("\nToby - Human Resources")
employee_file.close
"""
employee_file = open("Instructionals/Learn_Python/employees1.txt","w") #writing with a file name that does not exist will create a new file
employee_file.write("\nToby - Human Resources")
employee_file.close
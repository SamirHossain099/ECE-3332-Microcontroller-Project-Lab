#User defined functions:
#The function keyword is def

def SayHi(name, age):
#the code in the function needs to be indented
    #here is in the function
#here is outside the function
    print("Hello "+name+", you are "+str(age))

#To call the function:
SayHi("Mike",35)
SayHi("Steve",70)

#python is a functional language so it is good to break apart the code into functions for each task of the system

def cube(num):
    return num*num*num
    print("This wont print")
    #the return keyword breaks out of the function
#unless you put the return statement, a value is not returned to the caller
print(cube(3))
result = cube(4)
print(result)

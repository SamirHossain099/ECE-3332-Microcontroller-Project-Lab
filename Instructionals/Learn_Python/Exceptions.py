#Exceptions in python are the same basic concept as Java
#python uses try except blocks
try:
    number = int(input("Enter a number: "))
    print(10/number)
#except: #this syntax will catch any error but we can also specify the error type for more tailored exceptions
except ZeroDivisionError as err:
    print(err)
except ValueError:
    print("Invalid Input")

#it is better to have specific exceptions instead of just the broad except statement.

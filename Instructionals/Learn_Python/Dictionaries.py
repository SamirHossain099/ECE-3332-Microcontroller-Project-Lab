#Dictionaries are a data structure in python for storing data in key value pairs. 
#In terms of a normal dictionary, think of the word as the key, and the definition as the value.
#They can hold any data type
#syntax:
# name = {key:value,key:value}
monthConversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December"
} 
#note each key must be unique
#accessing values:
print(monthConversions["Nov"]) #index by key
print(monthConversions.get("Luv","Not a valid key")) #.get can specify a default value if the key specified does exist.
#lists are great for organizing large amounts of data
#syntax is: name = [<values>]
#Lists can hold mixed data types, it does not have to be homogeneous

#Basics:

friends = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]
friends[1] = "Mike" #modifying values
print(friends) #whole list
print(friends[0]) #single index
print(friends[-1]) #indexing from the back of the list
print(friends[1:]) #indexing in range from 1 to end of list
print(friends[1:3]) #indexing in range from 1 up to but not including 3

#List Functions:
lucky_numbers = [42,81,5,16,23,42]
#extend function allows a list to be appending to another list
friends.extend(lucky_numbers)
print(friends)
#append function allows a new item to the list
friends.append("Tim")
print(friends)
#insert function appends by index and shifts all the proceeding indexs
friends.insert(1,"Kelly")
print(friends)
#remove function removes a specific item from the list
friends.remove("Jim")
print(friends)
#pop function removes the last element of the list
friends.pop()
print(friends)
#clear function clears the entire list
#friends.clear()
print(friends)
#index function searches list for an item and returns the index of the item
#index will cause an error if the element searched is not in the list
print(friends.index("Kelly"))
#count function tells how many times the given value appears in a list
print(friends.count("Kelly"))
#sort function will sort the list in ascending alphbetical and numerical order
#print(friends.sort()) can not sort when the list has two different data types
lucky_numbers.sort()
print(lucky_numbers)
#reverse function will reverse the order of the list, flips the element order
lucky_numbers.reverse()
print(lucky_numbers)
#copy function copies lists
friends2 = friends.copy()
print(friends2)

#2D lists:
number_grid = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]
#[row][column]
print(number_grid[0][2])
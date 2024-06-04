"""
Use the self keyword to reference to the object's self

"""
class Student:
    def __init__(self, name, major, gpa, is_on_probation): #basically constructor
        self.name = name
        self.major = major
        self.gpa = gpa
        self.is_on_probation = is_on_probation
    def on_honor_roll(self):
        if self.gpa >= 3.5:
            return True
        else:
            return False

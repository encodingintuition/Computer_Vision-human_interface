import turtle
import time



# setup the drawing environment
ts = turtle.Screen()
karen = turtle.Turtle()

# the following repeats 4 times
counter = 0
while counter < 300:
    with open('move-2.txt', 'r') as com:
        command = int(com.read())
    if command == 1:
        karen.pencolor('brown')
        karen.backward(77)
    elif command == 2:
        karen.pencolor('purple')
        karen.left(60)
        karen.forward(100)
    elif command == 3:
        karen.speed(10)
        karen.pencolor('blue')
        for x in range(5):
            karen.right(15)
            karen.forward(50)
            karen.left(90)
            karen.forward(30)
        karen.speed(3)
    elif command == 5:
        karen.pencolor('red')
        karen.forward(50)
        karen.left(33)
        karen.forward(11)
    elif command == 6:
        karen.pencolor('green')
        karen.right(35)
        karen.forward(23)
    elif command == 7:
        karen.speed(10)
        karen.pencolor('orange')
        for x in range(30):
            karen.forward(50)
            karen.left(170)
        karen.speed(3)   
    elif command == 8:
        karen.pencolor('black')
        karen.forward(120)
    karen.forward(50)
    karen.left(90)
    time.sleep(3)
    print(command)
    counter = counter + 1

ts.exitonclick()

class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def move(self):
        print(f"{self.brand} is moving")
class Car(Vehicle):
    def honk(self):
        print(f"{self.brand} goes Beep beep!")

if __name__ == "__main__":
    car = Car("Toyota")
    car.move()
    car.honk()
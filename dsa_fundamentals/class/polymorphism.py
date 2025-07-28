class Bird:
    def make_sound(self):
        print("Some generic bird sound")
class Sparrow(Bird):
    def make_sound(self):
        print("Chirp chirp")

class Parrot(Bird):
    def make_sound(self):
        print("Squawk")

def play_sound(bird:Bird):
    bird.make_sound()

if __name__ == "__main__":
    birds = [Sparrow(), Parrot()]
    for b in birds:
        play_sound(b)
import ray
ray.init(namespace="my_namespace")

# Define the Counter actor.
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create a Counter actor in a specific namespace.
c = Counter.options(name="teacher").remote()

input("Press any key to exit...")
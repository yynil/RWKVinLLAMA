import ray

# Connect to the same Ray cluster with the same namespace
ray.init(address='auto', namespace="my_namespace")

# Get a handle to the existing actor
c = ray.get_actor("teacher")

# Call methods on the actor
print(ray.get(c.increment.remote()))  # Increment and get the new value
print(ray.get(c.get_counter.remote()))  # Get the current counter value
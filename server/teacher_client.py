if __name__ == '__main__':
    import ray
    ray.init(address='auto', namespace="teacher_name_space")
    remote_teacher = ray.get_actor("teacher")
    obj_ref = remote_teacher.haha.remote()
    print(ray.get(obj_ref))
    import torch
    input_ids = torch.randint(1000, 12223,(4,2048))
    import time
    start = time.time()
    obj_ref = remote_teacher.compute_logits.remote(input_ids)
    end = time.time()
    print(f'remote call time: {end-start}')
    start = time.time()
    results = ray.get(obj_ref)
    end = time.time()
    print(f'get time: {end-start}')
    print(results.shape)
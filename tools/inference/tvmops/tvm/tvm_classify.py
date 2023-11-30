import time


def classify(pipe, *, module):

    total_time = 0
    
    for idx, item in enumerate(pipe):
        start = time.time()
        
        image = item['image']
        
        inputs = {'image': image}
        module.set_input(**inputs)
        module.run()

        num_outputs = module.get_num_outputs()
        outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            outputs[output_name] = module.get_output(i).numpy()
        
        item['preds'] = outputs['output_0']

        total_time += (time.time() - start)
        item['inference_time'] = total_time

        yield item


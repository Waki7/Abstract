import numpy as np
import torch

import settings


def main():
    i = 0
    while (True):
        nd = np.random.random_sample((20000, 10000))
        tensor = torch.tensor(nd).to(**settings.ARGS)
        # tensor = tensor.to(settings.DTYPE_X)

        i += 1
        print(i)


if __name__ == "__main__":
    main()


#
# Warning: Error detected in CudnnBatchNormBackward. Traceback of forward call that caused the error:
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 64, in <module>
#     main()
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 60, in main
#     train_state_encoder('mobile', 'grid')
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 36, in train_state_encoder
#     trainer.train(network, CFG_EXECUTION['encoder_training'])
#   File "C:\Users\Ceyer\Documents\Projects\Abstract\grid_world\encoder_trainers\grid_encoding_task.py", line 89, in train
#     out = encoder.forward(new_batch)
#   File "C:\Users\Ceyer\Documents\Projects\Abstract\networks\mobile_net_v2.py", line 182, in forward
#     x = self.features(x)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\container.py", line 100, in forward
#     input = module(input)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "C:\Users\Ceyer\Documents\Projects\Abstract\networks\mobile_net_v2.py", line 68, in forward
#     return self.conv(x)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\container.py", line 100, in forward
#     input = module(input)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\container.py", line 100, in forward
#     input = module(input)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\module.py", line 550, in __call__
#     result = self.forward(*input, **kwargs)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\modules\batchnorm.py", line 106, in forward
#     exponential_average_factor, self.eps)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\nn\functional.py", line 1923, in batch_norm
#     training, momentum, eps, torch.backends.cudnn.enabled
#  (print_stack at ..\torch\csrc\autograd\python_anomaly_mode.cpp:60)
# Traceback (most recent call last):
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 64, in <module>
#     main()
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 60, in main
#     train_state_encoder('mobile', 'grid')
#   File "C:/Users/Ceyer/Documents/Projects/Abstract/run_encoder_train.py", line 36, in train_state_encoder
#     trainer.train(network, CFG_EXECUTION['encoder_training'])
#   File "C:\Users\Ceyer\Documents\Projects\Abstract\grid_world\encoder_trainers\grid_encoding_task.py", line 94, in train
#     critic_loss.backward()
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\tensor.py", line 198, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph)
#   File "C:\Users\Ceyer\Anaconda3\env_variations\dev\lib\site-packages\torch\autograd\__init__.py", line 100, in backward
#     allow_unreachable=True)  # allow_unreachable flag
# RuntimeError: CUDA error: an illegal memory access was encountered (operator () at C:/w/b/windows/pytorch/aten/src/ATen/native/cuda/CUDAScalar.cu:19)
# (no backtrace available)
#
# Process finished with exit code 1

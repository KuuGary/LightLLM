# LightLLM

LightLLM is an project based on [LLaMA](https://github.com/facebookresearch/llama) and LoRA (PEFT). 

## Installation & Dependencies

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KuuGary/LightLLM.git
   cd LightLLM

2. **Create and activate a virtual environment (optional)**
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
4. **Obtain LLaMA model checkpoints under Meta’s license. Adjust paths in LightLLM.py accordingly. Alternatively, you can replace LLaMA with other models if you prefer.**


## Prepare the Dataset
Place or link your CSV files under ```dataset/.```，Update ```data_provider/data_factory.py``` if you want to load your custom dataset.

## Check Logs and Outputs
Training logs are saved in train_result.log by default (see run_main.py).

LoRA checkpoints will be placed in ./checkpoints/. Feel free to delete or keep them as needed.

## Citation
If you find LightLLM helpful in your research, please consider citing our work. 
```
@inproceedings{hu2025lightllm,
  title={Lightllm: A versatile large language model for predictive light sensing},
  author={Hu, Jiawei and Jia, Hong and Hassan, Mahbub and Yao, Lina and Kusy, Brano and Hu, Wen},
  booktitle={Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems},
  pages={158--171},
  year={2025}
}
```

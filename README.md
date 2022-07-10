## Fusing Global and Local Features for Generalized AI-Synthesized Image Detection [[Paper Link](https://arxiv.org/pdf/2203.13964.pdf)] 

### [Yan Ju](https://littlejuyan.github.io/), [Shan Jia](https://shanface33.github.io/), [Lipeng Ke](https://scholar.google.com/citations?hl=en&user=qzlM2bMAAAAJ&view_op=list_works&sortby=pubdate), [Hongfei Xue](http://havocfixer.github.io/), [Koki Nagano](https://luminohope.org/), and [Siwei Lyu](https://cse.buffalo.edu/~siweilyu/). In [ICIP, 2022](https://cmsworkshops.com/ICIP2022/papers/accepted_papers.php).


## 1. Setup

Install packages: `pip install -r requirements.txt`

## 2. Dataset
###Training and validation dataset
We used the same training dataset in the paper "CNN-generated images are surprisingly easy to spot...for now". The training and validation dataset can be downloaded from [their repository](https://github.com/peterwang512/CNNDetection). The dataset should be arranged as this:
	
	Training and validation dataset
		|- train(90% of downloaded dataset)
			|_ 0_real
				xxxx.png
				...
			|_ 1_fake
				yyyy.png
				...
		|- val(10% of downloaded dataset)
			|_ 0_real
				xxxx.png
				...
			|_ 1_fake
				yyyy.png
				...
	
### Testing dataset

For the testing dataset, we composed a dataset of synthetic images generated with 19 various generation models based on several existing datasets such as [CNNDetection, Sheng-Yu Wang, et al., CVPR2020](https://github.com/peterwang512/CNNDetection), [Reverse_Engineering_GMs, Vishal Asnani, et al., ](https://github.com/vishal3477/Reverse_Engineering_GMs) and [Celeb-DF, Yuezun Li, et al., CVPR2020](https://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html). Besides, we also collect several models-generated datasets from SemaFor program, such as [StyleGAN3](https://github.com/NVlabs/stylegan3), [Taming Transformers](https://github.com/CompVis/taming-transformers), [BGM](https://github.com/ZHKKKe/MODNet), etc. 

Before testing the model, please arrange the testing dataset as following:
	
	Testing dataset
		|- Generation Model 1
			|_ 0_real
				xxxx.png
				...
			|_ 1_fake
				yyyy.png
				...
		|- Generation Model 2
			|_ 0_real
				xxxx.png
				...
			|_ 1_fake
				yyyy.png
				...
		|- Generation Model ...	


## Train the model

We provide an example script to train our model by running `bash train.sh`, in which you can change the following parameters:

`--name`: the directory name you want to save your checkpoints in.

`--blur_prob`: the probability of the image processed with Gaussian blur.

`--blur_sig`: the Gaussian blur parameter σ

`--jpg_prob`: the probability of the image processed with JPEG compression.

`--jpg_method`: compression method, cv2 or pil.
  
`--jpg_qual`: JPEG compression quality parameter.
  
`--dataroot`: path of training and validation datasets.
  
## Test the model
			
We provide an example script to test our model by running `bash test.sh`. 


## Acknowledgments
This repository borrows partially from [this work](https://github.com/peterwang512/CNNDetection).

This work is supported by the US Defense Advanced Research Projects Agency (DARPA) Semantic Forensic (SemaFor) program. We thank SemaFor TA4 teams and previous works for providing datasets for our training and testing. 

## Citation
If you find this useful for your research, please consider citing this [bibtex](https://github.com/littlejuyan/FusingGlobalandLocal/blob/main/bibtex.txt). 
	
